import torch
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PhysicsInformedLayer(nn.Module):
    def __init__(self, E=1, nu=0.3, sigma_y0=1, H=2):
        super().__init__()
        # 材料参数（可训练）
        self.E = nn.Parameter(torch.tensor(E, dtype=torch.float32))# 弹性模量
        self.nu = nn.Parameter(torch.tensor(nu, dtype=torch.float32))# 泊松比
        self.sigma_y0 = nn.Parameter(torch.tensor(sigma_y0, dtype=torch.float32))# 初始屈服应力
        self.H = nn.Parameter(torch.tensor(H, dtype=torch.float32))# 硬化模量
        
    def forward(self, eps):
        # 计算弹性常数（每次前向传播动态计算）
        G = self.E / (2 * (1 + self.nu)) # 剪切模量
        lambda_ = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)) # 拉梅常数
        
        # 弹性预测
        trace_eps = torch.sum(eps, dim=-1, keepdim=True)
        sigma_trial = 2 * G * eps + lambda_ * trace_eps
        
        # 屈服判断与塑性修正
        s_trial = sigma_trial - torch.mean(sigma_trial, dim=-1, keepdim=True)
        seq_trial = torch.sqrt(1.5 * torch.sum(s_trial**2, dim=-1, keepdim=True))
        
        phi = seq_trial - self.sigma_y0
        is_plastic = torch.sigmoid(10 * phi)  # 软化屈服条件
        
        delta_ep = torch.maximum(phi / (3 * G + self.H), torch.zeros_like(phi))
        delta_ep *= is_plastic
        
        n = s_trial / (seq_trial + 1e-8)
        sigma = sigma_trial - 2 * G * delta_ep * n
        
        return sigma  # 输出主应力分量


class LightEVPSCPredictor(nn.Module):
    def __init__(self, hidden_dim, num_layers, embedding_dim, num_heads, fc_dim):
        super(LightEVPSCPredictor, self).__init__()

        self.input_dim = 6
        self.constant_dim = 1
        self.stress_dim = 3
        self.micro_dim = 2

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.fc_dim = fc_dim

        self.fc_constant = nn.Linear(self.constant_dim, self.embedding_dim)  # 一维常量映射到x维
        self.gru = nn.GRU(self.input_dim + self.embedding_dim, self.hidden_dim, num_layers=self.num_layers,batch_first=True)
       # self.gru2 = nn.GRU(self.micro_dim + self.input_dim, self.fc_dim, batch_first=True)
        self.gru2 = nn.GRU(self.hidden_dim, self.fc_dim, batch_first=True)
        self.ffn = nn.Linear(self.fc_dim, self.stress_dim+self.micro_dim)
        

        self.attention = nn.MultiheadAttention(self.hidden_dim, num_heads=self.num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

        # 物理知识引导的输出层
        self.physics_layer = PhysicsInformedLayer()
        
         # 预测微观组织
        self.micro_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.micro_dim),
        )

        # 预测应力
        self.stress_net = nn.Sequential(
            nn.Linear(self.micro_dim+self.input_dim, self.fc_dim),
            nn.ReLU(),
            nn.Linear(self.fc_dim, self.stress_dim)
        )
    
    

    def forward(self, x, constant, return_attention=False):
        # x shape: (batch_size, 100, 6)
        # constant shape: (batch_size,)
        batch_size, seq_len, _ = x.shape
        # 处理常量输入
        const_mapped = self.fc_constant(constant.unsqueeze(-1))  # (batch_size, x)
        const_mapped = const_mapped.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, 100, x)
        # 合并输入
        combined = torch.cat([x, const_mapped], dim=-1)
         # GRU处理
        features, _ = self.gru(combined)

        # 注意力机制
        attn_output, attn_weights = self.attention(features, features, features)
        features = features + attn_output
        features = self.layer_norm(features)
        
        stress,_ = self.gru2(features)
        adjustment = self.ffn(stress)
        micro = adjustment[:,:,3:]
        output = adjustment[:,:,:3]
        
        # 微观组织预测
        #micro = self.micro_net(features)
        # 位错开根号
        #m = torch.sqrt(micro)
        # 物理模型预测
        #physics_stress = self.physics_layer(x[:,:,:3])
        # 网络调整项
        #micro_mix = torch.cat([micro, x], dim=-1)
        #adjustment = self.stress_net(micro_mix)
        #stress,_ = self.gru2(micro_mix)
       #adjustment = self.ffn(stress)
        
        #output = adjustment #+ physics_stress

        if return_attention:
            return micro, output, attn_weights
        else:
            return micro, output
        


