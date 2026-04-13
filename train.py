import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LightEVPSCPredictor
from EVPSCdataset1 import EVPSCDataset
import matplotlib.pylab as plt
import numpy as np
import torch.nn.functional as F
import os


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EVPSCTrainer:
    def __init__(self, train_dataloader, val_dataloader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = 3000
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        # 损失函数定义
        self.stress_criterion = nn.MSELoss()
        self.dislocation_criterion = nn.MSELoss()
        self.twin_criterion = nn.MSELoss()


    def time_consistency_loss(self, predictions):
        """W
        计算时间一致性损失
        Args:
            pWredictions: 预测值，形状为 (batch_size, time_steps, num_features)
        Returns:
            loss: 时间一致性损失
        """
        # 计算相邻时间步之间的差异
        diff = predictions[:, 1:, :] - predictions[:, :-1, :]  # 形状为 (batch_size, time_steps-1, num_features)
        
        # 计算L2范数（均方误差）
        loss = torch.mean(diff**2)  # 对所有时间步和特征取均值
        
        return loss

    
    def physics_loss(self, pred, target, strain):
        """
        针对归一化数据的物理损失函数
        """
        # 获取物理层参数
        physics_layer = self.model.physics_layer
        elastic_moduli = physics_layer.E
        yield_stress = physics_layer.sigma_y0
        hardening_param = physics_layer.H
        # 1. 应力预测损失 - 使用归一化友好的损失函数
        #stress_loss = nn.functional.mse_loss(pred, target)
        
        # 2. 弹性区约束损失（考虑归一化尺度）
        elastic_mask = (strain[:, :, :3].abs() <= yield_stress.unsqueeze(0).unsqueeze(0)).float()
        elastic_loss = torch.mean(
            torch.pow(pred[:,:,:3] - elastic_moduli.unsqueeze(0).unsqueeze(0) * strain[:, :, :3], 2) * elastic_mask
        )
        
        # 3. 屈服条件损失（针对归一化数据）
        yield_condition_loss = torch.mean(
            torch.clamp(
                torch.abs(pred[:,:,:3]) - (yield_stress.unsqueeze(0).unsqueeze(0) + 
                hardening_param.unsqueeze(0).unsqueeze(0) * strain[:,:,:3]), 
                min=0
            ) ** 2
        )
        
        # 总损失，调整权重以适应归一化数据
        total_loss = (
           # stress_loss + 
            0.01 * elastic_loss + 
            0.01 * yield_condition_loss
        )
        
        return total_loss



    def custom_loss(self, pred_stress, true_stress, 
                    pred_dislocation, true_dislocation,
                    pred_twin, true_twin, strain):
        """
        自定义多目标耦合损失函数
        """
        stress_loss = self.stress_criterion(pred_stress, true_stress) #+ self.time_consistency_loss(pred_stress)
        dislocation_loss = self.dislocation_criterion(pred_dislocation, true_dislocation)# + self.time_consistency_loss(pred_dislocation)
        twin_loss = self.twin_criterion(pred_twin, true_twin)# + self.time_consistency_loss(pred_twin)
        physics_loss = self.physics_loss(pred_stress, true_stress, strain)
        
        total_loss = dislocation_loss + twin_loss + stress_loss #+  physics_loss 
        
        return total_loss


    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            # 数据预处理
            
            inputs = batch['input'].to(self.device)
            rate = batch['rate'].to(self.device)
          
            stress_target = batch['stress'].to(self.device)
            dislocation_target = batch['micro'][:,:,0].unsqueeze(-1).to(self.device)
            twin_target = batch['micro'][:,:,1].unsqueeze(-1).to(self.device)
                
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 模型前向传播
            pred_micro, pred_stress = self.model(inputs,rate)
            pred_dislocation = pred_micro[:,:, 0].unsqueeze(-1)
            pred_twin = pred_micro[:,:, 1].unsqueeze(-1)

            # 计算损失

            loss = self.custom_loss(
                pred_stress, stress_target,
                pred_dislocation, dislocation_target, 
                pred_twin, twin_target, inputs
            )
            
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # 参数更新
            self.optimizer.step()

            total_loss += loss.item()
        
        return total_loss / len(dataloader)

    def init_weights(self, model):
            for name, param in model.named_parameters():
                if 'weight' in name:
                    # Xavier初始化适用于tanh激活函数
                    if 'gru' in name:
                        torch.nn.init.xavier_uniform_(param)
                    # Transformer使用xavier_normal_或kaiming初始化
                    elif 'transformer' in name:
                        if len(param.shape)<2:
                            torch.nn.init.xavier_uniform_(param.unsqueeze(0))
                        else:
                            torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)


    def train(self, hidden_dim, num_layers, embedding_dim,num_heads,fc_dim):

        # 初始化模型和其他组件
        self.model = LightEVPSCPredictor(hidden_dim, num_layers, embedding_dim, num_heads, fc_dim).to(self.device)
        # 计算参数量
        total_params = count_parameters(self.model)
        print(f"Total trainable parameters: {total_params}")
        # 初始化权重
        self.model.apply(self.init_weights)
        #self.model.load_state_dict(torch.load('best_model.pth'))

        # 设置优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-3,              # 学习率
            betas=(0.9, 0.999),   # beta参数
            eps=1e-8,             # epsilon值
            weight_decay=0.01     # 权重衰减
        )

        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.epochs * 0.1,  # 10%的步数用于预热
                num_training_steps=self.epochs
            )
                
        best_val_loss = float('inf')
        self.train_losses = []    #保存当前epoch的loss
        self.val_losses = []
        for epoch in range(self.epochs):
            # 训练
            train_loss = self.train_epoch(self.train_dataloader)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, mape,mae = self.validate(self.val_dataloader)
            self.val_losses.append(val_loss)

            # 学习率调度
            self.scheduler.step()
            
            # 模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_mape = mape
                best_val_mae = mae
                torch.save(self.model, 'models/' + str(hidden_dim) + str(num_layers) + str(embedding_dim) + str(num_heads) + str(fc_dim)+ 'model_best_complete.pth')
            
            #print(f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

        torch.save(self.model, 'models/' + str(hidden_dim) + str(num_layers) + str(embedding_dim) + str(num_heads) +str(fc_dim)+  'model_complete.pth')
        return best_val_loss, best_val_mape, best_val_mae
    
    def validate(self, dataloader):
        self.model.eval()
        total_val_loss = 0
        total_mape = 0
        total_mae = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                rate = batch['rate'].to(self.device)
                
                stress_target = batch['stress'].to(self.device)
                dislocation_target = batch['micro'][:,:,0].unsqueeze(-1).to(self.device)
                twin_target = batch['micro'][:,:,1].unsqueeze(-1).to(self.device)
                
                # 模型前向传播
                #import time
                #t = time.time()
                pred_micro, pred_stress = self.model(inputs,rate)
                #print(time.time()-t)
                pred_dislocation = pred_micro[:,:, 0].unsqueeze(-1)
                pred_twin = pred_micro[:,:, 1].unsqueeze(-1)

                val_loss = self.stress_criterion(pred_stress, stress_target) \
                            + self.dislocation_criterion(pred_dislocation, dislocation_target) + \
                            self.twin_criterion(pred_twin, twin_target) 

                total_val_loss += val_loss.item()

                mape = mean_absolute_percentage_error(pred_stress.cpu().numpy(), stress_target.cpu().numpy()) + \
                mean_absolute_percentage_error(pred_dislocation.cpu().numpy(), dislocation_target.cpu().numpy()) + \
                mean_absolute_percentage_error(pred_twin.cpu().numpy(), twin_target.cpu().numpy()) 
                total_mape += mape

                mae = mean_absolute_error(pred_stress.cpu().numpy(), stress_target.cpu().numpy()) + \
                mean_absolute_error(pred_dislocation.cpu().numpy(), dislocation_target.cpu().numpy()) + \
                mean_absolute_error(pred_twin.cpu().numpy(), twin_target.cpu().numpy()) 
                total_mae += mae
        
        return total_val_loss / len(dataloader), total_mape/ len(dataloader),total_mae / len(dataloader)


if __name__=="__main__":
        
    # 创建数据加载器
    path =  'D:/data/01_data/data/'
    train_dataset = EVPSCDataset(path = path,  train=True)
    val_dataset = EVPSCDataset(path = path,  train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型和优化器
    trainer = EVPSCTrainer(train_loader, val_loader)
    # 训练模型 评估模型
    test_loss = trainer.train(hidden_dim=256,num_layers=2, 
                            embedding_dim=2,num_heads=4,
                            fc_dim=128)
    