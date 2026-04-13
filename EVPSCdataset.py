import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d

class EVPSCDataset(Dataset):
    def __init__(self, path,  train=True):
       
        self.path = path
        self.train = train
       
        #### 加载仿真数据
        stresses = []
        strains = []
        densitys = []
        st_rates = []
        rate_map = {'1e-03':np.log(1e-3),
                    '1e-01':np.log(1e-1),
                    '1e+01':np.log(1e1),
                    '1e+03':np.log(1e3),
                    '3e+03':np.log(3e3),
                    '5e+03':np.log(5e3),
                    '4000':np.log(4e3),
                    '2000':np.log(2e3),
                    }
        
        files = os.listdir(path)
        for i in range(len(files)):
            file_path = path + files[i] + '/str_str.csv'
            density_path = path + files[i] + '/Density.csv'
            # EVM = np.squeeze(pd.read_csv(file_path, usecols=[0]).values[:-3, :])
            # ss1 = pd.read_csv(file_path, usecols=[11, 12, 13]).values[:-3, :]
            # st1 = pd.read_csv(file_path, usecols=[5, 6, 7,8,9,10]).values[:-3, :]
            # density1 = pd.read_csv(density_path, usecols=[1, 2]).values[:-3, :]
            if files[i].split('_')[-1] == '2000':
                EVM = np.squeeze(pd.read_csv(file_path, usecols=[0]).values[:-3, :])
                ss1 = pd.read_csv(file_path, usecols=[8, 9, 10]).values[:-3, :]
                st1 = pd.read_csv(file_path, usecols=[2, 3, 4,5,6,7]).values[:-3, :]
                density1 = pd.read_csv(density_path, usecols=[1, 2]).values[:-3, :]
            elif files[i].split('_')[-1] == '4000':
                EVM = np.squeeze(pd.read_csv(file_path, usecols=[0]).values[:-3, :])
                ss1 = pd.read_csv(file_path, usecols=[8, 9, 10]).values[:-3, :]
                st1 = pd.read_csv(file_path, usecols=[2, 3, 4,5,6,7]).values[:-3, :]
                density1 = pd.read_csv(density_path, usecols=[1, 2]).values[:-3, :]
            else:
                EVM = np.squeeze(pd.read_csv(file_path, usecols=[0]).values[:-3, :])
                ss1 = pd.read_csv(file_path, usecols=[11, 12, 13]).values[:-3, :]
                st1 = pd.read_csv(file_path, usecols=[5, 6, 7,8,9,10]).values[:-3, :]
                density1 = pd.read_csv(density_path, usecols=[1, 2]).values[:-3, :]
            for j in range(density1.shape[0]):
                if density1[j, 0] != 0:
                    density1[j, 0] = np.log(density1[j, 0])
            
            xnew = np.linspace(0, 0.16, 201)
            ss2 = np.zeros(shape=(201, ss1.shape[1]))
            st2 = np.zeros(shape=(201, st1.shape[1]))
            density2 = np.zeros(shape=(201, density1.shape[1]))
            
            if np.max(EVM) >= 0.16:
                for k in range(ss1.shape[1]):
                    f = interp1d(EVM, ss1[:,k],kind='linear')
                    ss2[:,k] = f(xnew)
           
                for k in range(st1.shape[1]):
                    f = interp1d(EVM, st1[:,k],kind='linear')
                    st2[:,k] = f(xnew)

                for k in range(density1.shape[1]):
                    f = interp1d(EVM, density1[:,k], kind='linear')
                    density2[:,k] = f(xnew)

            if np.max(EVM) < 0.16:
                for k in range(ss1.shape[1]):
                    f = interp1d(EVM, ss1[:,k],kind='linear', fill_value='extrapolate')
                    ss2[:,k] = f(xnew)
                for k in range(st1.shape[1]):
                    f = interp1d(EVM, st1[:,k],kind='linear', fill_value='extrapolate')
                    st2[:,k] = f(xnew)
                for k in range(density1.shape[1]):
                    f = interp1d(EVM, density1[:,k], kind='linear', fill_value='extrapolate')
                    density2[:,k] = f(xnew)
                
            # s1 = ss2[2:21, :]
            # s2 = ss2[21::2, :]
            # ss = np.concatenate((s1,s2),axis=0)

            # s1 = st2[2:21, :]
            # s2 = st2[21::2, :]
            # st = np.concatenate((s1,s2),axis=0)

            # d1 = density2[2:21, :]
            # d2 = density2[21::2, :]
            # density = np.concatenate((d1,d2),axis=0)

            ss = ss2[1::2, :]
            st = st2[1::2, :]
            density = density2[1::2, :]
            rate = rate_map[files[i].split('_')[-1]]
            stresses.append(ss)
            strains.append(st)
            densitys.append(density)
            st_rates.append(rate)

        st_rates = np.array(st_rates)
        strains = np.array(strains)
        stresses = np.array(stresses)
        densitys = np.array(densitys)
        dislocation = np.expand_dims(densitys[:,:,0],-1)
        twin = np.expand_dims(densitys[:,:,1], -1)
       
        #### 数据归一化
        self.sc_st = StandardScaler(with_mean=False)
        strain_n = self.sc_st.fit_transform(strains.reshape(-1,1))
        strain_n = strain_n.reshape(strains.shape)
        
        self.sr_s = StandardScaler(with_mean=False)
        sr_n = self.sr_s.fit_transform(st_rates.reshape(-1, 1))
        sr_n = sr_n.reshape(st_rates.shape)

        self.sc_s = StandardScaler(with_mean=False)
        stress_n = self.sc_s.fit_transform(stresses.reshape(-1, 1))
        stress_n = stress_n.reshape(stresses.shape)
       
        self.dis_s = StandardScaler(with_mean=False)
        #self.dis_s = MinMaxScaler()
        dis_n = self.dis_s.fit_transform(dislocation.reshape(-1, 1))  
        dis_n = dis_n.reshape(dislocation.shape)

        self.tw_s = StandardScaler(with_mean=False)
        #self.tw_s = MinMaxScaler()
        tw_n = self.tw_s.fit_transform(twin.reshape(-1, 1))
        tw_n = tw_n.reshape(twin.shape)

        x_data = strain_n
        constant_data = sr_n
        micro_data = np.concatenate((dis_n, tw_n), axis=2)
        stress_data = stress_n
        #y = np.concatenate( (np.concatenate((stress_n, dis_n), axis=2), tw_n), axis=2)

        """保存所有标准化器"""
        joblib.dump(self.sc_st, 'strain_scaler.pkl')
        joblib.dump(self.sr_s, 'strain_rate_scaler.pkl')
        joblib.dump(self.sc_s, 'stress_scaler.pkl')
        joblib.dump(self.dis_s, 'dislocation_scaler.pkl')
        joblib.dump(self.tw_s, 'twin_scaler.pkl')
    
        
        # 数据集划分比例  
        x_train, x_test, c_train, c_test, micro_train, micro_test, stress_train, stress_test = train_test_split(x_data, constant_data, micro_data, stress_data,test_size=0.2, random_state=1)
        print(x_train.shape)
        print(x_test.shape)
        print(c_train.shape)
        print(c_test)
        print(micro_train.shape)
        print(micro_test.shape)
        print(stress_train.shape)
        print(stress_test.shape)
        
        # 转换为PyTorch张量
        self.x_train = torch.FloatTensor(x_train)
        self.x_test = torch.FloatTensor(x_test)
        self.c_train = torch.FloatTensor(c_train)
        self.c_test = torch.FloatTensor(c_test)
        self.micro_train = torch.FloatTensor(micro_train)
        self.micro_test = torch.FloatTensor(micro_test)
        self.stress_train = torch.FloatTensor(stress_train)
        self.stress_test = torch.FloatTensor(stress_test)

        joblib.dump(self.x_train, 'x_train')
        joblib.dump(self.micro_train, 'micro_train')
        joblib.dump(self.stress_train, 'stress_train')
        joblib.dump(self.c_train, 'c_train')

        joblib.dump(self.c_test, 'c_test')
        joblib.dump(self.x_test, 'x_test')
        joblib.dump(self.stress_test, 'stress_test')
        joblib.dump(self.micro_test, 'micro_test')


    def __len__(self):
        if self.train==True:
            return self.x_train.shape[0]
        
        if self.train == False:
            return self.x_test.shape[0]


    def __getitem__(self, idx):
        if self.train==True:
            sample = {
                'input': self.x_train[idx],
                'rate':self.c_train[idx],
                'stress': self.stress_train[idx],
                'micro': self.micro_train[idx]
            }
        if self.train == False:
            sample = {
                'input': self.x_test[idx],
                'rate':self.c_test[idx],
                'stress': self.stress_test[idx],
                'micro':self.micro_test[idx]
            }

        return sample

# 使用示例
if __name__ == "__main__":
    # 创建训练集和验证集
    path =  'D:/data/01_data/data/'
    train_dataset = EVPSCDataset(path = path, train=True)
    val_dataset = EVPSCDataset(path = path,  train=False)
    
    # 检查数据形状
    sample = train_dataset[90]
    print("Sample shapes:")
    print(f"Input shape: {sample['input'].shape}")       
    print(f"micro shape: {sample['micro'].shape}") 
    print(f"stress shape: {sample['stress'].shape}")    
    print(f"rate shape: {sample['rate'].shape}")  
    print(sample['rate'])    