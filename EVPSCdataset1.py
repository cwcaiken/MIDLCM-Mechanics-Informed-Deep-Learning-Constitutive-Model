import torch
from torch.utils.data import Dataset
import numpy as np
import os
import joblib
from scipy.interpolate import interp1d

class EVPSCDataset(Dataset):
    def __init__(self, path,  train=True):
       
        self.path = path
        self.train = train
       
        #### 加载仿真数据
        self.x_train = joblib.load('x_train')
        self.micro_train = joblib.load('micro_train')
        self.stress_train = joblib.load('stress_train')
        self.c_train = joblib.load('c_train')

        self.c_test = joblib.load('c_test')
        self.x_test = joblib.load('x_test')
        self.stress_test = joblib.load('stress_test')
        self.micro_test = joblib.load('micro_test')


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