import os

import numpy as np
import torch

os.makedirs(os.path.join('.','data'),exist_ok=True)
data_file=os.path.join('.','data','house_tiny')
print(data_file)
with open(data_file,"w") as f:
    f.write('A,B,C,D,E,F\n')  # 列名
    f.write('NA,Pave,127500,1,2,3\n')  # 每行表示一个数据样本
    f.write('NA,Pave,106000,6,8,9\n')
    f.write('4,NA,178100,45,67,8\n')
    f.write('NA,NA,140000,86,54,33\n')

import pandas as pd
data=pd.read_csv(data_file)
ss=data.isnull().sum()

max_data=0
max_index=0

for i,d in ss.items():
    if d>max_data:
        max_data=d
        max_index=i

print(max_index,max_data)

data=data.drop(max_index,axis=1)

data=pd.get_dummies(data,dummy_na=True)

print(data)

data=torch.tensor(data.to_numpy(dtype=float))

print(data,data.type())
