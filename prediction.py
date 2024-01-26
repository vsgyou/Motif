#%%
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import random
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.adam import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
#%%
from preprocess import *
from model import *
#%%
class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride = 1):
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samles = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data).view(-1,1)

        X = torch.zeros(num_samles, input_window, input_size)
        y = torch.zeros(num_samles, output_window)

        for i in range(num_samles):
            X[i,:] = data_tensor[i*stride : i*stride + input_window]
            y[i,:] = data_tensor[i*stride + input_window : i *stride+self.seq_len]
        self.x = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len

# Data
stock_code = '005930.KS'
start_date = '2020-07-31'
end_date = '2023-07-31'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
#%%
train_set, test_set = split_data(data = close, input_window = 7, test_len = 80)
train_set = windowDataset(data = train_set, input_window = 7, output_window = 1, input_size = 1, stride = 1)
num_samples = train_set.x.shape[0]
# train_x = train_set[:][0]
# train_y = train_set[:][1]

train_set_org = copy.deepcopy(train_set)
bound,train_trans = pred_minmax(train_set)   # sequence별 max,min저장
k = 3

x_pattern = torch.zeros([train_trans.x.shape[0],train_trans.x.shape[1],k])
x_pattern_noscale = torch.zeros([train_set_org.x.shape[0],train_set_org.x.shape[1],k]) 
bound_list = torch.zeros([train_trans.x.shape[0],k+1,2])

for i in range(num_samples):

    dist = pd.DataFrame(data = torch.sum(torch.sqrt((train_trans.x[i] - train_trans.x)**2), 
    dim = (1,2)), columns = ['Values'])
    dist_sort = dist.sort_values(by = 'Values')
    ind = dist_sort.index[1:k+1].tolist()

    x_pattern[i,:] = torch.cat([train_trans.x[j] for j in ind],dim = 1)   #[num_sample, seq_len, k]
    x_pattern_noscale[i,:] = torch.cat([train_set_org.x[j] for j in ind], dim=1)
    bound_list[i,:] = torch.cat((torch.FloatTensor(bound[i]).unsqueeze(0),
                                 torch.FloatTensor([bound[j] for j in ind])),
                                 dim=0)    # [num_sample, k+1, 2] 기존,패턴의 min,max 저장
torch.FloatTensor(bound)
#%%
train_set_org.x = torch.cat([train_set_org.x,x_pattern_noscale],dim = 2)    # minmax 안된 x
# scaling을 거치고 찾은 패턴을 찾아 변환되기 전의 값으로 합침
# train_set_org scaling
train_set_org.x = train_set_org.x - 50000
train_set_org.y = train_set_org.y - 50000








#%%

train_trans.x = torch.cat([train_trans.x, x_pattern], dim = 2)





#%%
# 데이터 로더
train_loader = DataLoader(train_trans, batch_size = 64, shuffle = False, drop_last = True)
loader = DataLoader(train_trans, batch_size = 1, shuffle = False, drop_last = True)
#%%
def train(model, data_loader, optimizer, criterion):
    model.train()
    h0 = torch.zeros(1,64,8)
    total_loss = []
    for input,label in data_loader:
        
        input = input
        label = label

        optimizer.zero_grad()

        pred = model(input, h0)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    return sum(total_loss)/len(total_loss), pred
#%%
# 모델 학습
model = RNN(input_size = 4, hidden_size = 8, num_layers = 1)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()

train(model, data_loader = train_loader, optimizer = optimizer, criterion=criterion)

for epoch in range(200):
    train_loss,pred = train(model,train_loader, optimizer, criterion)
    if epoch % 10 == 0:
        print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')




#%%







for epoch in range(200):
    iterator = tqdm(loader)
    for data,label in train_loader:
        optim.zero_grad()

        h0 = torch.zeros(1, data.shape[0], 8)
        pred = model(data, h0)
        loss = nn.MSELoss()(pred, label)
        optim.step()
        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")
torch.save(model.state_dict(), "./rnn.pth")

# 모델 성능평가
preds = []
total_loss = 0
with torch.no_grad():
    model.load_state_dict(torch.load("rnn.pth"))
    for data, label in loader:
        h0 = torch.zeros(1, data.shape[0], 8)
        pred = model(data, h0)
        preds.append(pred.item())
        loss = nn.MSELoss()(pred,label)
        total_loss += loss / len(loader)










