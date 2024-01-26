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
# Data
stock_code = '005930.KS'
start_date = '2020-07-31'
end_date = '2023-07-31'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
k = 3

epochs = 2000
early_stopping_count = 0
early_stopping = 15
best_valid_loss = float('inf')
#%%
train_set, test_set = split_data(data = close, input_window = 7, test_len = 30)
train_set, valid_set = split_data(data = train_set, input_window = 7, test_len = 30)

train_set = windowDataset(data = train_set, input_window = 7, output_window = 1, input_size = 1, stride = 1)
valid_set = windowDataset(data = valid_set, input_window = 7, output_window = 1, input_size = 1, stride = 1)
test_set = windowDataset(data = test_set, input_window = 7, output_window = 1, input_size = 1, stride = 1)
close_set = windowDataset(data = close, input_window = 7, output_window = 1, input_size = 1, stride = 1)


train_num_samples = train_set.x.shape[0]
valid_num_samples = valid_set.x.shape[0]
test_num_samples = test_set.x.shape[0]

#%%
train_set_org = copy.deepcopy(train_set)
valid_set_org = copy.deepcopy(valid_set)
test_set_org = copy.deepcopy(test_set)
close_set_org = copy.deepcopy(close_set)

train_bound, train_trans = pred_minmax(train_set)   # sequence별 max,min저장
valid_bound, valid_trans = pred_minmax(valid_set)   # sequence별 max,min저장
test_bound, test_trans = pred_minmax(test_set)   # sequence별 max,min저장
close_bound, close_trans = pred_minmax(close_set)

train_pattern = torch.zeros([train_trans.x.shape[0],train_trans.x.shape[1],k])
train_pattern_noscale = torch.zeros([train_set_org.x.shape[0],train_set_org.x.shape[1],k]) 
valid_pattern = torch.zeros([valid_trans.x.shape[0],valid_trans.x.shape[1],k])
valid_pattern_noscale = torch.zeros([valid_set_org.x.shape[0],valid_set_org.x.shape[1],k]) 
test_pattern = torch.zeros([test_trans.x.shape[0],test_trans.x.shape[1],k])
test_pattern_noscale = torch.zeros([test_set_org.x.shape[0],test_set_org.x.shape[1],k]) 

train_bound_list = torch.zeros([train_trans.x.shape[0],k+1,2])
valid_bound_list = torch.zeros([valid_trans.x.shape[0],k+1,2])
test_bound_list = torch.zeros([test_trans.x.shape[0],k+1,2])

#%%
# train_pattern
for i in range(train_num_samples):

    train_dist = pd.DataFrame(data = torch.sum(torch.sqrt((train_trans.x[i] - train_trans.x)**2), 
    dim = (1,2)), columns = ['Values'])

    train_dist_sort = train_dist.sort_values(by = 'Values')
    train_ind = train_dist_sort.index[1:k+1].tolist()

    train_pattern[i,:] = torch.cat([train_trans.x[j] for j in train_ind],dim = 1)   #[num_sample, seq_len, k]
    train_pattern_noscale[i,:] = torch.cat([train_set_org.x[j] for j in train_ind], dim=1)
    train_bound_list[i,:] = torch.cat((torch.FloatTensor(train_bound[i]).unsqueeze(0),
                                 torch.FloatTensor([train_bound[j] for j in train_ind])),
                                 dim=0)    # [num_sample, k+1, 2] 기존,패턴의 min,max 저장
# valid_pattern

for i in range(valid_num_samples):
    valid_dist = pd.DataFrame(data = torch.sum(torch.sqrt((valid_trans.x[i] - close_trans.x[:train_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    valid_dist_sort = valid_dist.sort_values(by = 'Values')
    valid_ind = valid_dist_sort.index[0:k].tolist()

    valid_pattern[i,:] = torch.cat([close_trans.x[j] for j in valid_ind], dim = 1)
    valid_pattern_noscale[i,:] = torch.cat([close_set_org.x[j] for j in valid_ind], dim = 1)
    
    valid_bound_list[i,:] = torch.cat((torch.FloatTensor(valid_bound[i]).unsqueeze(0), 
                                       torch.FloatTensor([close_bound[j] for j in valid_ind])), 
                                       dim = 0)
# test_pattern
for i in range(test_num_samples):
    test_dist = pd.DataFrame(data = torch.sum(torch.sqrt((test_trans.x[i] - close_trans.x[:train_num_samples+valid_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    test_dist_sort = test_dist.sort_values(by = 'Values')
    test_ind = test_dist_sort.index[0:k].tolist()

    test_pattern[i,:] = torch.cat([close_trans.x[j] for j in test_ind], dim = 1)
    test_pattern_noscale[i,:] = torch.cat([close_set_org.x[j] for j in test_ind], dim = 1)
    
    test_bound_list[i,:] = torch.cat((torch.FloatTensor(test_bound[i]).unsqueeze(0), 
                                       torch.FloatTensor([close_bound[j] for j in test_ind])), 
                                       dim = 0)



train_set_org.x = torch.cat([train_set_org.x, train_pattern], dim = 2)
valid_set_org.x = torch.cat([valid_set_org.x, valid_pattern], dim = 2)
test_set_org.x = torch.cat([test_set_org.x, valid_pattern], dim =2)
#train_trans.x = torch.cat([train_trans.x, x_pattern],dim = 2)

#%%
#train_set_org.x = torch.cat([train_set_org.x,train_pattern_noscale],dim = 2)    # minmax 안된 x
# scaling을 거치고 찾은 패턴을 찾아 변환되기 전의 값으로 합침
# train_set_org scaling

#train_trans.x = torch.cat([train_trans.x, train_pattern], dim = 2)





#%%
# 데이터 로더
train_loader = DataLoader(train_set_org, batch_size = 64, shuffle = False, drop_last = True)
valid_loader = DataLoader(valid_set_org, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set_org, batch_size = 1, shuffle = False)
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
    return sum(total_loss)/len(total_loss)

def valid(model, data_loader, criterion):
    
    model.eval()
    h0 = torch.zeros(1,1,8)
    total_loss = []
    
    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label

            pred = model(input, h0)
            loss = criterion(pred, label)
            total_loss.append(loss)
        return sum(total_loss) / len(total_loss)

def eval(model, data_loader):
    
    model.eval()
    h0 = torch.zeros(1,1,8)
    predictions = []
    total_loss = []

    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label

            pred = model(input, h0)
            predictions.append(pred)
        return predictions
#%%
# 모델 학습
model = RNN(input_size = 4, hidden_size = 8, num_layers = 1)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.MSELoss()
with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:
        train_loss= train(model,train_loader, optimizer, criterion)
        valid_loss = valid(model, valid_loader, criterion)

        if epoch % 10 == 0:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_lstm.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= early_stopping:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{train_loss.item():5f}')
            print(f'best valid loss :{best_valid_loss}')
            break
#%%
model = RNN(input_size = 4, hidden_size = 8, num_layers = 1)
model.load_state_dict(torch.load('best_lstm.pth'))

predictions = eval(model, test_loader)




#%%







# for epoch in range(200):
#     iterator = tqdm(loader)
#     for data,label in train_loader:
#         optim.zero_grad()

#         h0 = torch.zeros(1, data.shape[0], 8)
#         pred = model(data, h0)
#         loss = nn.MSELoss()(pred, label)
#         optim.step()
#         iterator.set_description(f"epoch{epoch} loss:{loss.item()}")
# torch.save(model.state_dict(), "./rnn.pth")

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










