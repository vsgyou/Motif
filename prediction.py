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
import wandb
#%%
from preprocess import *
from model import *
#%%

wandb.init(
    # set the wandb project where this run will be logged
    project="3years data (input = 6, hidden = 8) all scaling",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.001,
    "architecture": "LSTM",
    "dataset": "Samsung",
    "epochs": 200,
    }
)
#%%
# Data
stock_code = '005930.KS'
start_date = '2020-07-30'
end_date = '2023-07-30'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
k = 5

epochs = 200
early_stopping_count = 0
early_stopping = 15
best_valid_loss = float('inf')
#%%
train_set, test_set = split_data(data = close, input_window = 7, test_len = 60)
train_set, valid_set = split_data(data = train_set, input_window = 7, test_len = 60)

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

#%%
# 원래 sequence와 label은 기존 값, 패턴만 스케일링
train_set.x = torch.cat([train_set.x, train_pattern], dim = 2)
valid_set.x = torch.cat([valid_set.x, valid_pattern], dim = 2)
test_set.x = torch.cat([test_set.x, test_pattern], dim =2)
# #%%
# # 원래 sequence와 패턴 모두 스케일링 x
# train_set_org.x = torch.cat([train_set_org.x, train_pattern_noscale],dim = 2)
# train_set_org.x = train_set_org.x / 5000
# train_set_org.y = train_set_org.y / 5000
# valid_set_org.x = torch.cat([valid_set_org.x, valid_pattern_noscale], dim = 2)
# valid_set_org.x = valid_set_org.x / 5000
# valid_set_org.y = valid_set_org.y / 5000
# test_set_org.x = torch.cat([test_set_org.x, test_pattern_noscale], dim =2)
# test_set_org.x = test_set_org.x / 5000
# test_set_org.y = test_set_org.y / 5000

#%%
#train_set_org.x = torch.cat([train_set_org.x,train_pattern_noscale],dim = 2)    # minmax 안된 x
# scaling을 거치고 찾은 패턴을 찾아 변환되기 전의 값으로 합침
# train_set_org scaling

#train_trans.x = torch.cat([train_trans.x, train_pattern], dim = 2)

#%%
# 데이터 로더
train_loader = DataLoader(train_set, batch_size = 64, shuffle = False, drop_last = True)
valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

# test_set.x = test_pattern
# train_set.x = train_pattern
# valid_set.x = valid_pattern

# train_loader = DataLoader(train_set_org, batch_size = 64, shuffle = False, drop_last = True)
# valid_loader = DataLoader(valid_set_org, batch_size = 1, shuffle = False)
# test_loader = DataLoader(test_set_org, batch_size = 1, shuffle = False)

#%%

# 모델 학습
#model = RNN(input_size = 4, hidden_size = 8, num_layers = 1)
model = LSTM(input_size = 6, hidden_size = 32, output_size = 1, num_layers = 3)
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
model = LSTM(input_size = 6, hidden_size = 32, output_size = 1, num_layers = 3)
model.load_state_dict(torch.load('best_lstm.pth'))

predictions = eval(model, test_loader)

value = [tensor.item() for tensor in predictions]
value
plt.plot(value,color = 'red')
plt.title('Prediction')
plt.plot(test_set.y)
plt.title('true')
plt.legend()
#%%
# 정확도 계산
pred_labels = [1 if (predictions[i+1] - predictions[i]).item() > 0 else 0 for i in range(len(predictions)-1)]
true_labels = [1 if (test_set.y[i+1] - test_set.y[i]).item() > 0 else 0 for i in range(len(test_set.y)-1)]


def calculate_accuracy(pred_labels, true_labels):
    # 예측과 실제 레이블의 길이가 같은지 확인
    if len(pred_labels) != len(true_labels):
        raise ValueError("두 리스트의 길이가 일치하지 않습니다.")

    # 맞춘 예측의 개수 계산
    correct_predictions = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)

    # 전체 예측 개수 계산
    total_predictions = len(pred_labels)

    # 정확도 계산
    accuracy = correct_predictions / total_predictions

    return accuracy

# 예시로 정확도 측정
accuracy = calculate_accuracy(pred_labels, true_labels)
print(f"정확도: {accuracy * 100:.2f}%")




# %%
