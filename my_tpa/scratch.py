#%%
import yfinance as yf
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
# %%
stock_code = '005930.KS'
start_date = '2020-07-30'
end_date = '2023-07-30'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
k = 10
#%%
def split_data(data, input_window, test_len):
    train_set, test_set = train_test_split(data, test_size = test_len, shuffle = False)
    test_set = pd.concat([train_set[-input_window:], test_set])
    return train_set, test_set

class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride = 1):
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data).view(-1,1)

        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples, output_window)

        for i in range(num_samples):
            X[i,:] = data_tensor[i*stride : i*stride + input_window]
            y[i,:] = data_tensor[i*stride + input_window : i *stride+self.seq_len]
        self.x = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len

def MinMax(data):
    for i in range(data.x.shape[0]):
        max = data.x[i].max()
        min = data.x[i].min()
        data.x[i] = (data.x[i] - min) / (max - min)
        data.y[i] = (data.y[i] - min) / (max - min)
    return data

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
train_scale = MinMax(train_set)
valid_scale = MinMax(valid_set)
test_scale = MinMax(test_set)
close_scale = MinMax(close_set)

train_pattern = torch.zeros([train_scale.x.shape[0], train_scale.x.shape[1], k])
valid_pattern = torch.zeros([valid_scale.x.shape[0], valid_scale.x.shape[1], k])
test_pattern = torch.zeros([test_scale.x.shape[0], train_scale.x.shape[1], k])

# %%
# train_pattern
for i in range(train_num_samples):
    train_dist = pd.DataFrame(data = torch.sum(torch.sqrt((train_scale.x[i] - train_scale.x)**2), dim = (1,2)), columns = ['Values'])
    train_dist_sort = train_dist.sort_values(by = 'Values')
    train_ind = train_dist_sort.index[1:k+1].tolist()
    train_pattern[i,:] = torch.cat([train_scale.x[j] for j in train_ind], dim = 1)
# valid_pattern
for i in range(valid_num_samples):
    valid_dist = pd.DataFrame(data = torch.sum(torch.sqrt((valid_scale.x[i] - close_scale.x[:train_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    valid_dist_sort = valid_dist.sort_values(by = 'Values')
    valid_ind = valid_dist_sort.index[0:k].tolist()

    valid_pattern[i,:] = torch.cat([close_scale.x[j] for j in valid_ind], dim = 1)
# test_pattern
for i in range(test_num_samples):
    test_dist = pd.DataFrame(data = torch.sum(torch.sqrt((test_scale.x[i] - close_scale.x[:train_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    test_dist_sort = test_dist.sort_values(by = 'Values')
    test_ind = test_dist_sort.index[0:k].tolist()

    test_pattern[i,:] = torch.cat([close_scale.x[j] for j in test_ind], dim = 1)

# %%
train_set.x = torch.cat([train_scale.x, train_pattern], dim = 2)
valid_set.x = torch.cat([valid_scale.x, valid_pattern], dim = 2)
test_set.x = torch.cat([test_scale.x, test_pattern], dim =2)
# %%
train_loader = DataLoader(train_set, batch_size = 64, shuffle = False, drop_last = True)
valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

#%%

for inputs, labels in train_loader:
    inputs = inputs
    labels = labels
inputs.shape
labels.shape
# %%
lstm = nn.LSTM(input_size = 11, hidden_size = 16, num_layers = 1, batch_first = True)
outputs, (hn,cn) = lstm(inputs)
h_before = outputs[:,:-1,:] # 64, 6, 16
h_now = hn.reshape(-1,1,hn.shape[2]) # 64,1,16
h_before = h_before.unsqueeze(1) # 64, 1, 6, 16
con = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (6,1))
H = con(h_before).squeeze(dim=2) # [64,32,1,16] -> [64,32,16]
H = H.permute(0,2,1)    # [64,16,32]
linear1 = nn.Linear(16,32)
linear2 = nn.Linear(16+32, 16)
fc_layer = nn.Linear(16,1)
w_h = linear1(h_now)    # 64,1,32
score = torch.sum(torch.mul(H,w_h), dim = 2)    # 64,16
sigmoid = nn.Sigmoid()
a = sigmoid(score)
v = torch.sum(torch.mul(a.unsqueeze(2), H),dim=1) # 64,32
h_v = torch.cat((h_now.squeeze(dim = 1),v),dim=1)
h_prime = linear2(h_v)

outputs = fc_layer(h_prime)
#%%

lstm = nn.LSTM(input_size = 11, hidden_size = 16, num_layers = 1, batch_first = True)
outputs, (hn, cn) = lstm(inputs)
h_before = outputs[:,:-1,:]
h_now = hn.reshape(-1,1,hn.shape[2])
h_before = h_before.unsqueeze(1)
con = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (16,1))
H = con(h_before.permute(0,1,3,2)).squeeze(dim = 2) # 64,32,6
H = H.permute(0,2,1) # 64, 6, 32
linear1 = nn.Linear(16,32)
linear2 = nn.Linear(16+32, 16)
fc_layer = nn.Linear(16,1)
w_h = linear1(h_now).shape # 64,1,32
score=torch.sum(torch.mul(H,w_h),dim = 2) # 64,6
sigmoid = nn.Sigmoid()
a = sigmoid(score)
v = torch.sum(torch.mul(a.unsqueeze(2),H),dim=1)# 64,32
h_v = torch.cat((h_now.squeeze(dim=1),v),dim=1)
h_prime = linear2(h_v)
outputs = fc_layer(h_prime)