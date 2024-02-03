#%%
import yfinance as yf
import torch.nn as nn
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
#%%
code = 'AAPL'
start_num = '2020-07-30'
end_num = '2023-07-30'
data = yf.download(code, start_num, end_num)
# %%
def split_data(data, input_window, test_len):
    train_set, test_set = train_test_split(data, test_size = test_len, shuffle = False)
    test_set = pd.concat([train_set[-input_window:],test_set])
    return train_set, test_set

class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride = 1):
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = (L - self.seq_len) // stride + 1
        
        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples,output_window)
        data_tensor = torch.tensor(data.values)
        for i in range(num_samples):
            X[i,:] = data_tensor[i*stride : i*stride + input_window]
            y[i,:] = data_tensor[i*stride + input_window : i*stride + self.seq_len,3]
            self.x = X
            self.y = y
            self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.x[idx]
    def __len__(self):
        return self.len

# %%
train_set, test_set = split_data(data = data, 
                                 input_window = 7, 
                                 test_len = 60)
train_set = windowDataset(train_set, input_window= 7, output_window= 1, input_size = train_set.shape[1])
test_set = windowDataset(test_set, input_window = 7, output_window= 1, input_size = test_set.shape[1])
train_loader = DataLoader(train_set, batch_size = 64, shuffle = False, drop_last = True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)
#%%
for input, label in train_loader:
    input = input
    label = label
#%%
lstm = nn.LSTM(input_size = data.shape[1], hidden_size = 8, num_layers = 1,batch_first = True)
output,(hn,cn) = lstm(input)
output.shape
hn.shape
cn.shape
hn[:,:,0]

conv = nn.Conv2d(in_channels = 7, out_channels = 32, kernel_size = [6,1], stride = 1)

input = input.reshape(64,7,6,1)
conv(input).shape
input.shape
# %%
