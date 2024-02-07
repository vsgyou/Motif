#%%
import yfinance as yf
import torch.nn as nn
import torch
import pandas as pd
import tensorflow as tf
import torch.optim as optim
import matplotlib.pyplot as plt
import tqdm
from model import *
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
#%%
code = 'AAPL'
start_num = '2020-07-30'
end_num = '2023-07-30'
data = yf.download(code, start_num, end_num)
data = data.iloc[:,:4]
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
        y = torch.zeros(num_samples, output_window)
        data_tensor = torch.tensor(data.values)
        for i in range(num_samples):
            X[i,:] = data_tensor[i*stride : i*stride + input_window]
            y[i,:] = data_tensor[i*stride + input_window : i*stride + self.seq_len,3]
            self.x = X
            self.y = y
            self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
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
input_size = 4
hidden_size = 8
seq_length = 7
filter_num = 32
lstm_model = LSTM(input_size = input_size, hidden_size = hidden_size)
tpa_model = TemporalPatternAttentionMechanism(seq_length = seq_length, input_size = hidden_size, filter_num = filter_num)

criterion = nn.MSELoss()
optimizer = optim.Adam(list(lstm_model.parameters())+ list(tpa_model.parameters()), lr = 0.001) 
# %%
# 학습
num_epochs = 200
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    lstm_model.train()
    tpa_model.train()
    train_loss = 0.0
    for input_data, label in train_loader:
        optimizer.zero_grad()

        lstm_output, hn = lstm_model(input_data)
        output = tpa_model(hn, lstm_output)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(train_loss)
    train_losses.append(train_loss)

# %%
class TPA(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, filter_num):
        super(TPA, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers = 1, batch_first = True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (seq_len-1,1))
        self.linear1 = nn.Linear(hidden_size, filter_num)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(filter_num + hidden_size, hidden_size)
        self.fc_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, input):
        output, (hn,cn) = self.lstm(input)
        h_before = output[:,:-1,:]
        h_now = hn.reshape(-1,1,hn.shape[2])
        h_before = h_before.unsqueeze(1) # [batch, 1, seq_len-1, hidden_size]

        H = self.conv(h_before).squeeze() # [batch, out_channels, hidden_size]
        H = H.permute(0,2,1) # [batch, hidden_size, out_channels]

        w_h = self.linear1(h_now)
        score = torch.sum(torch.mul(H, w_h), dim = 2)
        
        a = self.sigmoid(score)
        v = torch.sum(torch.mul(a.unsqueeze(2), H), dim=1) # [batch, filter_num]
        
        h_v = torch.cat((h_now.squeeze(), v), dim=1) # [batch, filter_num+hidden_size]
        h_prime = self.linear2(h_v) # [batch, hidden_size]
        
        output = self.fc_layer(h_prime) # [batch, 1]
        return output

# %%
model = TPA(seq_len = 7, input_size = 4, hidden_size = 8, filter_num = 32)
output = model(input)
output
# %%

