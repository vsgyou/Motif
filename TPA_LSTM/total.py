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
from sklearn.preprocessing import MinMaxScaler
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
        data_tensor = torch.tensor(data)
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
scaler = MinMaxScaler()
train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)

train_set = windowDataset(train_set, input_window= 7, output_window= 1, input_size = train_set.shape[1])
test_set = windowDataset(test_set, input_window = 7, output_window= 1, input_size = test_set.shape[1])
train_loader = DataLoader(train_set, batch_size = 64, shuffle = False, drop_last = True)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)


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
input_size = 4
hidden_size = 8
seq_length = 7
filter_num = 32
model = TPA(seq_len = 7, input_size = 4, hidden_size = 8, filter_num = 32)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) 
# %%
epochs = 200
train_losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * input.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1} / {epochs}, Loss : {epoch_loss}")
# %%
test_losses = []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        test_loss = criterion(outputs, labels)
        test_losses.append(test_loss.item())
    avg_test_loss = sum(test_losses)/len(test_losses)
    print(f"Average Test Loss : {avg_test_loss}")


#%%    
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
    
for inputs, labels in test_loader:
    inputs = inputs
    labels = labels

input_size = 4
hidden_size = 8
filter_num = 32
lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first = True)
conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (6,1))
linear1 = nn.Linear(hidden_size, filter_num)
linear2 = nn.Linear(filter_num + hidden_size, hidden_size)
sigmoid = nn.Sigmoid()
fc_layer = nn.Linear(hidden_size, 1)
output, (hn,cn) = lstm(inputs)
h_before = output[:,:-1,:]
h_now = hn.reshape(-1,1,hn.shape[2])
h_before = h_before.unsqueeze(1)
H = conv(h_before).squeeze()