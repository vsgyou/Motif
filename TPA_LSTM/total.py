#%%
import yfinance as yf
import torch.nn as nn
import torch
import pandas as pd
import tensorflow as tf
from tensorflow.layers import dense
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



input = input.reshape(64,7,6)


# %%
attn_size = 8
attn_length = 7
batch_size = 64

query = hn
query = query.squeeze(0)
query.shape
atten_states = output
filter_num = 32
filter_size = 1
linear = nn.Linear(query.size(1), filter_num, bias = False)
w = linear(query)
w = w.view(-1,1,filter_num)
atten_states.shape
reshape_attn_vecs  = atten_states.reshape([batch_size,1,attn_length,attn_size])
conv = nn.Conv2d(in_channels = 1,
                 out_channels = 32,
                 kernel_size = [7,1],
                 stride = 1)
conv_vecs = conv(reshape_attn_vecs)
conv_vecs = conv_vecs.squeeze()
feature_dim = attn_size - filter_size + 1
conv_vecs = conv_vecs.reshape([batch_size, feature_dim, filter_num])
conv_vecs.shape

s = torch.sum(torch.mul(conv_vecs,w),dim = 2)
s.shape
sigmoid = nn.Sigmoid()
a = sigmoid(s)
a.shape

d = torch.sum(torch.mul(a.reshape([-1,feature_dim,1]),conv_vecs),[1])
d.shape
new_conv_vec = d
con = torch.concat([query,new_conv_vec],axis = 1)
dense_layer =nn.Linear(con.shape[1],attn_size)
new_attns = dense_layer(con)
start_indices = [0,1,0]
sizes = [-1,-1,-1]
new_attn_states = atten_states.narrow(1, start_indices[1], sizes[1]).narrow(2, start_indices[2], sizes[2])
#%%
class TemporalPatternAttentionMechanism():
    def __call__(self, hidden, input, batch_size, seq_length, input_size):
    
#hidden : output of LSTM (hn) [batch, hidden]
#input : output of LSTM output [batch, seq_len, hidden]
        filter_num = 32
        filter_size = 1
        Linear1 = nn.Linear(hidden.shape[1], filter_num, bias = False)
        w = Linear1(hidden).view(-1,1,filter_num)
        reshape_input = input.reshape([batch_size, 1, seq_length, input_size])
        conv = nn.Conv2d(in_channels = 1, output_channels = 32, kernel_size = [seq_length, 1], stride = 1)
        H_c = conv(reshape_input).squeeze()
        
        feature_dim = input_size + filter_size - 1
        H_c = H_c.reshape([batch_size, feature_dim, filter_num])

        s = torch.sum(torch.mul(H_c, w), dim = 2)
        sigmoid = nn.Sigmoid()
        a = sigmoid(s)

        v = torch.sum(torch.mul(a.reshape([-1, feature_dim, 1]), H_c), dim = 1)
        new_h = torch.concat([hidden, v], axis = 1)
        Linear2 = nn.Linear(new_h.shape[1], input_size)
        h_prime = Linear2(new_h)
        
                






