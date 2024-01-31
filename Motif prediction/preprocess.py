#%%
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

def window(data, seq_len, stride = 1):
    L = data.shape[0]
    data_tensor = torch.tensor(data)
    data_tensor = data_tensor.view(-1,1)
    num_samples = (L - seq_len) // stride + 1
    X = torch.zeros(num_samples,seq_len,1)

    for i in range(num_samples):
        X[i,:] = data_tensor[i*stride : i*stride+seq_len]
    return X

def split_data(data, input_window, test_len):
    train_set, test_set = train_test_split(data, test_size = test_len, shuffle = False)
    test_set = pd.concat([train_set[-input_window:],test_set])
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

class windowDataset_candle(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride = 1):
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samples = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data.values)

        X = torch.zeros(num_samples, input_window, input_size)
        y = torch.zeros(num_samples, output_window)

        for i in range(num_samples):
            X[i,:] = data_tensor[i*stride : i*stride + input_window,:]
            y[i,:] = data_tensor[i*stride + input_window : i *stride+self.seq_len,-1]
        self.x = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len




def loader(train_set, test_set, batch_size = 64):
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader  = DataLoader(test_set, batch_size = 1, shuffle = False)
    return train_loader, test_loader

def euclidean_distance(a,b):
    diff = a-b
    squared_diff = diff ** 2
    sum_squared_diff = sum(squared_diff)
    distance = torch.sqrt(sum_squared_diff)
    return distance

def normal(data):
    para = []
    for i in range(data.shape[0]):
        para.append([data[i].mean(),data[i].std()])
        data[i] = (data[i].mean()-data[i])/data[i].std()
    return para, data

def minmax(data):
    for i in range(data.shape[0]):
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data

def pred_minmax(data):
    bound = []
    for i in range(data.x.shape[0]):
        max = data.x[i].max()
        min = data.x[i].min()
        bound.append([max,min])
        data.x[i] = (data.x[i] - min) / (max - min)
        data.y[i] = (data.y[i] - min) / (max - min)
    return bound, data

def MA(data,p):
    result = data.copy()
    result = data.rolling(window = p).mean()
    for i in range(p-1):
        result[i] = data[:i+1].mean()
    return result

# %%
