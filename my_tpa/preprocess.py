#%%
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

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

# %%
