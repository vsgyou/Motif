#%%
import torch
#%%
def window(data, seq_len, stride = 1):
    L = data.shape[0]
    data_tensor = torch.tensor(data)
    data_tensor = data_tensor.view(-1,1)
    num_samples = (L - seq_len) // stride + 1
    X = torch.zeros(num_samples,seq_len,1)

    for i in range(num_samples):
        X[i,:] = data_tensor[i*stride : i*stride+seq_len]
    return X
#%%
def euclidean_distance(a,b):
    diff = a-b
    squared_diff = diff ** 2
    sum_squared_diff = sum(squared_diff)
    distance = torch.sqrt(sum_squared_diff)
    return distance

# %%
def normal(data):
    for i in range(data.shape[0]):
        data[i] = (data[i].mean()-data[i])/data[i].std()
    return data

def minmax(data):
    for i in range(data.shape[0]):
        data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
    return data
# %%
