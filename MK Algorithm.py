#%%
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
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

#%%
# Data
stock_code = '005930.KS'
start_date = '2023-01-01'
end_date = '2023-07-31'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7
X = window(data = close, seq_len = seq_len)

# %%
best_so_far = np.inf
R = 5
ref = random.sample(range(0,X.shape[0]),R)
Dist = pd.DataFrame(index = ref, columns = range(0,X.shape[0]))
S = []
for i in range(0,R):
    for j in range(0,X.shape[0]):
        if abs(ref[i] -j) <= seq_len/2:
            Dist.iloc[i,j] = np.nan
        else:
            Dist.iloc[i,j] = euclidean_distance(X[ref[i]],X[j]).numpy()

        if Dist.iloc[i,j] < best_so_far:
            best_so_far = Dist.iloc[i,j]
            L1 = ref[i]
            L2 = j              
    S.append(list([ref[i],Dist.iloc[i,:].std()]))
S_sorted = sorted(S, key = lambda x:x[1], reverse = True)
S_sorted_list = [item[0] for item in S_sorted]

offset = 0
abandon = False
while abandon == False:
    offset = offset+1
    abandon = True
    for j in tqdm(range(len(X)-offset)):
        for i in (range(len(ref))):
            Dist_min = Dist.loc[S_sorted_list[i]]
            Dist_min_sort = Dist_min.sort_values()
            lower_bound = abs(Dist_min_sort[j] - Dist_min_sort[j+offset])
            if lower_bound > best_so_far:
                break
            elif i == 0:
                 abandon = False
            d = euclidean_distance(X[Dist_min_sort.index[j]],X[Dist_min_sort.index[j+offset]])
            d = d.numpy()
            if d < best_so_far:
                best_so_far = d
                L1 = Dist_min_sort.index[j]
                L2 = Dist_min_sort.index[j+offset]

print(ref,L1,L2)
#%%
plt.plot(X[L1])
plt.plot(X[L2])
# %%

plt.plot(close[50:100])
plt.plot(close[L1:L1+seq_len],color = 'red')
plt.plot(close[L2:L2+seq_len],color = 'red')


# %%