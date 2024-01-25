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
from preprocess import *
#%%
# Data
stock_code = '005930.KS'
start_date = '2021-07-20'
end_date = '2023-07-20'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
close_MA = MA(close,10)
seq_len = 7
X = window(data = close_MA, seq_len = seq_len)
X = minmax(X)
# X = minmax(X)
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
            if abs(Dist_min_sort.index[j] - Dist_min_sort.index[j+offset]) > int(seq_len/2):
                lower_bound = abs(Dist_min_sort.iloc[j] - Dist_min_sort.iloc[j+offset])
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
                    print(j,j+offset,L1,L2)
Dist_min_sort.index[14]
print(ref,L1,L2)

#%%
plt.plot(X[L1])
plt.plot(X[L2])
# %%

plt.plot(close_MA[0:500])
plt.plot(close_MA[L1:L1+seq_len],color = 'ed')
plt.plot(close_MA[L2:L2+seq_len],color = 'red')
#%%
plt.plot(X[L1])
plt.plot(X[L2])


plt.plot(close[0:500])
plt.plot(close[L1:L1+seq_len],color = 'red')
plt.plot(close[L2:L2+seq_len],color = 'red')

# %%