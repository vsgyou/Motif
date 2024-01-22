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
from model import *
#%%
# Data
stock_code = '005930.KS'
start_date = '2023-01-01'
end_date = '2023-07-31'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7
X = window(data = close, seq_len = seq_len)
X = normal(X)
# %%
target = X[-1]
input = X[:-1]
dist = torch.sqrt(torch.sum((input - target)**2, dim = (1,2)))
#dist_mat = torch.cat([torch.arange(len(dist)).unsqueeze(1),dist.unsqueeze(1)],dim=1)
np.array(dist)
dist_df = pd.DataFrame(dist.numpy(),columns = ['Value'])
# 마지막 sequence와의 거리 행렬 Dist
motiflet = []
d = np.inf
# %%
k = 10
for i in range(X.shape[0]):
    idx = dist_df.index[dist_df['Value']<d]
    if len(idx) >= k:
        candidate =  