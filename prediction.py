#%%
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.adam import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
#%%
from preprocess import *
from model import *
#%%
class windowDataset(Dataset):
    def __init__(self, data, input_window, output_window, input_size, stride = 1):
        L = data.shape[0]
        self.seq_len = input_window + output_window
        num_samles = (L - self.seq_len) // stride + 1
        data_tensor = torch.tensor(data).view(-1,1)

        X = torch.zeros(num_samles, input_window, input_size)
        y = torch.zeros(num_samles, output_window)

        for i in range(num_samles):
            X[i,:] = data_tensor[i*stride : i*stride + input_window]
            y[i,:] = data_tensor[i*stride + input_window : i *stride+self.seq_len]
        self.x = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __len__(self):
        return self.len

# Data
stock_code = '005930.KS'
start_date = '2020-07-31'
end_date = '2023-07-31'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
#%%
train_set, test_set = split_data(data = close, input_window = 7, test_len = 80)
train_set = windowDataset(data = train_set, input_window = 7, output_window = 1, input_size = 1, stride = 1)
num_samples = train_set.x.shape[0]
# train_x = train_set[:][0]
# train_y = train_set[:][1]

train_set_org = train_set
bound,train_trans = pred_minmax(train_set)   # sequence별 max,min저장
k = 3
x_pattern = torch.zeros([train_trans.x.shape[0],train_trans.x.shape[1],k])
#x_pattern_noscale = torch.zeros([train_set.x.shape[0],train_set.x.shape[1],k]) 
bound_list = torch.zeros([train_trans.x.shape[0],k+1,2])
for i in range(num_samples):
    dist = pd.DataFrame(data = torch.sum(torch.sqrt((train_trans.x[i] - train_trans.x)**2),
                                         dim = (1,2)), columns = ['Values'])
    dist_sort = dist.sort_values(by = 'Values')
    ind = dist_sort.index[1:k+1].tolist()
    x_pattern[i,:] = torch.cat([train_trans.x[j] for j in ind],dim = 1)   #[num_sample, seq_len, k]
    #x_pattern[i,:] = [train_set_org.x[j] for j in ind]
    bound_list[i,:] = torch.cat((torch.FloatTensor(bound[i]).unsqueeze(0),torch.FloatTensor([bound[j] for j in ind])),dim=0)    # [num_sample, k+1, 2] 기존,패턴의 min,max 저장
torch.FloatTensor(bound)
train_trans.x.shape
x_pattern.shape
train_trans.x = torch.cat([train_trans.x, x_pattern], dim = 2)
(train_trans.x.shape, train_trans.y.shape)
#%%
# 데이터 로더
train_loader = DataLoader(train_trans, batch_size = 64, shuffle = False, drop_last = True)
loader = DataLoader(train_trans, batch_size = 1, shuffle = False, drop_last = True)
#%%
# for data,label in train_loader:
#     data = data
#     label = label


#%%
# 모델 학습
model = RNN()
optim = Adam(params = model.parameters(), lr = 0.0001)
for epoch in range(200):
    iterator = tqdm(loader)
    for data,label in train_loader:
        optim.zero_grad()

        h0 = torch.zeros(1, data.shape[0], 8)
        pred = model(data, h0)
        loss = nn.MSELoss()(pred, label)
        optim.step()
        iterator.set_description(f"epoch{epoch} loss:{loss.item()}")
torch.save(model.state_dict(), "./rnn.pth")

# 모델 성능평가
preds = []
total_loss = 0
with torch.no_grad():
    model.load_state_dict(torch.load("rnn.pth"))
    for data, label in loader:
        h0 = torch.zeros(1, data.shape[0], 8)
        pred = model(data, h0)
        preds.append(pred.item())
        loss = nn.MSELoss()(pred,label)
        total_loss += loss / len(loader)





















for batch_idx, samples in enumerate(train_loader):
    x_train, y_train = samples



#%%





#%%
def calculate_weights(k):
    # 0부터 1까지를 k개의 구간으로 나누기
    ranks = np.arange(k, 0, -1) / k

    # 각 구간에 대한 가중치 할당
    weights = ranks / np.sum(ranks)

    return weights.tolist()


k = 10
X = window(data = input, seq_len = seq_len)
X.reshape(136,7,1)
#X = normal(X)
# %%
dist = torch.sum(torch.sqrt((X[-1] - X[:-1])**2), dim = (1,2))
#dist_mat = torch.cat([torch.arange(len(dist)).unsqueeze(1),dist.unsqueeze(1)],dim=1)
np.array(dist)
dist_df = pd.DataFrame(dist.numpy(),columns = ['Value'])
sort_df = dist_df.sort_values(by = 'Value')
ind = sort_df.iloc[:k].index.tolist()

X_ind = X[ind].reshape(seq_len,k)
input_data = torch.cat([X[-1],X_ind],dim = 1)
input_data.reshape(1,7,11)

#%%
lstm = nn.LSTM(input_size = 7, hidden_size = 11, batch_first = True)
out,_= lstm(input_data)
out.shape
fc = nn.Linear(11,1)
fc_out = fc(out)
class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, output_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:,-1,:])
        return out
#%%













weight = calculate_weights(k)
model = SimpleModel(input_size = seq_len, hidden_size = 64, output_size = 1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
input_data = torch.Tensor(X[ind[:]]).view(-1,seq_len,1)
scaler = MinMaxScaler()
scaler.fit(close.values.reshape(-1,1))

target_data  = torch.Tensor(scaler.transform(target.reshape(1,-1)))
#%%
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, target_data)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#%%
model.eval()
with torch.no_grad():
    predicted_value = model(input_data)

predicted_value = scaler.inverse_transform(predicted_value.numpy())


# %%