#%%
import yfinance as yf
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
#%%
from preprocess import *
from model import *
# %%
stock_code = '005930.KS'
start_date = '2020-07-30'
end_date = '2023-07-30'
samsung_data = yf.download(stock_code, start = start_date, end = end_date)
close = samsung_data['Close']
seq_len = 7    
<<<<<<< HEAD
k = 5
=======
k = 30
>>>>>>> 5a14e52beb4a9da9fc5aac1b788ce08a0f757eed

epochs = 200
early_stopping_count = 0
early_stopping = 15
best_valid_loss = float('inf')

hidden_size = 64
filter_num = 128

#%%
train_set, test_set = split_data(data = close, input_window = seq_len, test_len = 60)
train_set, valid_set = split_data(data = train_set, input_window = seq_len, test_len = 60)
train_set = windowDataset(data = train_set, input_window = seq_len, output_window = 1, input_size = 1, stride = 1)
valid_set = windowDataset(data = valid_set, input_window = seq_len, output_window = 1, input_size = 1, stride = 1)
test_set = windowDataset(data = test_set, input_window = seq_len, output_window = 1, input_size = 1, stride = 1)
close_set = windowDataset(data = close, input_window = seq_len, output_window = 1, input_size = 1, stride = 1)

train_num_samples = train_set.x.shape[0]
valid_num_samples = valid_set.x.shape[0]
test_num_samples = test_set.x.shape[0]

train_scale = MinMax(train_set)
valid_scale = MinMax(valid_set)
test_scale = MinMax(test_set)
close_scale = MinMax(close_set)

train_pattern = torch.zeros([train_scale.x.shape[0], train_scale.x.shape[1], k])
valid_pattern = torch.zeros([valid_scale.x.shape[0], valid_scale.x.shape[1], k])
test_pattern = torch.zeros([test_scale.x.shape[0], train_scale.x.shape[1], k])

# %%
# train_pattern
for i in range(train_num_samples):
    train_dist = pd.DataFrame(data = torch.sum(torch.sqrt((train_scale.x[i] - train_scale.x)**2), dim = (1,2)), columns = ['Values'])
    train_dist_sort = train_dist.sort_values(by = 'Values')
    train_ind = train_dist_sort.index[1:k+1].tolist()
    train_pattern[i,:] = torch.cat([train_scale.x[j] for j in train_ind], dim = 1)
# valid_pattern
for i in range(valid_num_samples):
    valid_dist = pd.DataFrame(data = torch.sum(torch.sqrt((valid_scale.x[i] - close_scale.x[:train_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    valid_dist_sort = valid_dist.sort_values(by = 'Values')
    valid_ind = valid_dist_sort.index[0:k].tolist()

    valid_pattern[i,:] = torch.cat([close_scale.x[j] for j in valid_ind], dim = 1)
# test_pattern
for i in range(test_num_samples):
    test_dist = pd.DataFrame(data = torch.sum(torch.sqrt((test_scale.x[i] - close_scale.x[:train_num_samples-seq_len+i])**2), dim = (1,2)), columns = ['Values'])
    
    test_dist_sort = test_dist.sort_values(by = 'Values')
    test_ind = test_dist_sort.index[0:k].tolist()

    test_pattern[i,:] = torch.cat([close_scale.x[j] for j in test_ind], dim = 1)

# %%
train_set.x = torch.cat([train_scale.x, train_pattern], dim = 2)
valid_set.x = torch.cat([valid_scale.x, valid_pattern], dim = 2)
test_set.x = torch.cat([test_scale.x, test_pattern], dim =2)
# %%
train_loader = DataLoader(train_set, batch_size = 64, shuffle = False, drop_last = False)
valid_loader = DataLoader(valid_set, batch_size = 1, shuffle = False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle = False)

# %%
model = TPA_my2(seq_len = seq_len, input_size = train_set.x.shape[2], hidden_size = hidden_size, output_size= 1, filter_num = filter_num)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
epochs = 100
train_losses, valid_losses = [], []


# %%
with tqdm(range(1, epochs+1)) as tr:
    for epoch in tr:
        train_loss = train(model,train_loader, optimizer, criterion)
        valid_loss, valid_predictions, valid_labels = valid(model, valid_loader, criterion)
        valid_acc = calculate_accuracy(valid_predictions, valid_labels)


        # wandb.log({"train_loss":train_loss, 
        #            "valid_loss":valid_loss, 
        #            "valid_acc":valid_acc})
        
        if epoch % 10 == 0:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{valid_loss.item():5f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_TPA.pth')
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= early_stopping:
            print(f'epoch:{epoch}, train_loss:{train_loss.item():5f}')
            print(f'epoch:{epoch}, valid_loss:{train_loss.item():5f}')
            print(f'best valid loss :{best_valid_loss}')
            break
# %%
model = TPA_my2(seq_len = seq_len, input_size = train_set.x.shape[2], hidden_size = hidden_size, output_size = 1, filter_num = filter_num)
model.load_state_dict(torch.load('best_TPA.pth'))
test_predictions, test_labels = eval(model, test_loader)
test_acc = calculate_accuracy(test_predictions, test_labels)
test_predictions_np = [tensor.item() for tensor in test_predictions]
test_labels_np = [tensor.item() for tensor in test_labels]

test_mse = torch.mean((torch.tensor(test_predictions) - torch.tensor(test_labels))**2)

plt.plot(test_predictions_np,color = 'red', label = "pred")
plt.plot(test_labels_np, label = "true")
plt.legend()
print(f'test_acc :{test_acc:5f}, test_mse : {test_mse:5f}')
# %%


plt.plot(train_scale.x[1][:,1])