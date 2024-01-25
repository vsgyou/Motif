#%%
import torch
import torch.nn as nn


#%%
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(input_size = 4, hidden_size = 8, num_layers = 1, batch_first = True)
        self.fc1 = nn.Linear(in_features = 56, out_features = 32)
        self.fc2 = nn.Linear(in_features = 32, out_features = 1)
        self.relu = nn.ReLU()
    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        x = torch.reshape(x,(x.shape[0],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.flatten(x)
        return x
# %%
