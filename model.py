#%%
import torch
import torch.nn as nn


#%%
class RNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size = input_size,
                           hidden_size = hidden_size, num_layers = num_layers, batch_first = True)
        self.fc1 = nn.Linear(in_features = 8,
                              out_features = 64)
        self.fc2 = nn.Linear(in_features = 64,
                              out_features = 1)
        self.relu = nn.ReLU()
    def forward(self, x, h0):
        x, hn = self.rnn(x, h0)
        x = x[:,-1,:]
        x = torch.reshape(x,(x.shape[0],-1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = torch.flatten(x)
        return x
# %%
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, (hn,cn) = self.lstm(input)
        result = output[:,-1,:]
        result = self.fc(result)

        return result
# %%
#%%
def train(model, data_loader, optimizer, criterion):
    
    model.train()
    total_loss = []

    for input,label in data_loader:
        
        input = input
        label = label

        optimizer.zero_grad()

        pred = model(input)
        loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    return sum(total_loss)/len(total_loss)

def valid(model, data_loader, criterion):
    
    model.eval()
    total_loss = []
    
    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label

            pred = model(input)
            loss = criterion(pred, label)
            total_loss.append(loss)
        return sum(total_loss) / len(total_loss)

def eval(model, data_loader):
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label

            pred = model(input)
            predictions.append(pred)
        return predictions
# %%
