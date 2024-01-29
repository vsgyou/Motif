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
    predictions = []
    labels = []
    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label
            labels.append(label)

            pred = model(input)
            predictions.append(pred)
            loss = criterion(pred, label)
            total_loss.append(loss)
        return sum(total_loss) / len(total_loss), predictions, labels

def eval(model, data_loader):
    
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for input, label in data_loader:

            input = input
            label = label
            labels.append(label)
            pred = model(input)
            predictions.append(pred)
        return predictions, labels
#%%
def calculate_accuracy(pred_labels, true_labels):
    # 예측과 실제 레이블의 길이가 같은지 확인
    if len(pred_labels) != len(true_labels):
        raise ValueError("두 리스트의 길이가 일치하지 않습니다.")

    pred_labels = [1 if (pred_labels[i+1] - pred_labels[i]).item() > 0 else 0 for i in range(len(pred_labels)-1)]
    true_labels = [1 if (true_labels[i+1] - true_labels[i]).item() > 0 else 0 for i in range(len(true_labels)-1)]

    # 맞춘 예측의 개수 계산
    correct_predictions = sum(1 for pred, true in zip(pred_labels, true_labels) if pred == true)

    # 전체 예측 개수 계산
    total_predictions = len(pred_labels)

    # 정확도 계산
    accuracy = correct_predictions / total_predictions

    return accuracy

# 예시로 정확도 측정

# %%
