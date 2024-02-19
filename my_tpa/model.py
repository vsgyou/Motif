#%%
import torch.nn as nn
import torch

class TPA_my(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size, filter_num):
        super(TPA_my, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers = 1, batch_first = True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (seq_len,1))
        self.linear1 = nn.Linear(hidden_size, filter_num)
        self.sigmoid = nn.Softmax()
        self.linear2 = nn.Linear(filter_num + hidden_size, hidden_size)
        self.fc_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        output, (hn,cn) = self.lstm(input)
        #h_before = output[:,:-1,:]
        h_before = output
        h_now = hn.reshape(-1,1,hn.shape[2])
        h_before = h_before.unsqueeze(1) # [batch, 1, seq_len-1, hidden_size]

        H = self.conv(h_before).squeeze(dim = 2) # [batch, out_channels, hidden_size]
        H = H.permute(0,2,1) # [batch, hidden_size, out_channels]

        w_h = self.linear1(h_now)
        score = torch.sum(torch.mul(H, w_h), dim = 2)
        
        a = self.sigmoid(score)
        v = torch.sum(torch.mul(a.unsqueeze(2), H), dim=1) # [batch, filter_num]
        
        h_v = torch.cat((h_now.squeeze(dim = 1), v), dim=1) # [batch, filter_num+hidden_size]
        h_prime = self.linear2(h_v) # [batch, hidden_size]
        
        output = self.fc_layer(h_prime) # [batch, output_size]
        return output
    
class TPA_my2(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, output_size, filter_num):
        super(TPA_my2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers = 1, batch_first = True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (hidden_size,1))
        self.linear1 = nn.Linear(hidden_size, filter_num)
        self.sigmoid = nn.Softmax()
        self.linear2 = nn.Linear(filter_num + hidden_size, hidden_size)
        self.fc_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        output, (hn,cn) = self.lstm(input)
        #h_before = output[:,:-1,:]
        h_before = output
        h_now = hn.reshape(-1,1,hn.shape[2])
        h_before = h_before.unsqueeze(1) # [batch, 1, seq_len-1, hidden_size]

        H = self.conv(h_before.permute(0,1,3,2)).squeeze(dim = 2) # [batch, out_channels, hidden_size]
        H = H.permute(0,2,1) # [batch, hidden_size, out_channels]

        w_h = self.linear1(h_now)
        score = torch.sum(torch.mul(H, w_h), dim = 2)
        
        a = self.sigmoid(score)
        v = torch.sum(torch.mul(a.unsqueeze(2), H), dim=1) # [batch, filter_num]
        
        h_v = torch.cat((h_now.squeeze(dim = 1), v), dim=1) # [batch, filter_num+hidden_size]
        h_prime = self.linear2(h_v) # [batch, hidden_size]
        
        output = self.fc_layer(h_prime) # [batch, output_size]
        return output


def train(model, data_loader, optimizer, criterion):
    
    model.train()
    total_loss = []

    for input,label in data_loader:
        
        input = input
        label = label
        optimizer.zero_grad()

        pred = model(input)
        loss = criterion(pred, label.squeeze(2))

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
            loss = criterion(pred, label.squeeze(2))
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


# %%
