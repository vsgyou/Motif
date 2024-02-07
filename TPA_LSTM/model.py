import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = 1, batch_first = True)

    def forward(self, x):
        output, (hn,cn) = self.lstm(x)
        hn = hn.squeeze()
        return output, hn

# %%
class TemporalPatternAttentionMechanism(nn.Module):
    def __init__(self, seq_length, input_size, filter_num):
        super(TemporalPatternAttentionMechanism, self).__init__()
        self.filter_num = filter_num
        self.filter_size = 1
        self.input_size = input_size
        self.seq_length = seq_length
        self.linear1 = nn.Linear(input_size, filter_num, bias = False)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (self.seq_length, self.filter_size))
        self.linear2 = nn.Linear(input_size + filter_num, input_size)
        self.fc_layer = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden, input):
        w = self.linear1(hidden).unsqueeze(1) # [batch, 1, filter_num]

        conv_input = input.unsqueeze(1) # [batch, 1, seq_length, input_size]
        H_c = self.conv(conv_input).squeeze() # [batch, filter_num, input_size]
        H_c = H_c.reshape([-1, self.input_size, self.filter_num]) # [batch, input_size, filter_num]
        

        # Attention scoring
        s = torch.sum(torch.mul(H_c, w), dim = 2) # [batch, input_size]
        a = self.sigmoid(s)

        # Context vector
        v = torch.sum(torch.mul(a.unsqueeze(2), H_c), dim = 1) # [batch, input_size]

        # Concatenate context vector with hidden state
        concat_input = torch.cat((hidden.squeeze(), v), dim = 1) # [batch, input_size, filter_num]

        h_prime = self.linear2(concat_input)
        output = self.fc_layer(h_prime)
        return output
    
    
class TPA_my(nn.Module):
    def __init__(self, seq_len, input_size, hidden_size, filter_num):
        super(TPA_my, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = self.hidden_size, num_layers = 1, batch_first = True)
        self.conv = nn.Conv2d(in_channels = 1, out_channels = filter_num, kernel_size = (seq_len-1,1))
        self.linear1 = nn.Linear(hidden_size, filter_num)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(filter_num + hidden_size, hidden_size)
        self.fc_layer = nn.Linear(hidden_size, 1)
    
    def forward(self, input):
        output, (hn,cn) = self.lstm(input)
        h_before = output[:,:-1,:]
        h_now = hn.reshape(-1,1,hn.shape[2])
        h_before = h_before.unsqueeze(1) # [batch, 1, seq_len-1, hidden_size]

        H = self.conv(h_before).squeeze() # [batch, out_channels, hidden_size]
        H = H.permute(0,2,1) # [batch, hidden_size, out_channels]

        w_h = self.linear1(h_now)
        score = torch.sum(torch.mul(H, w_h), dim = 2)
        
        a = self.sigmoid(score)
        v = torch.sum(torch.mul(a.unsqueeze(2), H), dim=1) # [batch, filter_num]
        
        h_v = torch.cat((h_now.squeeze(), v), dim=1) # [batch, filter_num+hidden_size]
        h_prime = self.linear2(h_v) # [batch, hidden_size]
        
        output = self.fc_layer(h_prime) # [batch, 1]
        return output