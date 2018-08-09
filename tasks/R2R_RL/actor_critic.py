import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class EncoderHistory(nn.Module):
    def __init__(self, action_size, embed_size, feature_size):
        super(EncoderHistory, self).__init__()

        self.hidden_size = 1024
        self.n_layer = 2

        self.embed = nn.Embedding(action_size, embed_size)
        self.lstm = nn.LSTM(embed_size + feature_size, self.hidden_size, self.n_layer, batch_first=True)
        self.linear = nn.Linear(self.hidden_size + feature_size, self.hidden_size)


    def init_hidden(self, batch_size):
        h_n = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        c_n = torch.zeros(self.n_layer, batch_size, self.hidden_size).cuda()
        return h_n, c_n


    def forward(self, action, feature, h_0, c_0):

        embedding = self.embed(action)
        input = torch.cat((embedding, feature), 1)

        output, (h_n, c_n) = self.lstm(input.unsqueeze(1), (h_0, c_0))
        output = torch.cat((output, feature.unsqueeze(1)), 2)
        output = F.relu(output)
        output = self.linear(output)
        output = F.relu(output)


        print(output.size())
        return output.squeeze(1), h_n, c_n

class A2CAgent(nn.Module):
    def __init__(self):
        super(A2CAgent, self).__init()
        self.linear1 = nn.Linear(4, 128)
