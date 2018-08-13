import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


class EncoderHistory(nn.Module):
    def __init__(self, action_size, embed_size, feature_size, output_size):
        super(EncoderHistory, self).__init__()

        self.hidden_size = output_size
        self.n_layer = 2

        self.embed = nn.Embedding(action_size, embed_size)
        self.lstm = nn.LSTM(embed_size + feature_size, self.hidden_size, self.n_layer, batch_first=True, dropout=0.5)
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
        output = F.dropout(output, 0.5)
        output = F.relu(output)
        output = self.linear(output)

        return output.squeeze(1), h_n, c_n


class A2CAgent(nn.Module):
    def __init__(self, inst_size, context_size, action_size):
        super(A2CAgent, self).__init__()
        self.linear1 = nn.Linear(inst_size + context_size, 512)
        self.linear2 = nn.Linear(512, 256)

        self.actor_linear1 = nn.Linear(256, 128)
        self.actor_linear2 = nn.Linear(128, 64)
        self.actor_linear3 = nn.Linear(64, action_size)

        self.critic_linear1 = nn.Linear(256, 128)
        self.critic_linear2 = nn.Linear(128, 64)
        self.critic_linear3 = nn.Linear(64, 1)


    def forward(self, instr, instr_lengths, context):


        idx = (torch.LongTensor(instr_lengths) - 1).view(-1, 1).expand(len(instr_lengths), instr.size(2))
        idx = idx.unsqueeze(1).cuda()
        last_instr = instr.gather(1, Variable(idx)).squeeze(1)

        context_cue = torch.cat((last_instr, context), 1)

        output = self.linear1(context_cue)
        output = F.dropout(output, 0.5)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.dropout(output, 0.5)
        output = F.relu(output)

        actor_output = self.actor_linear1(output)
        actor_output = F.dropout(actor_output, 0.5)
        actor_output = F.relu(actor_output)
        actor_output = self.actor_linear2(actor_output)
        actor_output = F.dropout(actor_output, 0.5)
        actor_output = F.relu(actor_output)
        actor_output = self.actor_linear3(actor_output)

        critic_output = self.critic_linear1(output)
        critic_output = F.dropout(critic_output, 0.5)
        critic_output = F.relu(critic_output)
        critic_output = self.critic_linear2(critic_output)
        critic_output = F.dropout(critic_output, 0.5)
        critic_output = F.relu(critic_output)
        critic_output = self.critic_linear3(critic_output)

        return actor_output, critic_output
 




