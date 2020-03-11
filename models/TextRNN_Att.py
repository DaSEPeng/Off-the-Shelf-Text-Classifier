"""

Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)   # 后面再调整
        self.lstm = nn.LSTM(config.embed_dim,config.hidden_size,config.num_layers,bidirectional=True,dropout=config.dropout)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden2_size)
        self.fc2 = nn.Linear(config.hidden2_size, config.class_num)


    def forward(self,text):
        text = text.transpose(0,1)  
        embed = self.embedding(text)    # [sent len, batch size, embed dim]
        output,(hidden,cell) = self.lstm(embed)   # output: [sent len, batch size, hid dim * num directions]
        H = output           # [sent len, batch size, hid dim * num directions]
        M = self.tanh(H)     # [sent len, batch size, hid dim * num directions]
        alpha = F.softmax(torch.matmul(M, self.w), dim=0).unsqueeze(-1)  # [sent len, batch size, 1]
        out = H * alpha      # [sent len, batch size, hid dim * num directions]
        out = torch.sum(out,0)  # [batch size, hid dim * num directions]
        out = self.fc1(out)
        out = self.fc2(out)
        return out
        
        
