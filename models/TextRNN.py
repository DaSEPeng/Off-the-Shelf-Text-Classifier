"""

模型

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)   # 后面再调整
        self.lstm = nn.LSTM(config.embed_dim,config.hidden_size,config.num_layers,bidirectional=True,dropout=config.dropout)
        # self.rnn = nn.RNN(config.embed_dim,config.hidden_size)
        self.fc = nn.Linear(config.hidden_size*2,config.class_num)
        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self,text):
        text = torch.transpose(text,0,1)                  
        # [sent len, batch size]
        embed = self.embedding(text)  
        # [sent len, batch size, embed dim]
        output,(hidden,cell) = self.lstm(embed)
        # output,hidden = self.rnn(embed)
        # [layer*direction_num,batch_size,hidden_dim]
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)  # [batch size, hid dim * num directions]
        # hidden = hidden[-1,:,:]
        # droped_hidden = self.dropout(hidden)
        result = self.fc(hidden)
        # print ("final output size: ", result.size())
        return result
