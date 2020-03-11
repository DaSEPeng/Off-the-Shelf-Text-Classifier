"""

Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification

"""

import torch
import torch.nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained,freeze=False)   # 后面再调整
        self.lstm = nn.LSTM(config.embed_dim,config.hidden_size,config.num_layers,bidirectional=True,dropout=config.dropout)
        


    def forward(self):
        pass
