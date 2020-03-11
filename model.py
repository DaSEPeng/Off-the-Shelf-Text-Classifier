"""

与模型相关的一些模块，加载/训练/测试

"""

from importlib import import_module
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def load_model(config):
    print ("loading model ...")
    model_name = config.model_name
    model = import_module('models.' + model_name).Model(config)
    print ("bingo! ",model_name + "loaded!")
    return model

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True)    # [n*1]
    correct = max_preds.squeeze(1).eq(y)                 # [1*n]
    return correct.sum() / torch.FloatTensor([y.shape[0]])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config,model,train_it):
    """
    train the model
    """
    start_time = time.time()    

    epoch_loss = 0
    epoch_acc = 0
    batch_num = 0

    model.train()                  # 将模型置到train模式
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    for batch in train_it:
        train_list = batch[0]
        label_list = batch[1]
        tmp_it = torch.tensor(train_list,dtype = torch.long)   # 默认是float     
        labels = torch.tensor(label_list,dtype = torch.long)   # [batch size]

        preds = model(tmp_it)                  # [batch size,class size]
             
        loss = F.cross_entropy(preds,labels)
        acc = categorical_accuracy(preds,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        batch_num += 1
    end_time = time.time()
    return epoch_loss / batch_num, epoch_acc / batch_num

def evaluate(config,model,eval_it):
    """
    evaluate the model
    """
    epoch_loss = 0
    epoch_acc = 0
    batch_num = 0

    model.eval()
    
    with torch.no_grad():
        for batch in eval_it:
            eval_list = batch[0]
            label_list = batch[1]
            tmp_it = torch.tensor(eval_list,dtype = torch.long)   # 默认是float
            labels = torch.tensor(label_list,dtype = torch.long)   # [batch size]
            
            preds = model(tmp_it)
            
            loss = F.cross_entropy(preds,labels)
            acc = categorical_accuracy(preds,labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            batch_num += 1
    return epoch_loss / batch_num, epoch_acc / batch_num
