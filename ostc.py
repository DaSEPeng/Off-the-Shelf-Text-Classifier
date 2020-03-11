"""

主模块

"""

## 加载模块
# 系统模块
from importlib import import_module
import argparse
import time
import copy
import torch
# 自定义模块
from config import Config             # 只是参数配置，不进行处理或者加载
from utils import set_seed,epoch_time
from data import build_dataset,build_iterator,build_vocab,construct_embedding
from model import load_model,init_network,train,evaluate,count_parameters


## 下面这些是自定义的参数，一些预定义的参数在config.py文件夹里
parser = argparse.ArgumentParser(description='OFF-THE-SHELF Text Classification')  # OSTC
# 先假设已经分割成了 训练集、测试集、验证集，后面再处理
# 后面可以由用户自定义测试不测试
parser.add_argument('--datasets',type=str,required=True,help='The root dir which includes the train/dev/test file')
parser.add_argument('--language',type=str,required=True,help='choose a language: chinese or english')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--use_word', default=True, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':

    # 设置随机种子
    set_seed(233)

    # 解析传入的参数
    datasets_path = args.datasets
    language = args.language
    model_name = args.model
    embed = args.embedding
    using_word = args.use_word

    # 整合参数
    config = Config(datasets_path,language,model_name,embed,using_word)

    # 构建数据集
    start_time = time.time()
    print("Loading data...")
    # 加载数据集
    train_data, dev_data, test_data = build_dataset(config)                                             
    print (len(train_data[1]))
    # 构建词表
    vocab_class = build_vocab(config,train_data)                                                        
    config.class_num = vocab_class.label_num                         
    # 构建词向量
    config.embedding_pretrained = construct_embedding(config,vocab_class)                               
    # 构建迭代器
    train_iter, dev_iter, test_iter = build_iterator(config,vocab_class,train_data,dev_data,test_data)  
    end_time = time.time()
    print("Time usage:", end_time - start_time)

    # 加载模型
    model = load_model(config)      
    init_network(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')                                                                    
    # 训练模型
    best_valid_loss = float('inf')
    for epoch in range(config.epoch_num):
        start_time = time.time() 

        train_loss,train_acc = train(config,model,copy.deepcopy(train_iter))                                         
        valid_loss,valid_acc = evaluate(config,model,copy.deepcopy(dev_iter))
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'test_model.pt')
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    test_loss, test_acc = evaluate(config,model,copy.deepcopy(test_iter))
    print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
