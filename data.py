"""

一些处理数据的基本函数

"""

import jieba
import spacy
import numpy as np
import torch    # 只在构建词向量的时候使用

def build_dataset(config):
    """
    构建训练集，分词，列表格式
    [["词1"，"词2"，"词3"，"词4"...],[...]]
    [标签1,标签2,...]
    [长度1,长度2,长度3,...]
    构成元组
    :param config:
    :return:
    """
    train_path = config.train_path
    dev_path = config.dev_path
    test_path = config.test_path

    def build_dataset_base(tmp_path):
        with open(tmp_path,'r',encoding='utf8',errors="ignore") as train_f:
            tmp_sents_num = 0
            tmp_contents = []
            tmp_labels = []
            tmp_line_lens = []              # 用来生成长度相近的batch
            while True:
                line = train_f.readline()
                if line:
                    str_line = line[:-1]   # 去掉换行符
                    text = ' '.join(str_line.split(' ')[:-1])
                    if config.using_word == True:
                        text_cut = list(jieba.cut(text))  # jieba.cut是生成器 
                    else:
                        text_cut = [i for i in text if i != ' ']  # TODO 词表大小还需要修改
                    label = str_line.split(' ')[-1]
                    text_len = len(text_cut)

                    tmp_contents.append(text_cut)
                    tmp_labels.append(label)
                    tmp_line_lens.append(text_len)
                    tmp_sents_num += 1
                else:
                    print ("tmp sentences number: ", tmp_sents_num)
                    break
            return (tmp_contents,tmp_labels,tmp_line_lens)

    train_data = build_dataset_base(train_path)
    dev_data = build_dataset_base(dev_path)
    test_data = build_dataset_base(test_path)
    return train_data,dev_data,test_data



class build_vocab:
    def __init__(self,config,dataset):
        self.max_vocab_size = config.max_vocab_size
        self.unk = config.unk
        self.pad = config.pad
        self.dataset = dataset
        self.tf = self.count_tf(self.dataset)   # 计算词频，返回字典
        self.vocab = self.construct_vocab(self.tf)    # list
        self.vocab_len = len(self.vocab)
        self.label_vocab = self.construct_label_vocab(self.dataset)  # 将标签对应成序号
        self.label_num = len(self.label_vocab)

    def count_tf(self,tmp_dataset):
        """
        计算词频，返回一个排好序的list
        """
        print ("counting tf....")
        tf_dict = {}
        tmp_content_list = tmp_dataset[0]
        for sent in tmp_content_list:
            for word in sent:
                if word in tf_dict:
                    tf_dict[word] += 1
                else:
                    tf_dict[word] = 1
        #  print (tf_dict)
        print ("bingo!")
        return tf_dict

    def construct_vocab(self,tf):
        """
        构建词表
        """
        print ("construct vocab...")
        sorted_tf = sorted(tf.items(),key=lambda x:x[1],reverse=True)   # 按照词频进行排序
        max_size = self.max_vocab_size
        tmp_vocab = []
        tmp_vocab.append(self.unk)
        tmp_vocab.append(self.pad)
        print (sorted_tf)
        print (len(sorted_tf))
        tmp_vocab.extend([sorted_tf[i][0] for i in range(max_size)])
        return tmp_vocab
            

    def construct_label_vocab(self,tmp_dataset):
        print ("construct label vocab....")
        label_list = tmp_dataset[1]
        label_set = set(label_list)
        label_dict = {}
        tmp_i = 0
        for label in label_set:
            label_dict[label]=tmp_i
            tmp_i += 1
        print (label_dict)
        print ('bingo!')
        return label_dict

    def stoi(self,tmp_str):
        for i in range (self.vocab_len):
            if self.vocab[i] == tmp_str:
                return i
        return 0     # unk
            
    def itos(self,tmp_int):
        if tmp_int<self.vocab_len:
            return self.vocab[tmp_int]
        else:
            return 'error!!!'


def construct_embedding(config,vocab_class):
    """
    将一个词表list映射成一个词嵌入列表
    ref: https://blog.csdn.net/nlpuser/article/details/83627709
    """
    print ("construct embedding ...")
    kept_num = 0
    embedding = torch.randn(vocab_class.vocab_len,config.embed_dim,dtype=torch.float)
    tmp_line = 0 
    with open(config.zh_embed_pretrained,'r',encoding='utf8') as embed_f:
        while True:
            line = embed_f.readline()
            if line:
                if tmp_line%100000 == 0:
                    print ("    processing line ", tmp_line)
                tmp_line += 1
                line_splited = line[:-1].split(' ')
                word = line_splited[0]
                if word in vocab_class.vocab:
                    index = vocab_class.stoi(word)
                    # print (line_splited[2],type(line_splited[2]))
                    vector = np.array([float(i) for i in line_splited[1:-1]])  # 最后一个是空格
                    embedding[index,:] = torch.from_numpy(vector)
                    kept_num += 1
            else:
                break
    print ("    ",kept_num,"/",vocab_class.vocab_len, " word kept!")
    return embedding  # 还没有完成


def build_iterator_base(config,vocab_class,tmp_data):
    """
    构建迭代器，并将长度相近的组成一个batch
    """
    print ("build iterator ...")
    batch_size = config.batch_size
    print ("    batch size: ", batch_size)
    tmp_contents = tmp_data[0]
    tmp_labels = tmp_data[1] 
    tmp_lens = tmp_data[2] 

    new_contents = []                  # 将词转化为词表中的序号
    op_num = 0
    for sent in tmp_contents:
        if op_num%10000==0:
            print ("    process sent num ",str(op_num)," ing")
        op_num +=1
        new_sent = []
        for word in sent:
            new_sent.append(vocab_class.stoi(word))
        new_contents.append(new_sent)

    new_labels = []
    for label in tmp_labels:
        new_labels.append(vocab_class.label_vocab[label])  
    
    sorted_lens = sorted(range(len(tmp_lens)), key=lambda k: tmp_lens[k],reverse= True)     # 或许可以不同reverse
    print("    max length: ",tmp_lens[sorted_lens[0]])
    print("    min length: ",tmp_lens[sorted_lens[-1]])
    
    batches = []
    batch_labels = []
    for i in range(0,len(sorted_lens),batch_size):
        batch = []
        labels = []
        tmp_batch_len = tmp_lens[sorted_lens[i]]     # 当前batch的长度
        # print ("tmp batch len: ", tmp_batch_len)
        if (i+batch_size)<len(sorted_lens):
            for j in range(i,i+batch_size):
                batch.append(new_contents[sorted_lens[j]])
                labels.append(new_labels[sorted_lens[j]])
        else:
            for j in range(i,len(sorted_lens)):
                batch.append(new_contents[sorted_lens[j]])
                labels.append(new_labels[sorted_lens[j]])
        for item in batch:
            while len(item)<tmp_batch_len:
                item.append(vocab_class.stoi('<PAD>'))
        batches.append(batch)
        batch_labels.append(labels)
    print ("    batch num: ",len(batches))
    return iter(zip(batches,batch_labels))


def build_iterator(config,vocab_class,train_data,dev_data,test_data):
    """
    构建迭代器(生成器)
    :return:
    """
    train_it = build_iterator_base(config,vocab_class,train_data)
    dev_it = build_iterator_base(config,vocab_class,dev_data)
    test_it = build_iterator_base(config,vocab_class,test_data)
    return train_it,dev_it,test_it
