"""

配置文件

"""


class Config():
    def __init__(self,datasets_path,language,model_name,embed,using_word):
        # 自定义参数
        self.train_path = datasets_path + 'train.txt'
        self.test_path = datasets_path + 'test.txt'
        self.dev_path = datasets_path + 'dev.txt'
        self.language = language
        self.model_name = model_name
        self.embedding = embed           # 'random' or 'pre_trained'  
        self.using_word = using_word
        
        # 公用参数
        self.class_num = -1             # 读取数据的时候自动读入并修改
        self.epoch_num = 10
        self.dropout = 0.8
        self.learning_rate = 1e-3 
        self.batch_size = 64  # 64              # batch size  不超过128
        self.embed_dim = 300
        self.max_vocab_size = 15000       # 词表大小
        self.unk = '<UNK>'
        self.pad = '<PAD>'
        self.zh_embed_pretrained = 'data/w2v/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
        self.en_embed_pretrained = ''
        self.embedding_pretrained = ''
        self.using_word = True

        # self.require_improvement = 1000
 
        # for TextRNN
        self.num_layers = 2
        self.hidden_size = 128
