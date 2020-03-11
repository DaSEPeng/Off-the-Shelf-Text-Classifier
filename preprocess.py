"""

将数据集和词向量处理成相应的模式

- 将数据集分割成训练集、验证集、测试集
    - 格式：分词后的文本+空格+类别
	- 例子：
        中华女子学院：本科层次仅1专业招男生 education
	    两天价网站背后重重迷雾：做个网站究竟要多少钱 science
	    东5环海棠公社230-290平2居准现房98折优惠 realty
	    卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球 sports
	    82岁老太为学生做饭扫地44年获授港大荣誉院士 society

@auther: DaSEPeng
@date: 2020/3/3
"""

# import jieba                # 注意jieba.cut()返回的是生成器，只能使用一次

## data path
data_path = 'data/'
# raw data path
raw_data_path = data_path + 'raw/'
raw_train_data_path = raw_data_path + 'raw_train.txt'
raw_dev_data_path = raw_data_path + 'raw_dev.txt'
raw_test_data_path = raw_data_path + 'raw_test.txt'
raw_class_data_path = data_path + 'raw_class.txt'
# raw_embedding_path = data_path + 'sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
# formal data path
formal_data_path = data_path + 'formal/'
formal_train_data_path = formal_data_path + 'train.txt'
formal_test_data_path = formal_data_path + 'test.txt'
formal_dev_data_path = formal_data_path + 'dev.txt'
# filtered_embedding_data_path = raw_embedding_path + '.filtered'  # 压缩方式对比：https://blog.csdn.net/jerr__y/article/details/74230765
vocab_data_path = data_path + 'vocab_list.txt'


def preprocess_base(raw_data_path,formal_data_path,class_path,type,vocab_path):
    """
    将文件处理成标准格式
    :param raw_data_path: 原始格式数据集
    :param formal_data_path: 标准格式数据集
    :param class_path: 类别信息
    :param type: 训练集/验证集/测试集
    :param vocab_path: 词表路径
    :return: 无
    """

    ## 构建类别词典
    class_dict = {}
    class_num = 0
    with open(class_path,'r',encoding='utf-8',errors='ignore') as class_f:
        while True:
            tmp_class = class_f.readline()                 # 注意有回车键
            if tmp_class:
                tmp_class = tmp_class.rstrip("\n")
                class_dict[class_num] = tmp_class    # 注意这里的键值对顺序
                class_num += 1
            else:
                break
    # print ("There are " + str(class_num) + " classes.")

    ## 逐行处理文件
    test_num = 0
    with open(raw_data_path,'r',encoding='utf-8',errors='ignore') as raw_data_f:
        with open(formal_data_path,'w',encoding='utf-8') as formal_data_f:
            vocab_set = set()
            while True:
                tmp_line = raw_data_f.readline()                         # 防止文件过大，不用readlines()，但是会慢很多
                if tmp_line:
                    tmp_line_content = tmp_line[:-2].rstrip()            # 文本内容，除去类别信息
                    # tmp_line_content_cut = jieba.cut(tmp_line_content)   # 返回的是迭代器(generator)，没有加载到内存
                    #
                    # # 注意，只在训练集的时候才构建词表，因此需要测试集/验证集的词汇尽量与训练集相似
                    # if type == 'train':
                    #     tmp_list = [i for i in tmp_line_content_cut]
                    #     for item in tmp_list:
                    #         vocab_set.add(item)
                    #     tmp_line_content_cut = tmp_list                                # 此时已经成了list，不是generator
                    #
                    # tmp_line_content_cut_str = ' '.join(tmp_line_content_cut)
                    tmp_class_num = tmp_line[-2]                                       # 字符串类型，内容是数字
                    tmp_class_str = class_dict[int(tmp_class_num)]

                    new_line = tmp_line_content + ' ' + tmp_class_str + '\n'
                    # 写入新文件
                    formal_data_f.write(new_line)
					
                # test_num+=1
                # 下面这部分用来测试
                # if test_num == 5:
                # break
                else:
                    break

    ## 如果是训练集，就存储词表
    # if type == 'train':
    #     with open(vocab_path,'w',encoding='utf-8') as f:
    #         for item in vocab_set:
    #             f.write(item)
    #             f.write('\n')
    return 

def preprocess(raw_train_path,raw_dev_path,raw_test_path,\
               formal_train_path,formal_dev_path,formal_test_path,\
               raw_class_path,vocab_path):
    """
    处理训练集/测试集/验证集
    :param raw_train_path:
    :param raw_dev_path:
    :param raw_test_path:
    :param formal_train_path:
    :param formal_dev_path:
    :param formal_test_path:
    :param raw_class_path:
    :param vocab_path:
    :return:
    """
    preprocess_base(raw_train_path,formal_train_path,raw_class_path,\
                    type='train',vocab_path=vocab_path)   # 只有训练集才构建词典
    preprocess_base(raw_dev_path,formal_dev_path, raw_class_path,\
                    type='dev',vocab_path=vocab_path)
    preprocess_base(raw_test_path,formal_test_path,raw_class_path,\
                    type='test',vocab_path=vocab_path)
    return


# def filter_embedding(raw_embedding_path,filtered_embedding_path,vocab_path):
#     """
#     从原始词向量集合中过滤出要用到的词向量，方便后面读取使用
#     :param raw_embedding_path: 原始词向量集合
#     :param filtered_embedding_path: 过滤后的词向量集合
#     :param vocab_path: 词表路径
#     :return: 无
#     """
#     ## 获得词表集合
#     vocab_set = set()
#     with open(vocab_path,'r',encoding='utf-8') as vocab_f:
#         while True:
#             line = vocab_f.readline()
#             if line:
#                 word = line.rstrip('\n')
#                 vocab_set.add(word)
#             else:
#                 break
#
#     ## 从词向量库中过滤出词表对应的词向量
#     total_lines = 0
#     kept_lines = 0
#     with open (raw_embedding_path,'r') as in_f:
#         with open (filtered_embedding_path, 'w') as out_f:              # 注意这里的逻辑，减少数据交换
#             while True:
#                 line = in_f.readline()
#                 if line:
#                     total_lines += 1
#                     word = line.split()[0]
#                     if word in vocab_set:
#                         kept_lines += 1
#                         out_f.write(line)
#                 else:
#                     break
#     print ("Filtered " + str(kept_lines) + " from " + str(total_lines))
#     return


if __name__=="__main__":
    ## 预处理数据集
    preprocess(raw_train_data_path,raw_dev_data_path,raw_test_data_path, \
                formal_train_data_path, formal_dev_data_path, formal_test_data_path, \
                raw_class_data_path,vocab_data_path)

    # ## 过滤词向量： 75839词
    # filter_embedding(raw_embedding_path,filtered_embedding_data_path,vocab_data_path)
