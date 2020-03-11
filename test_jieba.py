# import jieba
#
# sent = "我来测试一下jieba分词的存储问题"
#
# sent_list = jieba.cut(sent)
# print (type(sent_list))
# sent_list = [i for i in sent_list]
#
# for i in sent_list:
#     print (i)
#
# print ("俺来测试一下")
#
# for j in sent_list:
#     print (j)

# a = [['a','c','2','d'],['a','x']]
# b = [2,3]

# c = (a,b)
# print (c)
# print (c[0],c[1])


import test_aug
import numpy as np
import torch 
# def process_iter(it):
#     print (next(it))

a = iter(zip([[1,1],[2,2],[3,3]],[[4,4],[6,6],[6,6]]))
# print (next(a))
test_aug.process_it(a)
print ("*"*100)

#b,c = next(a)
# print (torch.tensor(b,dtype=torch.long).dtype)
#print (b)
#print (c)

# import torch
# import torch.nn as nn


