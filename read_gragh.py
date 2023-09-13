"""
    读取图（邻接矩阵）相关数据
"""

import pickle
from torch import tensor
import torch
save_path = '/data/liukai/space/CEM/map/' #图的保存路径

#读取保存图的邻接矩阵，可以通过[i,j]下标来访问具体元素
def read_matrix():
    with open(save_path+'matrix.pkl', 'rb') as f:
        loaded_adjacency_matrix = pickle.load(f)
    return loaded_adjacency_matrix

#读取邻接矩阵对应的词汇表索引，可以通过vocab_index[word]来访问word对应的索引值
def read_vocab_index():
    with open(save_path+'vocab_index.pkl', 'rb') as f:
        vocab_index = pickle.load(f)
    return vocab_index

matrix = read_matrix()
vocab_index = read_vocab_index()

def wordlist_to_keyword_vec(wordlist):
    
    word_len=len(wordlist)
    
    word_matrix=torch.zeros(word_len,word_len)
    for i in range(word_len):
        for j in range(word_len):
            if wordlist[i] in vocab_index.keys() and wordlist[j] in vocab_index.keys() :#词汇表中存在这俩词
                word_matrix[i][j]=torch.tensor(matrix[vocab_index[wordlist[i]],vocab_index[wordlist[j]]])
    return word_matrix

# if __name__ == "__main__":
#     wordlist = ['good','bad','happy','sad']
#     word_matrix=wordlist_to_keyword_vec(wordlist)
#     print(word_matrix)
#示例代码
# i_word = 'good' # 下标i对应的词
# j_word = 'bad'  # 下标j对应的词
# if i_word in vocab_index.keys() :#词汇表中存在这俩词
#     if j_word in vocab_index.keys():
#         print(matrix[vocab_index[i_word],vocab_index[j_word]]) #读取这俩词在矩阵中的值
#     else:
#         print('word: "'+j_word+'" not exists in vocab!')
# else:
#     print('word: "'+i_word+'" not exists in vocab!')
