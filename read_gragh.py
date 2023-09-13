"""
    读取图（邻接矩阵）相关数据
"""

import pickle

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

#示例代码
matrix = read_matrix()
vocab_index = read_vocab_index()
i_word = 'good' # 下标i对应的词
j_word = 'bad'  # 下标j对应的词
if i_word in vocab_index.keys() :#词汇表中存在这俩词
    if j_word in vocab_index.keys():
        print(matrix[vocab_index[i_word],vocab_index[j_word]]) #读取这俩词在矩阵中的值
    else:
        print('word: "'+j_word+'" not exists in vocab!')
else:
    print('word: "'+i_word+'" not exists in vocab!')