"""
    建立大图，以邻接矩阵的形式保存，同时保存词汇表、词汇表索引、词汇表与出现次数的映射、处理后的context、处理后的target
    保存后的图以及其他数据可以通过同目录下的read_gragh.py读取
"""

from src.utils import config
import os
import pickle
from tqdm import tqdm
import numpy as np
from scipy.sparse import dok_matrix
from collections import defaultdict
import enchant


#建立邻接矩阵的函数
def build_adjacency_matrix(context, response, vocab,count_map):
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    matrix = dok_matrix((len(vocab), len(vocab)), dtype=np.float32)
    
    for ctx_words, rsp_words in tqdm(zip(context, response),desc='matrix'):
        ctx_set = set(ctx_words)
        rsp_set = set(rsp_words)
        
        for ctx_word in ctx_set:
            ctx_idx = vocab_index.get(ctx_word)
            if ctx_idx is not None:
                ctx_count = count_map[ctx_word]
                for rsp_word in rsp_set:
                    rsp_idx = vocab_index.get(rsp_word)
                    if rsp_idx is not None:
                        matrix_value = 1.0 / ctx_count  # 除以i对应的词的出现次数
                        matrix[ctx_idx, rsp_idx] += matrix_value
    return matrix,vocab_index

english_dict = enchant.Dict("en_US") #英文词典，用于检查单词是否合法
save_path = '/data/liukai/space/CEM/map/' #图的保存路径
data_dir = config.data_dir  #源数据路径
cache_file = f"{data_dir}/dataset_preproc.p"
if os.path.exists(cache_file):
    print("LOADING empathetic_dialogue")
    with open(cache_file, "rb") as f:
        [data_tra, data_val, data_tst, _] = pickle.load(f)

#处理源数据，把训练集、测试集、验证集合起来
context_all = []
context_all.extend(data_tra['context'])
context_all.extend(data_val['context'])
context_all.extend(data_tst['context'])
target_all = []
target_all.extend(data_tra['target'])
target_all.extend(data_val['target'])
target_all.extend(data_tst['target'])
cs_all = []
cs_all.extend(data_tra['utt_cs'])
cs_all.extend(data_val['utt_cs'])
cs_all.extend(data_tst['utt_cs'])

#将utt_cs加到context当中，把context变成和target一样的维度：[[word1,word2,...],[word3,...]]
len_samples = len(context_all)
for i in range(len_samples):
    context_all[i] = [item for sublist in context_all[i] for item in sublist]
    cs_all[i] = [item for sublist1 in cs_all[i] for sublist2 in sublist1 for item in sublist2]
    context_all[i].extend(cs_all[i])

#根据新的context建立词汇表，通过英文词典来确保每个词汇有意义
dataset_vocabs = []
print('building vocab ...')
dataset_vocabs.extend([item for sublist in tqdm(context_all,desc='vocab') for item in sublist if english_dict.check(item)])

#建立 词汇表-->出现次数 的映射
dataset_vocabs_map = {}
print('creating vocab map ...')
for word in tqdm(dataset_vocabs,desc='map'):
    if word not in dataset_vocabs_map.keys():
        dataset_vocabs_map[word] = 1
    else:
        dataset_vocabs_map[word] += 1

#对词汇表进行排序,确保每次运行顺序一致
sorted_vocab = sorted(dataset_vocabs)

#构建邻接矩阵,得到词汇表的索引
adjacency_matrix,vocab_index = build_adjacency_matrix(context_all, target_all, sorted_vocab,dataset_vocabs_map)


# 保存邻接矩阵
print('saving adjacency_matrix ...')
with open(save_path+'matrix.pkl', 'wb') as f:
    pickle.dump(adjacency_matrix, f)
# 保存排序后的词汇表
print('saving vocab ...')
with open(save_path+'vocab.pkl', 'wb') as f:
    pickle.dump(sorted_vocab, f)
# 保存 词汇表-->出现次数 的映射
print('saving vocab-->cnt map ...')
with open(save_path+'vocab_cnt_map.pkl', 'wb') as f:
    pickle.dump(dataset_vocabs_map, f)
#保存词汇表索引
print('saving vocab_index ...')
with open(save_path+'vocab_index.pkl', 'wb') as f:
    pickle.dump(vocab_index, f)
#保存context
print('saving context ...')
with open(save_path+'context.pkl', 'wb') as f:
    pickle.dump(context_all, f)
#保存target
print('saving target ...')
with open(save_path+'target.pkl', 'wb') as f:
    pickle.dump(target_all, f)

# # 加载邻接矩阵
# with open('adjacency_matrix.pkl', 'rb') as f:
#     loaded_adjacency_matrix = pickle.load(f)

# # 加载词汇表
# with open('vocab.pkl', 'rb') as f:
#     loaded_vocab = pickle.load(f)
    
# # 加载词汇表索引
# with open('word_to_index.pkl', 'rb') as f:
#     loaded_word_to_index = pickle.load(f)

print('done!')






