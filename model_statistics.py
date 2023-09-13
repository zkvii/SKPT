
"""
    根据每个模型的metrics_result.txt生成包含所有模型机器指标结果统计的csv文件:model_statistics.csv
"""
import pandas as pd
models = ['cem_base','cem_en_rels','cem_en_rels_token','cem_en_rels_token_senti','cem_en_rels_token_senti_loss',
          'cem_en_token','empdg','mime','moel','cem_en_rels_token_senti_loss0','cem_en_rels_token_senti_loss1',
          'cem_en_rels_token_senti_loss2','cem_en_rels_token_senti_loss3','cem_en_rels_token_senti_loss4',
          'cem_en_rels_token_senti_loss5','cem_en_rels_token_senti_loss6','cem_en_rels_token_senti_loss7',
          'cem_en_rels_token_senti_loss8']

paths = ['/data/liukai/space/EMNLP2022-SEEK/save/seek/','/data/liukai/space/CEDual/Predictions/'] #这俩模型路径不同，需要单独标出来
temp = ['/data/liukai/space/CEM/save/test/'+model+'/' for model in models]
metrics_map = {'beam blue-1':1,'beam blue-2':2,'beam blue-3':3,'beam blue-4':4,'beam rouge-l':5,'beam dist-1':6, 
           'beam dist-2':7, 'greedy blue-1':8, 'greedy blue-2':9,'greedy blue-3':10,'greedy blue-4':11 ,
           'greedy rouge-l':12, 'greedy dist-1':13,'greedy dist-2':14}
rows = []
paths.extend(temp)  #所有模型的保存路径
for path in paths:
    row = [-1.0 for _ in range(len(metrics_map.keys())+1)]
    with open(path+'metric_result.txt','r') as file: #寻找保存路径下的机器指标结果metric_result.txt
        lines=file.readlines()
    for line in lines:
        if line.replace('\n','')!='':
            if line.split(':')[0].strip() == 'model':
                row[0] = line.split(':')[1].replace('\n','').strip()
            else:
                row[metrics_map[line.split(':')[0].strip()]] = float(line.split(':')[1].replace('\n','').strip())
    rows.append(row)

df = pd.DataFrame(rows,columns=['model','beam blue-1','beam blue-2','beam blue-3','beam blue-4','beam rouge-l',
                                'beam dist-1', 'beam dist-2', 'greedy blue-1', 'greedy blue-2','greedy blue-3'
                                ,'greedy blue-4' ,'greedy rouge-l', 'greedy dist-1','greedy dist-2'])
df.to_csv('model_statistics.csv',encoding='utf-8',index=0) #生成包含统计结果的csv
print('done!')