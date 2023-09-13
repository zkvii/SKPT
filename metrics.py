from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score, meteor_score
from nltk import word_tokenize
import pandas as pd
import numpy as np
from tqdm import tqdm
from rouge import Rouge
from tabulate import tabulate
from typing import List
from utils.cider.pyciderevalcap.cider.cider import Cider
from utils.cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from utils.cider.pydataformat.loadData import LoadData
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
from aac_metrics.functional import cider_d
from aac_metrics import evaluate
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
from aac_metrics.utils.checks import check_metric_inputs
from typing import Union, List, Iterable, Callable, Any
from torch import Tensor
import torch

"""
    根据模型保存目录下的 result.txt 生成机器指标计算结果metric_result.txt,每次需要指定一个模型
    有些模型的结果没有beam只有greedy,只需要注释掉所有beam相关的操作再运行即可
"""
# models = ['cem_en','cem_base','cem_en_rels','cem_en_rels_token','cem_en_rels_token_senti','cem_en_rels_token_senti_loss6','cem_en_token','empdg','mime','moel']

# models = ['cem_base', 'empdg', 'mime', 'moel', 'seek', 'gpt3', 'chatgpt','cem_en_rels_token_senti_loss6']
models = ['cem_base']
# model = models[5]     #要计算机器指标的模型的序号
# path = '/data/liukai/space/CEM/save/test/'+model+'/' #寻找模型目录
# test
paths = ['/data/liukai/space/CEM/save/test/'+model+'/' for model in models]

"""
    cider score
"""
metric_headers = ['Model', 'Bleu-1', 'Bleu-2', 'Bleu-3',
                  'Bleu-4', 'Rouge-L', 'Dist-1', 'Dist-2', 'Meteor', 'CIDEr']
metric_table = []
# def calc_cider(candidates, references, print_score: bool = True): #计算cider的函数

interpunctuations = [',', '.', ':', ';', '?',
                     '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '‘', '’', '-']  # 定义标点符号列表



def calc_distinct_n(n, candidates, print_score: bool = True):  # 计算dist-n的函数
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i: i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score

# def calc_meteor(refs,hypos):
#     meteor_scores=[]
#     for ref,hypo in zip(refs,hypos):
#         meteor_scores.append(single_meteor_score(ref,hypo))
#     return np.mean(meteor_scores)


def metric_models(paths: List, models: List):

    metrics = {}
    model_result_dict = {}
    for model_path, model_name in zip(paths, models):
        text = ''
        rouge = Rouge()
        with open(model_path+'results.txt') as file:
            raw_lines = file.readlines()
            for raw_line in raw_lines:
                text += raw_line
        blocks = text.split(
            '---------------------------------------------------------------\n')

        ref = []
        beam = []
        greedy = []
        gpt = []
        chatgpt = []
        for block in blocks:
            lines = block.split('\n')
            for line in lines:
                if 'Beam' in line.split(':')[0]:
                    beam.append(line.split(':')[1].replace(
                        '\n', ''))  # 得到beam的结果
                if 'Greedy' in line.split(':')[0]:
                    greedy.append(line.split(':')[1].replace(
                        '\n', ''))  # 得到greedy的结果
                if 'Ref' in line.split(':')[0]:
                    ref.append(line.split(':')[1].replace(
                        '\n', ''))  # 得到ref的结果
                if 'Davinci_result' in line.split(':')[0]:
                    gpt.append(line.split(':')[1].replace(
                        '\n', ''))
                if 'Chatgpt_result' in line.split(':')[0]:
                    chatgpt.append(line.split(':')[1].replace(
                        '\n', ''))

        if model_name == 'gpt3':
            greedy = gpt
            beam = []
        if model_name == 'chatgpt':
            greedy = chatgpt
            beam = []

        # assert len(beam) == len(greedy)  # 确保一一对应
        if len(beam) == len(ref):
            # beam_ret = calc_bleu_n_rougue_l_dist_n(ref,  beam, False)
            # model_result_dict[model_name+'_beam'] = beam_ret
            beam_metric = aac_lib_calc(ref, beam)
            for k in beam_metric:
                if f'beam_{k}' in metrics:
                    metrics[f'beam_{k}'].append(beam_metric[k])
                else:
                    metrics[f'beam_{k}'] = [beam_metric[k]]

        # assert len(beam) == len(ref)
        if len(greedy) == len(ref):
            aac_lib_calc(ref, greedy)
            for k in beam_metric:
                if f'beam_{k}' in metrics:
                    metrics[f'beam_{k}'].append(beam_metric[k])
                else:
                    metrics[f'beam_{k}'] = [beam_metric[k]]
            # greedy_ret = calc_bleu_n_rougue_l_dist_n(ref, greedy, False)
            # model_result_dict[model_name+'_greedy'] = greedy_ret
    return model_result_dict, metrics


# tokenizer = PTBTokenizer('gts')
# # gts=load
# _gts = tokenizer.tokenize(gts)
# tokenizer = PTBTokenizer('res')
# _res = tokenizer.tokenize(res)
# #
# beam_cider=[]
# greedy_cider=[]


# beam_cider = calc_cider(beam,ref,False) #cider分数计算
# greedy_cider = calc_cider(greedy,ref,False)
rouge = Rouge()


def filter_empty(refs, candidates):
    filterred_refs = []
    filterred_cans = []
    for (ref, candidate) in zip(refs, candidates):
        if ref == '' or ref == None or candidate == '' or candidate == None:
            continue
        filterred_refs.append(ref)
        filterred_cans.append(candidate)
    return filterred_cans, filterred_refs


def aac_lib_calc(refs, candidates):
    filterred_cans, filterred_refs = filter_empty(refs, candidates)
    refs = [[ref] for ref in filterred_refs]
    print(len(candidates), len(filterred_cans))
    try:
        aac_metric, _ = evaluate(filterred_cans, refs, metrics=[
                                 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'meteor', 'rouge_l', 'spice', 'spider', 'cider_d'])
    except Exception as e:
        print(e)
    print(aac_metric)
    return aac_metric


def calc_bleu_n_rougue_l_dist_n(ref, hypo, print_score: bool = True):
    bleu_1 = []
    bleu_2 = []
    bleu_3 = []
    bleu_4 = []
    rouge_l = []
    dist_1 = 0
    dist_2 = 0
    meteor = []

    cider_d = 0
    candidates = hypo
    # candidates=[[candidate] for candidate in hypo]
    references = [[refer] for refer in ref]
    aac_cider, _ = evaluate(candidates, references, metrics=['cider_d'])
    cider_d = aac_cider['cider_d']
    for i in tqdm(range(len(ref))):
        ref_resp = ref[i]
        if ref_resp == None or ref_resp == '':  # 不能为空串，不然rouge计算会出错
            ref_resp = ' '
        ref_cutwords = word_tokenize(ref_resp)
        ref_references = [
            [word for word in ref_cutwords if word not in interpunctuations]]

        resp = hypo[i]
        resp_cutwords = word_tokenize(resp)
        resp_references = [
            word for word in resp_cutwords if word not in interpunctuations]

        # greedy_resp = greedy[i]
        # greedy_cutwords = word_tokenize(greedy_resp)
        # greedy_references = [
        #     word for word in greedy_cutwords if word not in interpunctuations]

        bleu_1.append(sentence_bleu(
            ref_references, resp_references, weights=(1, 0, 0, 0)))
        bleu_2.append(sentence_bleu(
            ref_references, resp_references, weights=(0.5, 0.5, 0, 0)))
        bleu_3.append(sentence_bleu(
            ref_references, resp_references, weights=(0.33, 0.33, 0.33, 0)))
        bleu_4.append(sentence_bleu(
            ref_references, resp_references, weights=(0.25, 0.25, 0.25, 0.25)))
        meteor.append(meteor_score(ref_references, resp_references))

        try:
            rouge_l.append(rouge.get_scores(
                hyps=resp, refs=ref_resp)[0]['rouge-l']['f'])
        except Exception as e:
            print(e)
            print(resp)
            print('-----------------')
            print(ref_resp)
    dist_1 = calc_distinct_n(1, hypo, False)
    dist_2 = calc_distinct_n(2, hypo, False)
    return [np.mean(bleu_1), np.mean(bleu_2), np.mean(bleu_3),
            np.mean(bleu_4), np.mean(rouge_l), dist_1, dist_2, np.mean(meteor), cider_d]


if __name__ == '__main__':
    model_metric_dict, aac_metrics = metric_models(paths, models)
    # for key in model_metric_dict:
    #     metric_table.append([key]+model_metric_dict[key])
    # print(tabulate(metric_table, headers=metric_headers, tablefmt='github'))
    print(tabulate(model_metric_dict, headers='keys', tablefmt='tsv'))
    print(tabulate(aac_metrics, headers='keys', tablefmt='tsv'))

    with open('metric_result.tsv', 'w') as f:
        f.write(tabulate(metric_table, headers=metric_headers, tablefmt='tsv'))

    with open('aac_metrics.tsv', 'w') as f:
        f.write(tabulate(aac_metrics, headers='keys', tablefmt='tsv'))
