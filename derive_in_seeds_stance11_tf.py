#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: preprocess_data.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 
# @Created Time: Mon 10 Aug 2020 03:15:03 PM CST
# ------------------

import nltk
import math
import numpy as np

# TARGET_DIC_ALL = {'Climate Change is a Real Concern': 0, 'Donald Trump': 1,
#                   'Feminist Movement': 2, 'Hillary Clinton': 3, 'Legalization of Abortion': 4, 'Trade Policy': 5}
# TARGET_DIC_ALL = {'Climate Change is a Real Concern': 0, 'Donald Trump': 1,
#                   'Feminist Movement': 2, 'Hillary Clinton': 3, 'Legalization of Abortion': 4, 'Trade Policy': 5}
#TARGET_DIC = {'Donald Trump': 0, 'Feminist Movement': 1, 'Hillary Clinton': 2, 'Legalization of Abortion': 3}
#TARGET_DIC = {'Feminist Movement': 0, 'Legalization of Abortion': 1}

TARGET_DIC = {'Hillary Clinton': 0, 'Trade Policy': 1}
INDEX = 1

LABEL_DIC = {'AGAINST': 0, 'FAVOR': 1}

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout = open('./seed_words/seed_word.domain.TP.stance.3way11.tf', 'w', encoding='utf-8')
    word_freq_dic = {}
    word_stance_dic = {}
    label_count = [0, 0]
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = line.split('\t')
        target = item[1]
        if target not in TARGET_DIC:
            continue
        stance = item[-1]
        #if stance not in LABEL_DIC:
            #continue
        if stance in LABEL_DIC:
            label_count[LABEL_DIC[stance]] += 1
        text = ' '.join(item[2:-1])
        text = ' '.join(nltk.word_tokenize(text)).lower()
        for word in set(text.split()):
            if word not in word_freq_dic:
                word_freq_dic[word] = [0, 0]
                word_freq_dic[word][TARGET_DIC[target]] += 1
            else:
                word_freq_dic[word][TARGET_DIC[target]] += 1
            if stance not in LABEL_DIC:
                continue
            if word not in word_stance_dic:
                word_stance_dic[word] = [0, 0]
                word_stance_dic[word][LABEL_DIC[stance]] += 1
            else:
                word_stance_dic[word][LABEL_DIC[stance]] += 1
    fin.close()

    stance_weight_list = []
    stance_weight_dic = {}
    max_stance_weight = 0
    min_stance_weight = 100000
    for word in word_stance_dic:
        freq_list = word_stance_dic[word]
        weight = freq_list[0] - freq_list[1]
        weight = freq_list[0] / label_count[0] - freq_list[1] / label_count[1]
        stance_weight_dic[word] = weight
        stance_weight_list.append(weight)
        if weight > max_stance_weight:
            max_stance_weight = weight
        if weight < min_stance_weight:
            min_stance_weight = weight

    print(max_stance_weight, min_stance_weight)
    mu = np.mean(stance_weight_list)
    sigma = np.std(stance_weight_list)
    _range = np.max((np.abs(stance_weight_list)))

    min_weight = 100000
    max_weight = 0
    seed_weight = {}
    for word in word_freq_dic:
        freq_list = word_freq_dic[word]
        weight = freq_list[INDEX] / (sum(freq_list)+1)
        seed_weight[word] = weight
        if weight > max_weight:
            max_weight = weight
        if weight < min_weight:
            min_weight = weight
    seed_weight = sorted(seed_weight.items(), key= lambda a: -a[1])
    
    for (seed, weight) in seed_weight:
        if seed in stance_weight_dic:
            stance_weight = (stance_weight_dic[seed]-mu) / sigma
            stance_weight = 1 + stance_weight_dic[seed] / _range
            #stance_weight = -1 + ((2/(max_stance_weight-min_stance_weight)) * (stance_weight_dic[seed]-min_stance_weight))
            print(seed, stance_weight)
        else:
            stance_weight = 1
        weight = (weight-min_weight) / (max_weight-min_weight)
        weight *= stance_weight
        string = seed + '\t' + str(weight) + '\n'
        fout.write(string)
    fout.close()

if __name__ == '__main__':
    process('./raw_data/mul_all.orig')
