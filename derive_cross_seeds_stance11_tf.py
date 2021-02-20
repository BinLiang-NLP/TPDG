#!/usr/bin/env python3
# -*- coding:UTF8 -*-
# ------------------
# @File Name: preprocess_data.py
# @Version: 
# @Author: BinLiang
# @Mail: 18b951033@stu.hit.edu.cn
# @For: 考虑tf 
# @Created Time: Mon 10 Aug 2020 03:15:03 PM CST
# ------------------

import nltk
import math
import numpy as np

TARGET_DIC_ALL = {'Climate Change is a Real Concern': 0, 'Donald Trump': 1,
                  'Feminist Movement': 2, 'Hillary Clinton': 3, 'Legalization of Abortion': 4,'Trade Policy': 5}
#TARGET_DIC = {'Donald Trump': 0, 'Feminist Movement': 1, 'Hillary Clinton': 2, 'Legalization of Abortion': 3}

TARGET_DIC = {'Hillary Clinton': 0, 'Trade Policy': 1}
LABEL_DIC = {'AGAINST': 0, 'FAVOR': 1}
# TARGET_DIC = {'Feminist Movement': 0, 'Legalization of Abortion': 1}

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout = open('./seed_words/seed_word.domain.HCTP.stance.3way11.tf', 'w', encoding='utf-8')
    word_freq_dic = {}
    word_stance_dic = {}
    label_count = [0, 0]
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = line.split('\t')
        target = item[1]
        text = ' '.join(item[2:-1])
        text = ' '.join(nltk.word_tokenize(text)).lower()
        if target not in TARGET_DIC:
            continue
        stance = item[-1]
        #if stance not in LABEL_DIC:
            #continue
        if stance in LABEL_DIC:
            label_count[LABEL_DIC[stance]] += 1
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
    seed_weight = {}
    min_weight = 100000
    max_weight = 0
    weight_list = []
    stance_weight_list = []
    stance_weight_dic = {}
    for word in word_stance_dic:
        freq_list = word_stance_dic[word]
        weight = freq_list[0] / label_count[0] - freq_list[1] / label_count[1]
        #weight = freq_list[0] - freq_list[1]
        stance_weight_dic[word] = weight
        stance_weight_list.append(weight)

    mu = np.mean(stance_weight_list)
    sigma = np.std(stance_weight_list)
    _range = np.max((np.abs(stance_weight_list)))

    for word in word_freq_dic:
        freq_list = word_freq_dic[word]
        max_f = max(freq_list)
        min_f = min(freq_list)
        weight = (sum(freq_list)/len(TARGET_DIC))/(max_f-min_f+1)
        seed_weight[word] = weight
        if weight > max_weight:
            max_weight = weight
        if weight < min_weight:
            min_weight = weight
        weight_list.append(weight)
    seed_weight = sorted(seed_weight.items(), key= lambda a: -a[1])
    
    #mu = np.mean(weight_list)
    #sigma = np.std(weight_list)
    for (seed, weight) in seed_weight:
        if seed in stance_weight_dic:
            print(stance_weight_dic[seed])
            stance_weight = (stance_weight_dic[seed]-mu) / sigma
            stance_weight = 1 + stance_weight_dic[seed] / _range
            #stance_weight = 1 + stance_weight_dic[seed]
            print(seed, stance_weight)
        else:
            stance_weight = 1
        weight = (weight-min_weight) / (max_weight-min_weight)
        weight *= stance_weight
        #weight = (weight-mu) / sigma
        string = seed + '\t' + str(weight) + '\n'
        fout.write(string)
    fout.close()

if __name__ == '__main__':
    process('./raw_data/all.orig')
