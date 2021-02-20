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
from nltk.tokenize import TweetTokenizer
import json

label_dic = {'AGAINST': '-1', 'NONE': '0', 'FAVOR': '1'}



def process(filename):
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fout = open(filename+'.raw', 'w', encoding='utf-8')
    for line in fin:
        line = line.strip()
        if not line:
            continue
        item = line.split('\t')
        target = item[1].lower()
        text = ' '.join(item[2:-1])
        text = ' '.join(nltk.word_tokenize(text)).lower()
        label = label_dic[item[-1]]
        string = text + '\n' + target + '\n' + label + '\n'
        fout.write(string)
    fin.close()
    fout.close()


def process_wtwt(filename,merger):
    total_merger_set = {'CVS_AET','CI_ESRX','ANTM_CI','AET_HUM'}
    N_set = total_merger_set - {merger[2:]}
    tokenizer = TweetTokenizer()
    label_dic = {'refute': '-1', 'comment': '0', 'support': '1','unrelated':'2'}
    fin = open(filename, 'r', encoding='utf-8', errors='ignore')
    fin_data = json.load(fin)
    fout = open("./raw_data/"+merger.lower()+'.raw', 'w', encoding='utf-8')
    for line in fin_data:
        # line = line.strip()
        # if not line:
        #     continue
        # item = line.split('\t')
        target = line['merger']
        # target = item[1].lower()
        text = line['text']
        stance = line['stance']
        # text = ' '.join(item[2:-1])
        text = ' '.join(tokenizer.tokenize(text)).lower()
        text = text.replace('\n', ' ')
        text = text.replace('\u2026',' ')
        label = label_dic[stance]
        string = text + '\n' + target + '\n' + label + '\n'
        if merger[0]!='N' and target==merger:
            fout.write(string)
        elif merger[0]=='N' and target in N_set:
            fout.write(string)
        else:
            pass

    fin.close()
    fout.close()

if __name__ == '__main__':
    #process('./raw_data/a')
    # process('./raw_data/cc')
    # process('./raw_data/dt')
    # process('./raw_data/fm')
    # process('./raw_data/hc')
    # process('./raw_data/la')
    # process("./raw_data/tp")
    process("./raw_data/all.orig.bak")
    # process_wtwt("./raw_data/wtwt_with_text.json","CI_ESRX")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","N_CI_ESRX")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","CVS_AET")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","N_CVS_AET")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","ANTM_CI")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","N_ANTM_CI")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","AET_HUM")## #2 target name
    # process_wtwt("./raw_data/wtwt_with_text.json","N_AET_HUM")## #2 target name
