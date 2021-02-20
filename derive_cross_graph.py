# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

nlp = spacy.load('en_core_web_sm')


def load_seed_word(path):
    seed_words = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, weight = line.split('\t')
        seed_words[word] = weight
    fp.close()
    return seed_words


def dependency_adj_matrix(text, seed_words):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    
    for token in document:
        if str(token) in seed_words:
            weight = float(seed_words[str(token)])
        else:
            weight = 1
        if token.i < seq_len:
            matrix[token.i][token.i] = 1 * weight * 2
            # https://spacy.io/docs/api/token
            for child in token.children:
                if str(child) in seed_words:
                    weight += float(seed_words[str(child)])
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1 * weight
                    matrix[child.i][token.i] = 1 * weight
    #print(matrix)
    #print('='*30)
    return matrix

def process(filename, seedname):
    seed_words = load_seed_word(seedname)
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph.cross.stance.3way11t', 'wb')
    graph_idx = 0
    for i in range(0, len(lines), 3):
        text = lines[i].lower().strip()
        target = lines[i+1].lower().strip()
        adj_matrix = dependency_adj_matrix(text, seed_words)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    print('done !!!'+filename)
    fout.close() 

if __name__ == '__main__':
    # process('./raw_data/fm.raw', './seed_words/seed_word.domain.FMLA.stance.3way11.tf')
    # process('./raw_data/la.raw', './seed_words/seed_word.domain.FMLA.stance.3way11.tf')
    # process('./raw_data/dt.raw', './seed_words/seed_word.domain.DTHC.stance.3way11.tf')
    # process('./raw_data/hc.raw', './seed_words/seed_word.domain.DTHC.stance.3way11.tf')
    process('./raw_data/tp.raw', './seed_words/seed_word.domain.HCTP.stance.3way11.tf')
    process('./raw_data/hc.raw', './seed_words/seed_word.domain.HCTP.stance.3way11.tf')


