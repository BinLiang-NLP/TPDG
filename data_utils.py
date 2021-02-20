# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from transformers import BertTokenizer

class MyBertTokenizer():

    @classmethod
    def from_pretrained(cls, path,*args,**kwargs):
        obj = cls()
        obj.bert_tokenizer = BertTokenizer.from_pretrained(path,*args,**kwargs)
        return obj

    def tokenize(self,sentence):
        tokens = []
        sentence = sentence.lower()
        for c in sentence.split(" "):
            if c in self.bert_tokenizer.vocab:
                tokens.append(c)
            else:
                tokens.append("[UNK]")
        return tokens

    def convert_tokens_to_ids(self,tokens):
        return self.bert_tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self,ids):
        return self.bert_tokenizer.convert_ids_to_tokens(ids)

    def encode(self,text):
        tokens = self.tokenize(text)
        return self.convert_tokens_to_ids(tokens)




def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            try:
                word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
            except:
                print('WARNING: corrupted word vector of {} when being loaded from GloVe.'.format(tokens[0]))
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


class Dataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class DatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        # fin = open(fname+'.graph.inver.stance', 'rb')
        # idx2gragh_inver = pickle.load(fin)
        # fin.close()
        if fname.split("/")[-1][0:3]=="mul":
            fin = open(fname + '.graph.in.mul', 'rb')
            idx2gragh = pickle.load(fin)
            fin.close()
            fin = open(fname + '.graph.cross.mul', 'rb')
            idx2gragh_cross = pickle.load(fin)
            fin.close()
        else:
            fin = open(fname+'.graph.stance.3way11t', 'rb')
            idx2gragh = pickle.load(fin)
            fin.close()
            fin = open(fname+'.graph.cross.stance.3way11t', 'rb')
            idx2gragh_cross = pickle.load(fin)
            fin.close()


        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            stance = lines[i + 2].strip()

            text_indices = tokenizer.text_to_sequence(text)
            target_indices = tokenizer.text_to_sequence(target)
            stance = int(stance)+1
            in_graph = idx2gragh[i]
            cross_graph = idx2gragh_cross[i]

            data = {
                'text': text,
                'target': target,
                'text_indices': text_indices,
                'target_indices': target_indices,
                'stance': stance,
                'in_graph': in_graph,
                'cross_graph': cross_graph,
            }

            all_data.append(data)
        return all_data

    def __init__(self, dataset='dt_hc', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'dt_hc': {
                'train': './raw_data/dt.raw',
                'test': './raw_data/hc.raw'
            },
            'hc_dt': {
                'train': './raw_data/hc.raw',
                'test': './raw_data/dt.raw'
            },
            'fm_la': {
                'train': './raw_data/fm.raw',
                'test': './raw_data/la.raw'
            },
            'la_fm': {
                'train': './raw_data/la.raw',
                'test': './raw_data/fm.raw'
            },
            'dt_tp': {
                'train': './raw_data/dt.raw',
                'test': './raw_data/tp.raw'
            },
            'tp_dt': {
                'train': './raw_data/tp.raw',
                'test': './raw_data/dt.raw'
            },
            'hc_tp': {
                'train': './raw_data/hc.raw',
                'test': './raw_data/tp.raw'
            },
            'tp_hc': {
                'train': './raw_data/tp.raw',
                'test': './raw_data/hc.raw'
            },
            'mbs_dt': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_dt.raw'
            },
            'mdt_bs': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_bs.raw'
            },
            'mbs_hc': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_hc.raw'
            },
            'mhc_bs': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_bs.raw'
            },'mbs_tc': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_tc.raw'
            },'mtc_bs': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_bs.raw'
            },'mdt_hc': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_hc.raw'
            },'mhc_dt': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_dt.raw'
            },'mdt_tc': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_tc.raw'
            },'mtc_dt': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_dt.raw'
            },'mhc_tc': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_tc.raw'
            },
            'mtc_hc': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_hc.raw'
            },
        }
        text = DatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = Dataset(DatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = Dataset(DatesetReader.__read_data__(fname[dataset]['test'], tokenizer))


class BertDatasetReader(DatesetReader):
    def __init__(self, dataset='dt_hc',tokenizer = None):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'dt_hc': {
                'train': './raw_data/dt.raw',
                'test': './raw_data/hc.raw'
            },
            'hc_dt': {
                'train': './raw_data/hc.raw',
                'test': './raw_data/dt.raw'
            },
            'fm_la': {
                'train': './raw_data/fm.raw',
                'test': './raw_data/la.raw'
            },
            'la_fm': {
                'train': './raw_data/la.raw',
                'test': './raw_data/fm.raw'
            },
            'dt_tp': {
                'train': './raw_data/dt.raw',
                'test': './raw_data/tp.raw'
            },
            'tp_dt': {
                'train': './raw_data/tp.raw',
                'test': './raw_data/dt.raw'
            },
            'hc_tp': {
                'train': './raw_data/hc.raw',
                'test': './raw_data/tp.raw'
            },
            'tp_hc': {
                'train': './raw_data/tp.raw',
                'test': './raw_data/hc.raw'
            },'mbs_dt': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_dt.raw'
            },
            'mdt_bs': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_bs.raw'
            },
            'mbs_hc': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_hc.raw'
            },
            'mhc_bs': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_bs.raw'
            },'mbs_tc': {
                'train': './raw_data/mul_bs.raw',
                'test': './raw_data/mul_tc.raw'
            },'mtc_bs': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_bs.raw'
            },'mdt_hc': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_hc.raw'
            },'mhc_dt': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_dt.raw'
            },'mdt_tc': {
                'train': './raw_data/mul_dt.raw',
                'test': './raw_data/mul_tc.raw'
            },'mtc_dt': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_dt.raw'
            },'mhc_tc': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_tc.raw'
            },
            'm': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_hc.raw'
            },
            'mhc_tc': {
                'train': './raw_data/mul_hc.raw',
                'test': './raw_data/mul_tc.raw'
            },
            'mtc_hc': {
                'train': './raw_data/mul_tc.raw',
                'test': './raw_data/mul_hc.raw'
            },

        }
        self.train_data = Dataset(BertDatasetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = Dataset(BertDatasetReader.__read_data__(fname[dataset]['test'], tokenizer))

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        # fin = open(fname+'.graph.inver.stance', 'rb')
        # idx2gragh_inver = pickle.load(fin)
        fin.close()
        fin = open(fname + '.graph.stance.3way11t', 'rb')
        idx2gragh = pickle.load(fin)
        fin.close()
        fin = open(fname + '.graph.cross.stance.3way11t', 'rb')
        idx2gragh_cross = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            stance = lines[i + 2].strip()

            text_indices = tokenizer.encode(text)
            # target_indices = tokenizer.text_to_sequence(target)
            attention_mask = [1] *len(text_indices)
            stance = int(stance) + 1
            in_graph = idx2gragh[i]
            cross_graph = idx2gragh_cross[i]
            assert in_graph.shape[0]==len(text_indices)==len(attention_mask)==cross_graph.shape[0],"length error"
            data = {
                'text': text,
                'target': target,
                'text_indices': text_indices,
                'attention_mask': attention_mask,
                'stance': stance,
                'in_graph': in_graph,
                'cross_graph': cross_graph,
            }
            all_data.append(data)
        return all_data
    


