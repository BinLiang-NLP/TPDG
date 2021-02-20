# Introduction
This repository was used in our paper:  
  
**Target-adaptive Graph for Cross-target Stance Detection**
<br>
Bin Liang, Yonghao Fu, Lin Gui<sup>\*</sup>, Min Yang, Jiachen Du, Yulan He, Ruifeng Xu<sup>\*</sup>. *Proceedings of WWW 2021*
  
Please cite our paper and kindly give a star for this repository if you use this code.

## Requirements
- pytorch >= 0.4.0
- numpy >= 1.13.3
- sklearn
- python 3.6 / 3.7
- transformers

## Pretrained Models
Download glove.42B.300d.zip from [glove website](https://nlp.stanford.edu/projects/glove/) and unzip in project root path.

## Usage
* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* install requirements
```bash
pip install -r requirements.txt
```

## Training
* Train with command, optional arguments could be found in [train.py](/train.py)
```bash
python3 train.py 
  --model_name senticgcn 
  --dataset rest16 
  --save True 
  --learning_rate 1e-3 
  --batch_size 16 
  --hidden_dim 300
```





