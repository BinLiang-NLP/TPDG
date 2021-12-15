# Introduction
This repository was used in our paper:  

[**Target-adaptive Graph for Cross-target Stance Detection**](http://wrap.warwick.ac.uk/149336/1/WRAP-Target-adaptive-graph-cross-target-stance-detection-2021.pdf)
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
pip3 install spacy
```
and
```bash
python3 -m spacy download en
```
* install requirements
```bash
pip3 install -r requirements.txt
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


## Citation

The BibTex of the citation is as follow:

```bibtex
@inproceedings{10.1145/3442381.3449790,
author = {Liang, Bin and Fu, Yonghao and Gui, Lin and Yang, Min and Du, Jiachen and He, Yulan and Xu, Ruifeng},
title = {Target-Adaptive Graph for Cross-Target Stance Detection},
year = {2021},
isbn = {9781450383127},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3442381.3449790},
doi = {10.1145/3442381.3449790},
abstract = { Target plays an essential role in stance detection of an opinionated review/claim, since the stance expressed in the text often depends on the target. In practice, we need to deal with targets unseen in the annotated training data. As such, detecting stance for an unknown or unseen target is an important research problem. This paper presents a novel approach that automatically identifies and adapts the target-dependent and target-independent roles that a word plays with respect to a specific target in stance expressions, so as to achieve cross-target stance detection. More concretely, we explore a novel solution of constructing heterogeneous target-adaptive pragmatics dependency graphs (TPDG) for each sentence towards a given target. An in-target graph is constructed to produce inherent pragmatics dependencies of words for a distinct target. In addition, another cross-target graph is constructed to develop the versatility of words across all targets for boosting the learning of dominant word-level stance expressions available to an unknown target. A novel graph-aware model with interactive Graphical Convolutional Network (GCN) blocks is developed to derive the target-adaptive graph representation of the context for stance detection. The experimental results on a number of benchmark datasets show that our proposed model outperforms state-of-the-art methods in cross-target stance detection.},
booktitle = {Proceedings of the Web Conference 2021},
pages = {3453–3464},
numpages = {12},
keywords = {graph networks, cross-target stance detection, opinion mining},
location = {Ljubljana, Slovenia},
series = {WWW '21}
}
```


