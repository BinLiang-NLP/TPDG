# TPDG 
__TPDG__ __T__arget-adaptive __P__ragmatics __D__ependency __G__raph
- Code and preprocessed dataset for paper in WWW2021 titled "Target-adaptive Graph for Cross-target Stance Detection".
- Bin Liang, Yonghao Fu and Lin Gui et al.

# Requirements
- pytorch >= 0.4.0
- numpy >= 1.13.3
- sklearn
- python 3.6 / 3.7
- transformers

# Pretrained Models
Download glove.42B.300d.zip from [glove website](https://nlp.stanford.edu/projects/glove/) and unzip in project root path.

# Usage
> pip install -r requirements.txt
> python3 train.py --model_name senticgcn --dataset rest16 --save True --learning_rate 1e-3 --batch_size 16 --hidden_dim 300





