#!/bin/bash
#########################################################################
# File Name: run.sh
# Author: BinLiang
# mail: 18b951033@stu.hit.edu.cn 
# Created Time: Tue 26 May 2020 02:44:41 PM CST
#########################################################################

CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name icgcn --dataset dt_hc --save True --learning_rate 1e-3 --seed 522 --batch_size 16 --hidden_dim 300
