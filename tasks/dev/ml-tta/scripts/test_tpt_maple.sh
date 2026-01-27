#!/bin/bash

data_root='../DATASETS'
maple_weight=pretrain_weights/maple/vit-b16_maple_seed1.pth
gpu=$1
perform_tpt=$2
testsets=$3
is_bind=$4
arch=ViT-B/16
bs=64

python ./tpt_classification.py --data ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --maple ${maple_weight} \
--tpt ${perform_tpt} --is_bind ${is_bind}