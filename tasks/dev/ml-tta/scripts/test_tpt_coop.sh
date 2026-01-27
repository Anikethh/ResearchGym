#!/bin/bash

data_root='../DATASETS'
coop_weight=pretrain_weights/coop/vit-b16-model.pth.tar-50
gpu=$1
perform_tpt=$2
testsets=$3
is_bind=$4
# arch=RN50
# arch=RN101
arch=ViT-B/16
# arch=ViT-B/32
bs=64

python ./tpt_classification.py --data ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --coop ${coop_weight} \
--tpt ${perform_tpt} --is_bind ${is_bind}