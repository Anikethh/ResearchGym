#!/bin/bash

data_root='../DATASETS'
gpu=$1
perform_tpt=$2
testsets=$3
is_bind=$4
# arch=RN50
# arch=RN101
arch=ViT-B/16
# arch=ViT-B/32
bs=64
ctx_init=a_photo_of_a

python ./tpt_classification.py --data ${data_root} --test_sets ${testsets} \
--a ${arch} --b ${bs} --gpu ${gpu} --ctx_init ${ctx_init} \
--tpt ${perform_tpt} --is_bind ${is_bind}