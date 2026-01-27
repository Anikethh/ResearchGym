#!/usr/bin/env bash

train_tar="${1:-ILSVRC2012_img_train.tar}"
val_tar="${2:-ILSVRC2012_img_val.tar}"

mkdir -p train
mkdir -p val

echo "Extracting training set ... (might take a while)"
tar -xf "${train_tar}" -C train

echo "Extracting training categories ..."
cd train
find . -name "*.tar" | xargs -n1 -P8 -I {} bash -c 'mkdir -p "${1%.tar}"; tar -xf "${1}" -C "${1%.tar}"; rm -f "${1}"' -- {}
cd ..

echo "Extracting validation set ..."
tar -xf "${val_tar}" -C val

echo "Restructuring validation ..."
cd val

# Python like zip from two streams
function zip34() { while read word3 <&3; do read word4 <&4 ; echo $word3 $word4 ; done }

wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt
find . -name "*.JPEG" | sort > images.txt
zip34 3<images.txt 4<imagenet_2012_validation_synset_labels.txt | xargs -n2 -P8 bash -c 'mkdir -p $2; mv $1 $2' argv0

rm *.txt
cd ..

echo "train:" $(find train -name "*.JPEG" | wc -l) "images"
echo "val:" $(find val -name "*.JPEG" | wc -l) "images"