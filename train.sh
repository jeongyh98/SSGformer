#!/usr/bin/env bash
CONFIG=$1
PORT_=$2
GPU=$3

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=1 \
    --master_port=$PORT_ basicsr/train.py -opt $CONFIG --launcher pytorch


#  ./train.sh ./Allweather/Options/SSGformer.yml 1111 3
