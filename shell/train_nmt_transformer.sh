#!/usr/bin/env bash
cd ../
pwd

export PYTHONPATH=$PYTHONPATH:./src
export CUDA_VISIBLE_DEVICES=0

nohup python3.6 src/nmt/transformer/train.py \
    --batch_size=128 \
    --num_train_samples=196884 > log_train.txt 2>&1 &
