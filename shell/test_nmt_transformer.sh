#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:./src
export CUDA_VISIBLE_DEVICES=1

# 执行eval
python3.6 ./src/nmt/transformer/test.py \
    ./out/nmt/transformer/y_predict.txt \
    ./out/nmt/transformer/y_label.txt \
    1 \
    9990

# 计算blue 值
perl ./src/nmt/multi-blue.perl \
    ./out/nmt/transformer/y_label.txt < ./out/nmt/transformer/y_predict.txt
