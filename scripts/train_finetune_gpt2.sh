#!/bin/sh

python train.py \
    --data_path datasets/preprocessed/clip_caption.pkl \
    --gpt2_path pretrain_models/gpt2 \
    --bert_path pretrain_models/bert \
    --output_path output/trained/finetune \
    --lr 2e-5 \
    --epochs 40 \
    --prefix_len 10 \
    --constant_len 10 \
    --clip_size 512 \
    --bs_train 40 \
    --dev_size 1000 \
    --bs_eval 128 \
    --max_len 100 \
    --warmup_steps 5000 \
    --save_step 1000 \
    --eval_step 500 \
    --finetune_gpt2 \
    --mapping_type mlp \
    --do_train
