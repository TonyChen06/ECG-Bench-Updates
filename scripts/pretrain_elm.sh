#!/bin/bash
# Pretrain LLM from scratch on smollm-corpus
#
# Prerequisites:
#   1. Download corpus first: ./scripts/download_pretrain_corpus.sh
#
# Usage:
#   ./scripts/pretrain_elm.sh

# 1. Distributed pretraining without ECG examples
CUDA_VISIBLE_DEVICES=1,5 torchrun --nproc_per_node=2 -m ecg_bench.pretrain_elm \
    --llm smollm-135m \
    --reset_weights \
    --pretrain_corpus ./ecg_bench/data/pretrain_corpus \
    --seq_len 2048 \
    --batch_size 4 \
    --epochs 1 \
    --lr 5e-4 \
    --warmup 1000 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --optimizer adamw \
    --attention_type flash_attention_2 \
    --wandb \
    --distributed

# 2. Distributed pretraining with ECG examples (uncomment to use)
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 -m ecg_bench.pretrain_elm \
#     --llm smollm-135m \
#     --reset_weights \
#     --pretrain_corpus ./ecg_bench/data/pretrain_corpus \
#     --ecg_data ecg-qa-ptbxl-250-1250 \
#     --ecg_num_examples 10000 \
#     --ecg_raw \
#     --seq_len 2048 \
#     --batch_size 4 \
#     --epochs 1 \
#     --lr 5e-4 \
#     --warmup 1000 \
#     --weight_decay 0.1 \
#     --grad_clip 1.0 \
#     --beta1 0.9 \
#     --beta2 0.95 \
#     --optimizer adamw \
#     --attention_type flash_attention_2 \
#     --wandb \
#     --distributed

echo "Pretraining complete!"
