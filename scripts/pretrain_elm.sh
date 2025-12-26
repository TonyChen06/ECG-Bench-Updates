#!/bin/bash
# Pretrain LLM on ECG token understanding tasks

python -m ecg_bench.pretrain_elm \
    --llm qwen3-4b-instruct \
    --pretrain_data ./ecg_bench/data/pretraining \
    --peft \
    --attention_type flash_attention_2 \
    --device cuda:0 \
    --epochs 1 \
    --batch_size 4 \
    --lr 1e-4 \
    --llm_input_len 2048 \
    --warmup 5000 \
    --dev
