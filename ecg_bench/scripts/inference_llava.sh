#!/bin/bash

checkpoint=(
"merl_llama-3.2-1b-instruct_adam_160_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_True"
)
# --encoder_checkpoint=./runs/ecg-qa-mimic-iv-ecg-250-1250/0/encoderfree_adam_1024_256_100_1024_8e-05_0.9_0.99_1e-08_500_0.0001_True_None_None_None_None_False \
for c in "${checkpoint[@]}"; do
    # Extract the model prefix (clip_, merl_, etc.) from the checkpoint name
    model_prefix=$(echo "$c" | cut -d'_' -f1)
    
    python main.py \
    --data=ecg-qa-ptbxl-250-1250 \
    --model=merl_llama-3.2-1b-instruct \
    --device=cuda:6 \
    --seg_len=1250 \
    --peft \
    --inference=second \
    --encoder_checkpoint=./runs/ecg-qa-mimic-iv-ecg-250-1250/0/merl_adam_1024_256_100_1024_8e-05_0.9_0.99_1e-08_500_0.0001_True_None_None_None_None_False \
    --checkpoint=./runs/ecg-qa-ptbxl-250-1250/0/$c \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=1 \
    --epochs=1 \
    --instance_normalize
done