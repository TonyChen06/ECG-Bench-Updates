#!/bin/bash

models=("merl")

# ### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa-mimic-iv-ecg-250-1250 \
    --model=$model \
    --device=cuda:4 \
    --dis \
    --gpus=6,7 \
    --train=first \
    --batch_size=256 \
    --ref_global_bs=1024 \
    --seg_len=1250 \
    --lr=8e-5 \
    --weight_decay=1e-4 \
    --epochs=50 \
    --log \
    --instance_normalize
done


