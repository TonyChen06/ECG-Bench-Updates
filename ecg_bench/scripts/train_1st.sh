#!/bin/bash

models=("clip")

# ### MULTI GPU
for model in "${models[@]}"; do
    python main.py \
    --data=ecg-qa-mimic-iv-ecg-250-1250 \
    --model=$model \
    --device=cuda:4 \
    --dis \
    --gpus=4,5,6,7 \
    --train=first \
    --batch_size=512 \
    --ref_global_bs=512 \
    --seg_len=1250 \
    --lr=8e-5 \
    --weight_decay=1e-4 \
    --epochs=100 \
    --image \
    --instance_normalize
done


