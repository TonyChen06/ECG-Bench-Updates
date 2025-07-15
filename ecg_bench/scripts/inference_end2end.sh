#!/bin/bash

data=("ecg-qa_ptbxl_mapped_500")

for d in "${data[@]}"; do
    # Extract the model prefix (clip_, merl_, etc.) from the checkpoint name
    model_prefix=$(echo "$c" | cut -d'_' -f1)
    
    python main.py \
    --data=$d \
    --model=llama-3.2-1b-instruct \
    --device=cuda:7 \
    --seg_len=500 \
    --peft \
    --inference=end2end \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=1 \
    --pad_to_max=1024 \
    --attn_implementation=flash_attention_2 \
    --ecg_tokenizer=./data/tokenizer_3500_300000_True_None.pkl \
    --checkpoint=./runs/ecg-qa_ptbxl_mapped_500/0/llama-3.2-1b-instruct_4_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False
done