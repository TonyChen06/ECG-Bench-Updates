#!/bin/bash

data=("ecg-qa_ptbxl_mapped_500")


for d in "${data[@]}"; do
    python main.py \
    --data=ecg-qa_ptbxl_mapped_500 \
    --model=llama-3.2-1b-instruct \
    --gpus=2,3,6,7 \
    --dis \
    --ecg_tokenizer=./data/tokenizer_3500_300000_True_None.pkl \
    --seg_len=500 \
    --peft \
    --train=end2end \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=4 \
    --pad_to_max=1024 \
    --epochs=1 \
    --log \
    --attn_implementation=flash_attention_2
done
