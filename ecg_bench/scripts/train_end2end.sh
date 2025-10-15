#!/usr/bin/env bash

# ------------------- CONFIGURABLE LISTS -------------------
llms=("llama-3.2-1b-instruct")
# llms=("gemma-2-2b-it")
datasets=("ecg-instruct-45k-250-1250") # add more datasets here# ----------------------------------------------------------

for data in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    python main.py \
    --data="$data" \
    --model="${llm}" \
    --device=cuda:4 \
    --dis \
    --gpus=6,7 \
    --ref_global_bs=160 \
    --ecg_tokenizer=./data/tokenizer_5000_300000_True_None.pkl \
    --train=end2end \
    --batch_size=2 \
    --seg_len=1250 \
    --epochs=1 \
    --peft \
    --instance_normalize \
    --pad_to_max=1024 \
    --attn_implementation=flash_attention_2 \
    --system_prompt=./data/system_prompt_e2e.txt \
    --log
done
done

