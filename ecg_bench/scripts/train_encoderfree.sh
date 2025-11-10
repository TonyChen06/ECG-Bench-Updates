#!/usr/bin/env bash

# ------------------- CONFIGURABLE LISTS -------------------
llms=("llama-3.2-1b-instruct")
# llms=("llama-3.2-1b-instruct" "gemma-2-2b-it" "qwen2.5-1.5b-instruct")
patch_sizes=(5)  # Different patch sizes to ablate
# datasets=("ecg-qa-ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "ecg-instruct-45k-250-1250" "ecg-instruct-pulse-250-1250" "pretrain-mimic-250-1250")
datasets=("ecg-qa-ptbxl-250-1250")
seg_len=1250
# ----------------------------------------------------------

for data in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for patch_size in "${patch_sizes[@]}"; do
      python main.py \
        --data="$data" \
        --model="${llm}" \
        --device=cuda:2 \
        --dis \
        --gpus=6,7 \
        --ref_global_bs=160 \
        --train=encoder_free \
        --batch_size=2 \
        --seg_len=${seg_len} \
        --epochs=1 \
        --peft \
        --instance_normalize \
        --pad_to_max=1024 \
        --patch_size=${patch_size} \
        --attn_implementation=flash_attention_2 \
        --log \
        --system_prompt=./data/system_prompt_e2e.txt
    done
  done
done
