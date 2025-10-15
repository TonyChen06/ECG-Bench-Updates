#!/usr/bin/env bash

# ------------------- CONFIGURABLE LISTS -------------------
encoders=("merl")
#--encoder_checkpoint=./runs/ecg-qa-mimic-iv-ecg-250-1250/0/stmem_adam_1024_512_100_1024_8e-05_0.9_0.99_1e-08_500_0.0001_True_None_None_None_None_False \
# llms=("llama-3.2-1b-instruct" "gemma-2-2b-it" "qwen2.5-1.5b-instruct")
llms=("llama-3.2-1b-instruct")
# datasets=("ecg-qa-ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "ecg-instruct-45k-250-1250" "ecg-instruct-pulse-250-1250" "pretrain-mimic-250-1250") # add more datasets here# ----------------------------------------------------------
# datasets=("ecg-qa-ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "pretrain-mimic-250-1250")
datasets=("ecg-qa-ptbxl-250-1250" "ecg-qa-mimic-iv-ecg-250-1250" "pretrain-mimic-250-1250")
# datasets=("ecg-instruct-45k-250-1250")

for data in "${datasets[@]}"; do
  for llm in "${llms[@]}"; do
    for encoder in "${encoders[@]}"; do
      python main.py \
        --data="$data" \
        --model="${encoder}_${llm}" \
        --device=cuda:5 \
        --dis \
        --gpus=6,7 \
        --ref_global_bs=160 \
        --train=second \
        --batch_size=2 \
        --seg_len=1250 \
        --epochs=1 \
        --peft \
        --encoder_checkpoint=./runs/ecg-qa-mimic-iv-ecg-250-1250/0/merl_adam_1024_256_100_1024_8e-05_0.9_0.99_1e-08_500_0.0001_True_None_None_None_None_False \
        --instance_normalize \
        --pad_to_max=1024 \
        --attn_implementation=flash_attention_2 \
        --system_prompt=./data/system_prompt_e2e.txt \
        --log
    done
  done
done
