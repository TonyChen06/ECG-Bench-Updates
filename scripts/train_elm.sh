#!/bin/bash
# Training script for ECG-LLM
#
# Two modes supported:
# 1. --ecg_signal with --encoder: Uses encoder (projection, etc.) to process ECG signals
# 2. --ecg_raw: Uses pretrained ECG token embeddings (run pretrain_elm.sh first!)

# For ecg_raw mode with pretrained checkpoint:
# Uncomment and set ELM_CKPT to your pretrained checkpoint path

#datasets=(
  #ecg-qa-ptbxl-250-1250
  #ecg-qa-mimic-iv-ecg-250-1250
  #ecg-instruct-45k-250-1250
  #pretrain-mimic-250-1250
  # ecg-bench-pulse-250-1250
  # ecg-instruct-pulse-250-1250
#)
# for data in "${datasets[@]}"
# do
#   CUDA_VISIBLE_DEVICES=5,6 \
#   CUBLAS_WORKSPACE_CONFIG=:4096:8 \
#   torchrun --standalone --nproc_per_node=2 --master_port=10067 \
#   -m ecg_bench.train_elm \
#     --ecg_signal \
#     --llm=qwen3-4b-instruct \
#     --data="$data" \
#     --distributed \
#     --peft \
#     --encoder=projection \
#     --batch_size=2 \
#     --attention_type=flash_attention_2 \
#     --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
#     --wandb
#     echo "Finished training on $data"
#     echo "-----------------------------------"
# done

# ECG Raw mode with pretrained checkpoint


datasets=(
  ecg-qa-ptbxl-250-1250
  # ecg-qa-mimic-iv-ecg-250-1250
  # ecg-instruct-45k-250-1250
  # pretrain-mimic-250-1250
  # ecg-bench-pulse-250-1250
  # ecg-instruct-pulse-250-1250
)

ELM_CKPT="./ecg_bench/runs/pretraining/elm/0/checkpoints/epoch_best.pt"

for data in "${datasets[@]}"
do
  CUDA_VISIBLE_DEVICES=5,6 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  torchrun --standalone --nproc_per_node=2 --master_port=10067 \
  -m ecg_bench.train_elm \
    --ecg_raw \
    --llm=qwen3-4b-instruct \
    --data="$data" \
    --distributed \
    --peft \
    --elm_ckpt="${ELM_CKPT}" \
    --batch_size=1 \
    --llm_input_len 2560 \
    --attention_type=flash_attention_2 \
    --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
    --wandb
    echo "Finished training on $data"
    echo "-----------------------------------"
done
