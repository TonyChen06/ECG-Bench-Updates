#!/bin/bash
# Pretrain LLM on ECG token understanding tasks
#
# Two modes:
# 1. Single GPU: Use python directly
# 2. Multi-GPU: Use torchrun with --distributed flag
#
# Stage-based Training:
# - Use --stage N to train only on stage N (1-9)
# - Use --resume_from /path/to/checkpoint to continue from a previous stage
# - Stages are detachable: train stage 1, save, then train stage 2, etc.
#
# Scheduled Sampling (horizon-based curriculum):
# - Tasks 5 and 6 use scheduled sampling to learn long-range ECG generation
# - Horizon increases throughout training: 1 -> 2 -> 3 -> ... -> N tokens
# - ss_horizon_start: Starting horizon (default 1 = next token prediction)
# - ss_horizon_end: Ending horizon (default 10 = predict 10 tokens autoregressively)

# ============================================================================
# Single GPU Training (all stages)
# ============================================================================
# python -m ecg_bench.pretrain_elm \
#     --llm qwen3-4b-instruct \
#     --pretrain_data ./ecg_bench/data/pretraining \
#     --peft \
#     --attention_type flash_attention_2 \
#     --device cuda:0 \
#     --epochs 1 \
#     --batch_size 2 \
#     --wandb \
#     --lr 5e-5 \
#     --llm_input_len 2048 \
#     --warmup 5000

# ============================================================================
# Multi-GPU Distributed Training
# ============================================================================


CUDA_VISIBLE_DEVICES=0,5,6,7 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node=4 --master_port=10067 \
-m ecg_bench.pretrain_elm \
    --llm qwen3-4b-instruct \
    --pretrain_data ./ecg_bench/data/pretrainingAhri \
    --distributed \
    --peft \
    --attention_type flash_attention_2 \
    --epochs 1 \
    --batch_size 1 \
    --wandb \
    --max_grad_norm 1.0 \
    --lr 2e-5 \
    --llm_input_len 2048 \
    --warmup 3000 \
    --embed_skip_warmup \
    --stage 3 \
    --resume_from ./ecg_bench/runs/pretraining/elm/stage_2/0/checkpoints/epoch_best.pt
