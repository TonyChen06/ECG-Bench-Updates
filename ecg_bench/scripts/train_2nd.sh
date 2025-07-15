#!/bin/bash

# models=("stmem" "merl" "mlae" "mtae" "siglip" "clip" "vit")
models=("merl")
# llms=("gemma-2-2b-it" "llama-3.2-1b-instruct" "qwen2.5-1.5b-instruct")
llms=("llama-3.2-1b-instruct")

# ### MULTI GPU
for llm in "${llms[@]}"; do
    for model in "${models[@]}"; do
        python main.py \
        --data=ecg-qa_ptbxl_mapped_500 \
        --model=${model}_${llm} \
        --device=cuda:0 \
        --dis \
        --gpus=2,3,6,7 \
        --train=second \
        --batch_size=4 \
        --seg_len=500 \
        --epochs=1 \
        --peft \
        --pad_to_max=1024 \
        --attn_implementation=flash_attention_2 \
        --system_prompt=./data/system_prompt_e2e.txt \
        --encoder_checkpoint=./runs/ecg-qa_ptbxl_mapped_500/0/merl_bpe_256_50_1024_0.0001_0.9_0.99_1e-08_3000_0.0001_True_None_None_None_None_False
    done
done


# models=("vit" "clip" "siglip" )

# for model in "${models[@]}"; do
#     python main.py \
#     --data=ecg-qa_mimic-iv-ecg_mapped_1250 \
#     --model=$model \
#     --device=cuda:6 \
#     --train=first \
#     --batch_size=8 \
#     --seg_len=1250 \
#     --epochs=2 \
#     --instance_normalize \
#     --attn_implementation=flash_attention_2 \
#     --image \
#     --log
# done