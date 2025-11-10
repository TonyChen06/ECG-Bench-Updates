#!/bin/bash

checkpoint=(
"llama-3.2-1b-instruct_adam_160_2_1_1024_0.0001_0.9_0.99_1e-08_500_0.01_True_None_None_None_None_False"
)

# Specify the patch size used during training
patch_size=1250
seg_len=1250

for c in "${checkpoint[@]}"; do
    # Extract the LLM name from checkpoint (first part before optimizer info)
    llm_name=$(echo "$c" | cut -d'_' -f1,2,3,4 | sed 's/_[0-9].*$//')

    python main.py \
    --data=ecg-qa-ptbxl-250-1250 \
    --model=${llm_name} \
    --device=cuda:7 \
    --seg_len=${seg_len} \
    --peft \
    --inference=encoder_free \
    --checkpoint=./runs/ecg-qa-ptbxl-250-1250/0/$c \
    --system_prompt=./data/system_prompt_e2e.txt \
    --batch_size=1 \
    --epochs=1 \
    --instance_normalize \
    --patch_size=${patch_size}
done
