CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m ecg_bench.evaluate_elm \
--ecg_signal \
--llm=qwen3-4b-instruct \
--data=ecg-qa-ptbxl-250-1250 \
--device=cuda:5 \
--encoder=projection \
--peft \
--attention_type=flash_attention_2 \
--system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
--dev \
--elm_ckpt=./ecg_bench/runs/training/elm/5/checkpoints/epoch_best.pt

# datasets=(
#   ecg-qa-ptbxl-250-1250
#   #ecg-qa-mimic-iv-ecg-250-1250
#   #ecg-instruct-45k-250-1250
#   #pretrain-mimic-250-1250
#   # ecg-bench-pulse-250-1250
#   # ecg-instruct-pulse-250-1250
# )