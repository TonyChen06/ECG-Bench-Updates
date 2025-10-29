
datasets=(
  # ecg-qa-mimic-iv-ecg-250-1250
  ecg-qa-ptbxl-250-1250
  # pretrain-mimic-250-1250
  #ecg-instruct-45k-250-1250
  # ecg-bench-pulse-250-1250
  # ecg-instruct-pulse-250-1250
)
for data in "${datasets[@]}"
do
  CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 \
    --master_port=10067 \
    -m ecg_bench.train_elm \
    --distributed \
    --ecg_signal \
    --llm=llama-3.2-1b-instruct \
    --data="$data" \
    --encoder_ckpt=ecg_bench/runs/training/encoder/2/checkpoints/step_epoch_49_step_1560.pt \
    --peft \
    --encoder=merl \
    --batch_size=2 \
    --update_encoder \
    --ref_global_bs=160 \
    --attention_type=flash_attention_2 \
    --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
    --wandb
    echo "Finished training on $data"
    echo "-----------------------------------"
done

# python -m ecg_bench.train_elm \
# --ecg_signal \
# --batch_size=2 \
# --llm=llama-3.2-1b-instruct \
# --encoder=siglip-base-patch16-224 \
# --data=ecg-instruct-45k-250-1250 \
# --device=cuda:1 \
# --peft \
# --attention_type=flash_attention_2 \
# --system_prompt=./ecg_bench/configs/system_prompt/system_prompt.txt \
# --dev