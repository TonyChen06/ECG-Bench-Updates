# CUDA_VISIBLE_DEVICES=4,5 \
# torchrun --standalone --nproc_per_node=2 \
# -m ecg_bench.train_encoder \
# --ecg_signal \
# --encoder=mtae \
# --data=ecg-qa-mimic-iv-ecg-250-1250 \
# --batch_size=16 \
# --optimizer=adamw \
# --lr=2e-4 \
# --weight_decay=1e-5 \
# --distributed \
# --dev


CUDA_VISIBLE_DEVICES=6,7 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
torchrun --standalone --nproc_per_node=2 --master_port=10065 \
-m ecg_bench.train_encoder \
--ecg_signal \
--encoder=st_mem \
--data=ecg-qa-mimic-iv-ecg-250-1250 \
--batch_size=256 \
--optimizer=adamw \
--lr=8e-5 \
--ref_global_bs=1024 \
--weight_decay=1e-4 \
--distributed \
--wandb \
--epochs=50