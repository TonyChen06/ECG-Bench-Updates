CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
-m ecg_bench.train_encoder \
--ecg_signal \
--encoder=merl \
--data=ecg-qa-mimic-iv-ecg-250-1250 \
--batch_size=512 \
--ref_global_bs=2048 \
--optimizer=adamw \
--lr=8e-5 \
--weight_decay=1e-4 \
--distributed \
--wandb \
--epochs=200


# python -m ecg_bench.train_encoder \
# --ecg_signal \
# --encoder=merl \
# --data=ecg-qa-mimic-iv-ecg-250-1250 \
# --batch_size=16 \
# --optimizer=adamw \
# --lr=2e-4 \
# --weight_decay=1e-5 \
# --device=cuda:0 \
# --dev