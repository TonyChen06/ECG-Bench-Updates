#!/bin/bash
# Download 1% of smollm-corpus for pretraining
# Includes Cosmopedia v2 (~11%) and FineWeb-Edu (~89%)

python -m ecg_bench.download_pretrain_corpus \
    --sample_ratio 0.01 \
    --output_dir ./ecg_bench/data/pretrain_corpus \
    --seed 42 \
    --num_proc 4

echo "Download complete!"
echo "Corpus saved to: ./ecg_bench/data/pretrain_corpus/"
