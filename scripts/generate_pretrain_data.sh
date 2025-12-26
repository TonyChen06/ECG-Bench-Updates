#!/bin/bash
# Generate pretraining data for ECG token understanding

python -m ecg_bench.pretraining.generate_dataset \
    --output_dir ./ecg_bench/data/pretraining \
    --num_samples 100000 \
    --seed 42 \
    --task1_num_values 50 \
    --task2_num_tokens 20 \
    --task3_duration 2.0 \
    --task4_duration 1.0 \
    --task5_duration 2.0 \
    --task5_chunk_size 100 \
    --task6_duration 2.0 \
    --task6_chunk_size 100
