#!/bin/bash
# Generate pretraining data for ECG token understanding
#
# Stage order (curriculum, stages 1-9):
#   1: values_to_tokens (ECG embedding warmup)   - 30k
#   2: Simple comparison (magnitude/value)       - 40k  (8%)
#   3: Arithmetic (sequence_next/halfway)        - 40k  (8%)
#   4: Wave classification + properties          - 30k  (6%)
#   5: Wave transformation                       - 50k  (10%)
#   6: Wave generation                           - 60k  (12%)
#   7: ECG generation (real MIMIC data)          - 100k (20%)
#   8: Long-range wave prediction                - 80k  (16%)
#   9: Long-range ECG prediction                 - 100k (20%)
#   Total: ~550k main samples + ~5% review
#
# Curriculum mode (default): Stages are generated in order (1→9)
# with 5% review of earlier stages in each phase (using same distribution).
#
# Use --no_curriculum for random task sampling instead.

# Stage distribution as JSON (stages 1-9)
TASK_DIST='{"1": 30000, "2": 30000, "3": 100000, "4": 100000, "5": 70000, "6": 80000, "7": 100000, "8": 80000, "9": 100000}'
#TASK_DIST='{"1": 50, "2": 50, "3": 50, "4": 50, "5": 50, "6": 50, "7": 50, "8": 50, "9": 50}'
python -m ecg_bench.pretraining.generate_dataset \
    --output_dir ./ecg_bench/data/pretrainingAhri \
    --seed 42 \
    --curriculum \
    --review_ratio 0.1 \
    --task_distribution "$TASK_DIST" \
    --stage1_num_values 50 \
    --stage1_num_tokens 20 \
    --stage4_duration 2.0 \
    --stage5_duration 1.0 \
    --stage6_duration 2.0 \
    --stage6_chunk_size 100 \
    --stage7_chunk_size 100 \
    --stage8_context_tokens 200 \
    --stage9_context_tokens 200 \
    --mimic_dataset ecg-qa-mimic-iv-ecg-250-1250 \
    --mimic_fold 1
