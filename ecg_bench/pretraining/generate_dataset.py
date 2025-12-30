"""
Generate pretraining dataset for ECG token understanding.

This script generates a dataset of pretraining tasks and saves them to disk.
The dataset can be loaded by the pretraining dataloader.

Usage:
    python -m ecg_bench.pretraining.generate_dataset \
        --output_dir ./ecg_bench/data/pretraining \
        --num_samples 100000 \
        --seed 42
"""

import argparse
import json
import os
import random
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm
import numpy as np

from ecg_bench.pretraining.task_generators import (
    generate_task_0, generate_task_1, generate_task_2,
    generate_task_3, generate_task_4, generate_task_5,
    generate_task_6, generate_task_7, generate_task_8,
    generate_task_1_comparison, generate_task_2_arithmetic,
    task_to_conversation, Task,
    init_ecg_loader,
)


# =============================================================================
# Stage Mapping (9 stages, numbered 1-9)
# =============================================================================
# Stage 1: values_to_tokens (outputs ECG tokens for embedding training)
# Stage 2: magnitude_comparison + value_comparison (simple comparison)
# Stage 3: sequence_next + halfway_token (arithmetic)
# Stage 4-9: wave_classification, transformation, generation, ECG tasks
#
# This restructuring allows:
# - Stage 1 to warm up ECG token embeddings (outputs ECG tokens -> trains lm_head)
# - Stage 2 to teach simple ordering/comparison (no arithmetic)
# - Stage 3 to teach arithmetic on token indices


def task_hash(task: Task) -> str:
    """
    Generate a hash for a task based on its conversation content.
    Used to detect and skip exact duplicates.
    """
    # Concatenate all turn contents
    content = "".join(turn.content for turn in task.turns)
    return hashlib.md5(content.encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate pretraining dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_samples", type=int, default=100000, help="Total number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--task_weights", type=str, default=None,
                       help="JSON string of task weights, e.g., '{\"0\": 2, \"1\": 1}'")

    # Stage-specific parameters
    parser.add_argument("--stage1_num_values", type=int, default=50, help="Number of values for Stage 1 values_to_tokens")
    parser.add_argument("--stage1_num_tokens", type=int, default=20, help="Number of tokens for Stage 1 tokens_to_values")
    parser.add_argument("--stage4_duration", type=float, default=2.0, help="Duration for Stage 4 wave classification (seconds)")
    parser.add_argument("--stage5_duration", type=float, default=1.0, help="Duration for Stage 5 wave transformation (seconds)")
    parser.add_argument("--stage6_duration", type=float, default=2.0, help="Duration for Stage 6 wave generation (seconds)")
    parser.add_argument("--stage6_chunk_size", type=int, default=100, help="Chunk size for Stage 6 wave generation")
    parser.add_argument("--stage7_chunk_size", type=int, default=100, help="Chunk size for Stage 7 ECG generation")
    parser.add_argument("--stage8_context_tokens", type=int, default=200, help="Context tokens for Stage 8 long-range wave")
    parser.add_argument("--stage9_context_tokens", type=int, default=200, help="Context tokens for Stage 9 long-range ECG")

    # MIMIC dataset parameters for Stage 7
    parser.add_argument("--mimic_dataset", type=str, default="ecg-qa-mimic-iv-ecg-250-1250",
                       help="MIMIC dataset to use for Stage 7 ECG data")
    parser.add_argument("--mimic_fold", type=str, default="1", help="Fold to use from MIMIC dataset")

    # Curriculum learning parameters
    parser.add_argument("--curriculum", action="store_true", default=True,
                       help="Use curriculum learning (tasks in order)")
    parser.add_argument("--no_curriculum", action="store_true",
                       help="Disable curriculum learning (random task sampling)")
    parser.add_argument("--review_ratio", type=float, default=0.1,
                       help="Fraction of earlier tasks to include in each phase (default: 0.1)")
    parser.add_argument("--task_distribution", type=str, default=None,
                       help="JSON string of task sample counts, e.g., '{\"0\": 20000, \"1\": 20000, ...}'. "
                            "If provided, --num_samples is ignored for curriculum mode.")

    # Single-stage generation (for detachable stages)
    parser.add_argument("--stage", type=int, default=None,
                       help="Generate data for only a single stage (1-9). "
                            "Saves to output_dir/stage_N/. If not set, generates all stages together.")

    return parser.parse_args()


def generate_task_by_stage(
    stage: int,
    stage1_num_values: int = 50,
    stage1_num_tokens: int = 20,
    stage4_duration: float = 2.0,
    stage5_duration: float = 1.0,
    stage6_duration: float = 2.0,
    stage6_chunk_size: int = 100,
    stage7_chunk_size: int = 100,
    stage8_context_tokens: int = 200,
    stage9_context_tokens: int = 200,
) -> Task:
    """
    Generate a task for the specified stage.

    Stage mapping (1-9):
    - Stage 1: values_to_tokens (outputs ECG tokens for embedding training)
    - Stage 2: magnitude_comparison OR value_comparison
    - Stage 3: sequence_next OR halfway_token
    - Stages 4-9: wave_classification, transformation, generation, ECG tasks
    """
    if stage == 1:
        # Stage 1: values_to_tokens or tokens_to_values
        if random.random() < 0.5:
            return generate_task_0(num_values=stage1_num_values)
        else:
            return generate_task_1(num_tokens=stage1_num_tokens)
    elif stage == 2:
        # magnitude_comparison or value_comparison
        return generate_task_1_comparison()
    elif stage == 3:
        # sequence_next or halfway_token
        return generate_task_2_arithmetic()
    elif stage == 4:
        return generate_task_3(duration=stage4_duration)
    elif stage == 5:
        return generate_task_4(duration=stage5_duration)
    elif stage == 6:
        return generate_task_5(duration=stage6_duration, chunk_size=stage6_chunk_size)
    elif stage == 7:
        return generate_task_6(chunk_size=stage7_chunk_size)
    elif stage == 8:
        return generate_task_7(context_tokens=stage8_context_tokens)
    elif stage == 9:
        return generate_task_8(context_tokens=stage9_context_tokens)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def sample_task_type(task_weights: Dict[int, float]) -> int:
    """Sample a task type based on weights."""
    task_types = list(task_weights.keys())
    weights = [task_weights[t] for t in task_types]
    total = sum(weights)
    weights = [w / total for w in weights]
    return np.random.choice(task_types, p=weights)


def generate_dataset(
    num_samples: int,
    task_weights: Optional[Dict[int, float]] = None,
    max_retries: int = 100,
    curriculum: bool = True,
    review_ratio: float = 0.1,
    task_distribution: Optional[Dict[int, int]] = None,
    **task_params,
) -> List[Dict]:
    """
    Generate the full dataset with curriculum ordering.

    Args:
        num_samples: Target number of unique samples
        task_weights: Weights for each task type (ignored if curriculum=True)
        max_retries: Max attempts to generate a unique task before giving up
        curriculum: If True, generate in curriculum order (Task 0 → 8)
        review_ratio: Fraction of earlier tasks to sprinkle in each phase
        task_distribution: Dict mapping task_id to sample count (curriculum mode only)
        **task_params: Parameters passed to task generators
    """
    if curriculum:
        return generate_curriculum_dataset(
            num_samples, max_retries, review_ratio,
            task_distribution=task_distribution, **task_params
        )

    # Original random sampling logic
    if task_weights is None:
        task_weights = {
            0: 1.0,  # Values to tokens (explicit mapping)
            1: 1.0,  # Tokens to values (reverse mapping)
            2: 2.0,  # Token comparison (implicit understanding - more of these)
            3: 1.0,  # Wave classification
            4: 1.0,  # Wave transformation
            5: 1.0,  # Wave reconstruction
            6: 1.0,  # ECG reconstruction
            7: 1.0,  # Long-range wave prediction
            8: 1.0,  # Long-range ECG prediction
        }

    dataset = []
    seen_hashes: Set[str] = set()
    task_counts = {i: 0 for i in range(9)}
    duplicate_count = 0
    accepted_duplicates = 0

    pbar = tqdm(total=num_samples, desc="Generating tasks")

    while len(dataset) < num_samples:
        task_type = sample_task_type(task_weights)

        # Try to generate a unique task
        for attempt in range(max_retries):
            task = generate_task_by_type(task_type, **task_params)
            h = task_hash(task)

            if h not in seen_hashes:
                seen_hashes.add(h)
                conversation = task_to_conversation(task)

                sample = {
                    "task_type": task.task_type,
                    "task_id": task_type,
                    "text": conversation,
                    "metadata": task.metadata,
                }
                dataset.append(sample)
                task_counts[task_type] += 1
                pbar.update(1)
                break
            else:
                duplicate_count += 1
        else:
            # If we exhausted retries, just accept the duplicate
            accepted_duplicates += 1
            seen_hashes.add(h)
            conversation = task_to_conversation(task)
            sample = {
                "task_type": task.task_type,
                "task_id": task_type,
                "text": conversation,
                "metadata": task.metadata,
            }
            dataset.append(sample)
            task_counts[task_type] += 1
            pbar.update(1)

    pbar.close()

    print(f"\nDuplicates skipped: {duplicate_count}")
    print(f"Duplicates accepted (retries exhausted): {accepted_duplicates}")
    print("\nTask distribution:")
    for task_id, count in task_counts.items():
        print(f"  Task {task_id}: {count} ({count / len(dataset) * 100:.1f}%)")

    return dataset


def generate_curriculum_dataset(
    num_samples: int,
    max_retries: int = 100,
    review_ratio: float = 0.1,
    task_distribution: Optional[Dict[int, int]] = None,
    **task_params,
) -> List[Dict]:
    """
    Generate dataset with curriculum ordering and custom task distribution.

    Each phase focuses on one stage with review of earlier stages mixed in.
    Stages are ordered: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
    Review samples follow the same distribution as main samples (normalized to earlier stages).

    Args:
        num_samples: Target number of unique samples (ignored if task_distribution provided)
        max_retries: Max attempts to generate a unique task
        review_ratio: Fraction of samples that are review (earlier stages)
        task_distribution: Dict mapping stage (1-9) to number of main samples for that stage.
                          If provided, num_samples is ignored.
        **task_params: Parameters passed to task generators
    """
    num_stages = 9

    # Default: equal distribution (stages 1-9)
    if task_distribution is None:
        samples_per_stage = num_samples // num_stages
        task_distribution = {i: samples_per_stage for i in range(1, num_stages + 1)}
        # Add remainder to last stage
        task_distribution[num_stages] += num_samples - (samples_per_stage * num_stages)

    total_main_samples = sum(task_distribution.values())

    dataset = []
    seen_hashes: Set[str] = set()
    stage_counts = {i: 0 for i in range(1, num_stages + 1)}
    duplicate_count = 0
    accepted_duplicates = 0

    print(f"\nGenerating curriculum dataset with custom distribution:")
    print(f"  Total main samples: {total_main_samples}")
    print(f"  Review ratio: {review_ratio:.0%}")
    print(f"  Stage distribution:")
    for t, c in task_distribution.items():
        print(f"    Stage {t}: {c} ({c / total_main_samples * 100:.1f}%)")

    def sample_review_stage(current_stage: int) -> int:
        """Sample a review stage from earlier stages using the same distribution."""
        earlier_stages = list(range(1, current_stage))  # Stages 1 to current_stage-1
        weights = [task_distribution[t] for t in earlier_stages]
        total = sum(weights)
        weights = [w / total for w in weights]
        return np.random.choice(earlier_stages, p=weights)

    def generate_single_sample(stage: int, phase: int) -> Tuple[bool, int]:
        """Generate a single sample, returns (success, duplicates_encountered)."""
        nonlocal duplicate_count, accepted_duplicates

        for attempt in range(max_retries):
            task = generate_task_by_stage(stage, **task_params)
            h = task_hash(task)

            if h not in seen_hashes:
                seen_hashes.add(h)
                conversation = task_to_conversation(task)
                sample = {
                    "task_type": task.task_type,
                    "task_id": stage,  # Stage ID (1-9)
                    "phase": phase,  # Which training phase this sample belongs to
                    "text": conversation,
                    "metadata": task.metadata,
                }
                dataset.append(sample)
                stage_counts[stage] += 1
                return True, 0
            else:
                duplicate_count += 1

        # Accept duplicate if retries exhausted
        accepted_duplicates += 1
        seen_hashes.add(h)
        conversation = task_to_conversation(task)
        sample = {
            "task_type": task.task_type,
            "task_id": stage,  # Stage ID (1-9)
            "phase": phase,  # Which training phase this sample belongs to
            "text": conversation,
            "metadata": task.metadata,
        }
        dataset.append(sample)
        stage_counts[stage] += 1
        return True, 0

    for phase in range(1, num_stages + 1):  # Stages 1-9
        num_main = task_distribution[phase]

        # Calculate review samples based on main samples for this phase
        num_review = int(num_main * review_ratio) if phase > 1 else 0  # No review for stage 1
        phase_total = num_main + num_review

        pbar = tqdm(total=phase_total, desc=f"Stage {phase}")

        # Create interleaved schedule: weave review samples throughout main samples
        # E.g., if num_main=100 and num_review=10, insert a review every ~10 main samples
        # We use a simple approach: after every N main samples, insert a review
        if num_review > 0:
            review_interval = num_main / num_review  # Insert review every N main samples
        else:
            review_interval = float('inf')

        main_count = 0
        review_count = 0
        next_review_at = review_interval  # Insert first review after this many main samples

        for i in range(phase_total):
            # Check if it's time for a review sample
            if review_count < num_review and main_count >= next_review_at:
                # Insert a review sample
                review_stage = sample_review_stage(phase)
                generate_single_sample(review_stage, phase)
                review_count += 1
                next_review_at += review_interval  # Schedule next review
            else:
                # Insert a main sample
                generate_single_sample(phase, phase)
                main_count += 1
            pbar.update(1)

        pbar.close()
        print(f"  Stage {phase}: {num_main} main + {num_review} review = {phase_total} samples (interleaved)")

    print(f"\nDuplicates skipped: {duplicate_count}")
    print(f"Duplicates accepted (retries exhausted): {accepted_duplicates}")
    print("\nFinal stage distribution:")
    total = len(dataset)
    for stage_id, count in stage_counts.items():
        print(f"  Stage {stage_id}: {count} ({count / total * 100:.1f}%)")

    return dataset


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_dataset(dataset: List[Dict], output_dir: str, split_ratio: float = 0.95, shuffle: bool = False):
    """Save dataset to disk with train/test split.

    Args:
        dataset: List of samples
        output_dir: Output directory
        split_ratio: Train/test split ratio
        shuffle: If True, shuffle before saving (False for curriculum)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only shuffle if requested (not for curriculum)
    if shuffle:
        random.shuffle(dataset)
    split_idx = int(len(dataset) * split_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # Save train set
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for sample in train_data:
            f.write(json.dumps(convert_to_json_serializable(sample)) + "\n")
    print(f"Saved {len(train_data)} training samples to {train_path}")

    # Save test set
    test_path = os.path.join(output_dir, "test.jsonl")
    with open(test_path, "w") as f:
        for sample in test_data:
            f.write(json.dumps(convert_to_json_serializable(sample)) + "\n")
    print(f"Saved {len(test_data)} test samples to {test_path}")

    # Save metadata
    metadata = {
        "total_samples": len(dataset),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "split_ratio": split_ratio,
    }
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


def main():
    args = parse_args()

    # Set seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize ECG loader for Task 6 (uses real MIMIC data)
    print(f"Initializing ECG loader with dataset: {args.mimic_dataset} (fold {args.mimic_fold})...")
    init_ecg_loader(dataset_name=args.mimic_dataset, fold=args.mimic_fold)

    # Determine curriculum mode
    use_curriculum = args.curriculum and not args.no_curriculum

    # Parse task weights if provided (only used when not using curriculum)
    task_weights = None
    if args.task_weights:
        task_weights = json.loads(args.task_weights)
        task_weights = {int(k): v for k, v in task_weights.items()}

    # Parse task distribution if provided (curriculum mode only)
    task_distribution = None
    if args.task_distribution:
        task_distribution = json.loads(args.task_distribution)
        task_distribution = {int(k): v for k, v in task_distribution.items()}

    # Stage parameters
    stage_params = {
        "stage1_num_values": args.stage1_num_values,
        "stage1_num_tokens": args.stage1_num_tokens,
        "stage4_duration": args.stage4_duration,
        "stage5_duration": args.stage5_duration,
        "stage6_duration": args.stage6_duration,
        "stage6_chunk_size": args.stage6_chunk_size,
        "stage7_chunk_size": args.stage7_chunk_size,
        "stage8_context_tokens": args.stage8_context_tokens,
        "stage9_context_tokens": args.stage9_context_tokens,
    }

    print(f"Generating pretraining samples...")
    print(f"  Curriculum mode: {use_curriculum}")
    if use_curriculum:
        print(f"  Review ratio: {args.review_ratio:.0%}")
        if task_distribution:
            print(f"  Using custom task distribution")
        else:
            print(f"  Using equal distribution ({args.num_samples} total)")

    dataset = generate_dataset(
        args.num_samples,
        task_weights,
        curriculum=use_curriculum,
        review_ratio=args.review_ratio,
        task_distribution=task_distribution,
        **stage_params
    )

    print(f"\nSaving dataset to {args.output_dir}...")
    # Don't shuffle if using curriculum (preserve order)
    save_dataset(dataset, args.output_dir, shuffle=not use_curriculum)

    print("\nDone!")


if __name__ == "__main__":
    main()
