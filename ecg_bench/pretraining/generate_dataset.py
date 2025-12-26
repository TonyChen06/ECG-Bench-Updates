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
from typing import Dict, List, Optional, Set
from tqdm import tqdm
import numpy as np

from ecg_bench.pretraining.task_generators import (
    generate_task_0, generate_task_1, generate_task_2,
    generate_task_3, generate_task_4, generate_task_5,
    generate_task_6, task_to_conversation, Task,
)


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

    # Task-specific parameters
    parser.add_argument("--task1_num_values", type=int, default=50, help="Number of values for Task 1")
    parser.add_argument("--task2_num_tokens", type=int, default=20, help="Number of tokens for Task 2")
    parser.add_argument("--task3_duration", type=float, default=2.0, help="Duration for Task 3 (seconds)")
    parser.add_argument("--task4_duration", type=float, default=1.0, help="Duration for Task 4 (seconds)")
    parser.add_argument("--task5_duration", type=float, default=2.0, help="Duration for Task 5 (seconds)")
    parser.add_argument("--task5_chunk_size", type=int, default=100, help="Chunk size for Task 5")
    parser.add_argument("--task6_duration", type=float, default=2.0, help="Duration for Task 6 (seconds)")
    parser.add_argument("--task6_chunk_size", type=int, default=100, help="Chunk size for Task 6")

    return parser.parse_args()


def generate_task_by_type(
    task_type: int,
    task1_num_values: int = 50,
    task2_num_tokens: int = 20,
    task3_duration: float = 2.0,
    task4_duration: float = 1.0,
    task5_duration: float = 2.0,
    task5_chunk_size: int = 100,
    task6_duration: float = 2.0,
    task6_chunk_size: int = 100,
) -> Task:
    """Generate a task of the specified type."""
    if task_type == 0:
        return generate_task_0()
    elif task_type == 1:
        return generate_task_1(num_values=task1_num_values)
    elif task_type == 2:
        return generate_task_2(num_tokens=task2_num_tokens)
    elif task_type == 3:
        return generate_task_3(duration=task3_duration)
    elif task_type == 4:
        return generate_task_4(duration=task4_duration)
    elif task_type == 5:
        return generate_task_5(duration=task5_duration, chunk_size=task5_chunk_size)
    elif task_type == 6:
        return generate_task_6(duration=task6_duration, chunk_size=task6_chunk_size)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


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
    **task_params,
) -> List[Dict]:
    """
    Generate the full dataset with deduplication.

    Args:
        num_samples: Target number of unique samples
        task_weights: Weights for each task type
        max_retries: Max attempts to generate a unique task before giving up
        **task_params: Parameters passed to task generators
    """
    if task_weights is None:
        # Default weights - slightly favor simpler tasks
        task_weights = {
            0: 2.0,  # Token comparison (simple, good for warmup)
            1: 1.5,  # Values to tokens
            2: 1.5,  # Tokens to values
            3: 2.0,  # Wave classification (important)
            4: 1.5,  # Wave transformation
            5: 1.0,  # Wave reconstruction (complex)
            6: 0.5,  # ECG reconstruction (complex, fewer needed)
        }

    dataset = []
    seen_hashes: Set[str] = set()
    task_counts = {i: 0 for i in range(7)}
    duplicate_count = 0

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
            # (this shouldn't happen often with random generation)
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
    print("\nTask distribution:")
    for task_id, count in task_counts.items():
        print(f"  Task {task_id}: {count} ({count / len(dataset) * 100:.1f}%)")

    return dataset


def save_dataset(dataset: List[Dict], output_dir: str, split_ratio: float = 0.95):
    """Save dataset to disk with train/test split."""
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle and split
    random.shuffle(dataset)
    split_idx = int(len(dataset) * split_ratio)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    # Save train set
    train_path = os.path.join(output_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")
    print(f"Saved {len(train_data)} training samples to {train_path}")

    # Save test set
    test_path = os.path.join(output_dir, "test.jsonl")
    with open(test_path, "w") as f:
        for sample in test_data:
            f.write(json.dumps(sample) + "\n")
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

    # Parse task weights if provided
    task_weights = None
    if args.task_weights:
        task_weights = json.loads(args.task_weights)
        task_weights = {int(k): v for k, v in task_weights.items()}

    # Task parameters
    task_params = {
        "task1_num_values": args.task1_num_values,
        "task2_num_tokens": args.task2_num_tokens,
        "task3_duration": args.task3_duration,
        "task4_duration": args.task4_duration,
        "task5_duration": args.task5_duration,
        "task5_chunk_size": args.task5_chunk_size,
        "task6_duration": args.task6_duration,
        "task6_chunk_size": args.task6_chunk_size,
    }

    print(f"Generating {args.num_samples} pretraining samples...")
    dataset = generate_dataset(args.num_samples, task_weights, **task_params)

    print(f"\nSaving dataset to {args.output_dir}...")
    save_dataset(dataset, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
