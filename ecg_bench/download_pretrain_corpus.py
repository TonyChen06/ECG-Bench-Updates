"""
Download script for pretraining corpus from HuggingFace smollm-corpus.

Downloads 1% (by default) of Cosmopedia v2 and FineWeb-Edu-Dedup,
maintaining the original distribution (~11% Cosmopedia, ~89% FineWeb-Edu).

Usage:
    python -m ecg_bench.download_pretrain_corpus --sample_ratio 0.01
"""

import argparse
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Download pretrain corpus from smollm-corpus")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.01,
        help="Ratio of data to sample (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ecg_bench/data/pretrain_corpus",
        help="Output directory for downloaded corpus",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of processes for parallel processing",
    )
    return parser.parse_args()


def sample_streaming_dataset(dataset_name: str, config: str, num_samples: int, seed: int) -> list[dict]:
    """Sample from a streaming dataset."""
    print(f"Loading {config} from {dataset_name} (streaming)...")
    ds = load_dataset(dataset_name, config, split="train", streaming=True)

    # Shuffle with buffer and take samples
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    samples = []
    print(f"Sampling {num_samples:,} examples from {config}...")
    for i, item in enumerate(tqdm(ds, total=num_samples, desc=f"Sampling {config}")):
        if i >= num_samples:
            break
        # Extract only the text field
        if config == "cosmopedia-v2":
            samples.append({"text": item["text"], "source": "cosmopedia-v2"})
        elif config == "fineweb-edu-dedup":
            samples.append({"text": item["text"], "source": "fineweb-edu-dedup"})

    return samples


def main():
    args = get_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset sizes (approximate row counts)
    COSMOPEDIA_ROWS = 39_100_000  # ~39.1M rows
    FINEWEB_ROWS = 190_000_000     # ~190M rows

    # Calculate samples to take (maintaining original distribution)
    cosmopedia_samples = int(COSMOPEDIA_ROWS * args.sample_ratio)
    fineweb_samples = int(FINEWEB_ROWS * args.sample_ratio)

    print(f"Sample ratio: {args.sample_ratio}")
    print(f"Cosmopedia samples: {cosmopedia_samples:,}")
    print(f"FineWeb-Edu samples: {fineweb_samples:,}")
    print(f"Total samples: {cosmopedia_samples + fineweb_samples:,}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Sample from Cosmopedia v2
    cosmopedia_data = sample_streaming_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        cosmopedia_samples,
        args.seed,
    )

    # Sample from FineWeb-Edu-Dedup
    fineweb_data = sample_streaming_dataset(
        "HuggingFaceTB/smollm-corpus",
        "fineweb-edu-dedup",
        fineweb_samples,
        args.seed + 1,  # Different seed for variety
    )

    # Combine and save
    all_data = cosmopedia_data + fineweb_data
    print(f"\nTotal collected: {len(all_data):,} samples")

    # Convert to HuggingFace Dataset and save as parquet
    print("Converting to Dataset...")
    dataset = Dataset.from_list(all_data)

    # Shuffle the combined dataset
    print("Shuffling combined dataset...")
    dataset = dataset.shuffle(seed=args.seed)

    # Save as parquet
    output_path = output_dir / "pretrain_corpus.parquet"
    print(f"Saving to {output_path}...")
    dataset.to_parquet(str(output_path))

    # Also save metadata
    metadata = {
        "sample_ratio": args.sample_ratio,
        "cosmopedia_samples": len(cosmopedia_data),
        "fineweb_samples": len(fineweb_data),
        "total_samples": len(all_data),
        "seed": args.seed,
    }

    import json
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Corpus: {output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Cosmopedia: {len(cosmopedia_data):,} samples ({len(cosmopedia_data)/len(all_data)*100:.1f}%)")
    print(f"  FineWeb-Edu: {len(fineweb_data):,} samples ({len(fineweb_data)/len(all_data)*100:.1f}%)")


if __name__ == "__main__":
    main()
