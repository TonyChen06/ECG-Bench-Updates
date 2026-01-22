"""
Pretrain script for LLM pretraining from scratch.

This script:
1. Loads a pretrain corpus from parquet files
2. Optionally resets model weights to random initialization
3. Trains the model using causal LM loss
4. Saves checkpoints

Usage:
    python -m ecg_bench.pretrain_elm \
        --llm smollm-135m \
        --reset_weights \
        --pretrain_corpus ./ecg_bench/data/pretrain_corpus \
        --seq_len 2048 \
        --batch_size 8 \
        --epochs 1 \
        --lr 1e-4
"""

import gc
import torch

torch.set_num_threads(6)

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ecg_bench.configs.config import get_args
from ecg_bench.configs.constants import HF_LLMS, HF_CACHE_DIR, RUNS_DIR, ECG_RAW_TOKEN_PREFIX, ECG_RAW_NUM_BINS
from ecg_bench.utils.gpu_setup import init_dist, cleanup, GPUSetup, is_main, get_world_size, get_rank
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.dataloaders.pretrain_dataloader import (
    PretrainDataset,
    load_pretrain_corpus,
    load_ecg_data,
    mix_datasets,
)
from ecg_bench.elms.build_llm import BuildLLM
from ecg_bench.runners.elm_trainer import train
from ecg_bench.utils.file_manager import setup_experiment_folders
from ecg_bench.optimizers.scheduler import get_optimizer
from ecg_bench.utils.checkpoint import CheckpointManager
from ecg_bench.utils.wandb_setup import setup_wandb, cleanup_wandb
import wandb


def build_pretrain_tokenizer(args):
    """Build tokenizer for pretraining."""
    llm_tokenizer = AutoTokenizer.from_pretrained(
        HF_LLMS[args.llm]["tokenizer"],
        cache_dir=HF_CACHE_DIR,
    )

    # Set pad token if not present
    if getattr(llm_tokenizer, "pad_token", None) is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    # Add ECG raw tokens if ecg_raw mode is enabled
    if getattr(args, 'ecg_raw', False):
        ecg_tokens = [f"{ECG_RAW_TOKEN_PREFIX}{i}" for i in range(ECG_RAW_NUM_BINS)]
        llm_tokenizer.add_tokens(ecg_tokens)
        if is_main():
            print(f"Added {len(ecg_tokens)} ECG raw tokens to tokenizer")

    if args.dev and is_main():
        print("Tokenizer Info:")
        print(f"  Vocab size: {len(llm_tokenizer)}")
        print(f"  Pad token: {llm_tokenizer.pad_token} (id: {llm_tokenizer.pad_token_id})")
        print(f"  EOS token: {llm_tokenizer.eos_token} (id: {llm_tokenizer.eos_token_id})")

    return llm_tokenizer


def build_pretrain_dataloader(args, llm_tokenizer):
    """Build dataloader for pretraining."""
    # Load corpus
    if is_main():
        print(f"Loading pretrain corpus from {args.pretrain_corpus}...")
    corpus = load_pretrain_corpus(args.pretrain_corpus)
    if is_main():
        print(f"Loaded {len(corpus):,} pretrain samples")

    # Optionally mix in ECG data
    if args.ecg_data:
        ecg_raw = getattr(args, 'ecg_raw', False)
        if is_main():
            print(f"Loading ECG data from {args.ecg_data}...")
            if ecg_raw:
                print(f"  ECG raw mode enabled - will include ECG signal tokens")
        ecg_data = load_ecg_data(args.ecg_data, args.ecg_num_examples, seed=args.seed, ecg_raw=ecg_raw)
        if is_main():
            print(f"Loaded {len(ecg_data):,} ECG samples")
            print(f"Mixing datasets...")
        corpus = mix_datasets(corpus, ecg_data, seed=args.seed)
        if is_main():
            print(f"Total combined samples: {len(corpus):,}")

    # Create dataset
    dataset = PretrainDataset(corpus, llm_tokenizer, args)

    # Create sampler for distributed training
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=args.seed,
            shuffle=True,
        )
    else:
        sampler = None

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=2,
        sampler=sampler,
        pin_memory=True,
    )

    return dataloader


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "pretrain"
    args = get_args(mode)

    # Validate args
    if args.llm is None:
        raise ValueError("--llm is required for pretraining")
    if args.pretrain_corpus is None:
        raise ValueError("--pretrain_corpus is required for pretraining")

    if args.distributed:
        init_dist()

    run_dir = setup_experiment_folders(
        f"{RUNS_DIR}/pretrain/{args.llm}",
        args,
    )
    if is_main():
        print(f"Run dir: {run_dir}")
        if args.wandb:
            setup_wandb(args)

    set_seed(getattr(args, "seed", 1337))

    # Build tokenizer
    llm_tokenizer = build_pretrain_tokenizer(args)

    # Build dataloader
    dataloader = build_pretrain_dataloader(args, llm_tokenizer)

    # Build model
    # We need to set some attributes that BuildLLM expects
    args.encoder = None  # No encoder for pretraining
    args.output_hidden_states = False

    build_llm = BuildLLM(args, llm_tokenizer, llm_tokenizer.pad_token_id, llm_tokenizer.eos_token_id)
    llm_components = build_llm.build_llm()
    elm = llm_components["elm"]

    # Setup GPU
    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(elm, llm_components["find_unused_parameters"])
    if args.dev:
        gpu_setup.print_model_device(elm, args.llm)

    # Setup optimizer
    optimizer = get_optimizer(args, elm)

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(run_dir, args)

    # Training loop
    if is_main():
        print(f"\nStarting pretraining...")
        print(f"  Model: {args.llm}")
        print(f"  Reset weights: {args.reset_weights}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Epochs: {args.epochs}")
        if args.ecg_data:
            print(f"  ECG data: {args.ecg_data}")
            print(f"  ECG raw: {getattr(args, 'ecg_raw', False)}")

    for epoch in range(args.epochs):
        train_result = train(elm, dataloader, optimizer, epoch, args, checkpoint_manager)
        if args.wandb and is_main():
            wandb.log({"train/epoch_loss": train_result["average_loss"], "epoch": epoch})
        if checkpoint_manager.save_epoch(train_result["average_loss"]):
            checkpoint_manager.save_checkpoint(elm, optimizer, epoch, -1, is_best=True, prefix="epoch_")
        if checkpoint_manager.stop_early():
            if is_main():
                print(f"Early stopping at epoch {epoch}")
            break

    if args.distributed:
        cleanup()

    if is_main() and args.wandb:
        cleanup_wandb()

    if is_main():
        print(f"\nPretraining complete!")
        print(f"  Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
