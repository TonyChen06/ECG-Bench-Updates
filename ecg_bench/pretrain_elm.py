"""
Pretraining script for ECG token understanding.

This script trains the LLM on synthetic pretraining tasks to help it
understand the ECG token vocabulary before fine-tuning on real ECG data.

Usage:
    python -m ecg_bench.pretrain_elm \
        --llm qwen3-4b-instruct \
        --pretrain_data ./ecg_bench/data/pretraining \
        --peft \
        --epochs 3 \
        --lr 1e-4
"""

import gc
import torch

torch.set_num_threads(6)

from ecg_bench.configs.config import get_args
from ecg_bench.utils.gpu_setup import init_dist, cleanup, GPUSetup, is_main
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.pretraining.pretrain_dataloader import PretrainDataset
from ecg_bench.elms.build_llm import BuildLLM
from ecg_bench.elms.llm.hf_llm import HuggingFaceLLM
from ecg_bench.runners.elm_trainer import train
from ecg_bench.utils.file_manager import setup_experiment_folders
from ecg_bench.configs.constants import RUNS_DIR, HF_LLMS, ECG_RAW_NUM_BINS, ECG_RAW_TOKEN_PREFIX
from ecg_bench.optimizers.scheduler import get_optimizer
from ecg_bench.utils.checkpoint import CheckpointManager
from ecg_bench.utils.wandb_setup import setup_wandb, cleanup_wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb


def build_tokenizer(args):
    """Build and modify tokenizer for pretraining."""
    tokenizer = AutoTokenizer.from_pretrained(
        HF_LLMS[args.llm]["tokenizer"],
        cache_dir="./.huggingface",
    )

    # Set pad token if not set
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add ECG raw tokens
    ecg_tokens = [f"{ECG_RAW_TOKEN_PREFIX}{i}" for i in range(ECG_RAW_NUM_BINS)]
    tokenizer.add_tokens(ecg_tokens)

    if is_main():
        print(f"Tokenizer vocab size after adding ECG tokens: {len(tokenizer)}")

    return tokenizer


def build_dataloader(args, tokenizer, mode="train"):
    """Build dataloader for pretraining."""
    dataset = PretrainDataset(
        data_path=args.pretrain_data,
        mode=mode,
        llm_tokenizer_components={"llm_tokenizer": tokenizer},
        args=args,
    )

    shuffle = mode == "train"
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )

    return dataloader


def build_model(args, tokenizer):
    """Build LLM for pretraining."""
    # Set encoder to None for pretraining (no encoder needed)
    args.encoder = None

    llm_components = BuildLLM(
        args,
        tokenizer,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    ).build_llm()

    return llm_components


def main():
    gc.collect()
    torch.cuda.empty_cache()
    mode = "pretrain"
    args = get_args(mode)

    if args.distributed:
        init_dist()

    run_dir = setup_experiment_folders(
        f"{RUNS_DIR}/pretraining/elm",
        args,
    )

    if is_main():
        print(f"Run dir: {run_dir}")
        if args.wandb:
            setup_wandb(args)

    set_seed(getattr(args, "seed", 1337))

    # Build tokenizer first
    tokenizer = build_tokenizer(args)

    # Build dataloader
    dataloader = build_dataloader(args, tokenizer, mode="train")
    if is_main():
        print(f"Training samples: {len(dataloader.dataset)}")

    # Build model
    llm_components = build_model(args, tokenizer)

    # Setup GPU
    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(llm_components["llm"], llm_components["find_unused_parameters"])

    if args.dev:
        gpu_setup.print_model_device(elm, f"{args.llm}_pretrain")

    # Setup optimizer
    optimizer = get_optimizer(args, elm)
    checkpoint_manager = CheckpointManager(run_dir, args)

    # Training loop
    for epoch in range(args.epochs):
        train_result = train(elm, dataloader, optimizer, epoch, args, checkpoint_manager)

        if args.wandb and is_main():
            wandb.log({"pretrain/epoch_loss": train_result["average_loss"], "epoch": epoch})

        if is_main():
            print(f"Epoch {epoch}: average_loss = {train_result['average_loss']:.4f}")

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


if __name__ == "__main__":
    main()
