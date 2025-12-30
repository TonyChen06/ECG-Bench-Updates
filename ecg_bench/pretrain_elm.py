"""
Pretraining script for ECG token understanding.

This script trains the LLM on synthetic pretraining tasks to help it
understand the ECG token vocabulary before fine-tuning on real ECG data.

Supports detachable stages:
    # Train stage 0 from scratch
    python -m ecg_bench.pretrain_elm --stage 0 --llm qwen3-4b-instruct ...

    # Train stage 1, resuming from stage 0 checkpoint
    python -m ecg_bench.pretrain_elm --stage 1 --resume_from ./runs/pretraining/elm/0 ...

    # Train all stages together (original behavior)
    python -m ecg_bench.pretrain_elm --llm qwen3-4b-instruct ...

Usage:
    python -m ecg_bench.pretrain_elm \
        --llm qwen3-4b-instruct \
        --pretrain_data ./ecg_bench/data/pretraining \
        --peft \
        --epochs 3 \
        --lr 1e-4
"""

import gc
import os
import torch
from pathlib import Path

torch.set_num_threads(6)

from ecg_bench.configs.config import get_args
from ecg_bench.utils.gpu_setup import init_dist, cleanup, GPUSetup, is_main, barrier
from ecg_bench.utils.set_seed import set_seed
from ecg_bench.pretraining.pretrain_dataloader import PretrainDataset
from ecg_bench.elms.build_llm import BuildLLM
from ecg_bench.elms.llm.hf_llm import HuggingFaceLLM
from ecg_bench.runners.scheduled_sampling_trainer import train_with_scheduled_sampling
from ecg_bench.utils.file_manager import setup_experiment_folders
from ecg_bench.configs.constants import RUNS_DIR, HF_LLMS, ECG_RAW_NUM_BINS, ECG_RAW_TOKEN_PREFIX
from ecg_bench.optimizers.scheduler import get_optimizer
from ecg_bench.utils.checkpoint import CheckpointManager
from ecg_bench.utils.wandb_setup import setup_wandb, setup_pretrain_metrics, cleanup_wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ecg_bench.utils.gpu_setup import get_world_size, get_rank
from transformers import AutoTokenizer
import wandb


def build_tokenizer(args):
    """Build and modify tokenizer for pretraining.

    If resuming from a checkpoint, loads the tokenizer from that checkpoint.
    Otherwise, builds a fresh tokenizer and adds ECG tokens.
    """
    if args.resume_from:
        # Load tokenizer from previous checkpoint
        resume_path = Path(args.resume_from)

        # Handle both run directory and direct checkpoint file paths
        if resume_path.suffix == ".pt":
            # Direct checkpoint file - go up to run directory
            run_dir = resume_path.parent.parent  # .../checkpoints/epoch_best.pt -> ...
        else:
            run_dir = resume_path

        tokenizer_path = run_dir / "tokenizer"
        if tokenizer_path.exists():
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
            if is_main():
                print(f"Loaded tokenizer from {tokenizer_path}")
                print(f"Tokenizer vocab size: {len(tokenizer)}")
            return tokenizer
        else:
            if is_main():
                print(f"Warning: No tokenizer found at {tokenizer_path}, building fresh")

    # Build fresh tokenizer
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


def load_checkpoint(model, checkpoint_path, args):
    """Load model weights from checkpoint.

    Args:
        checkpoint_path: Either a run directory (e.g., ./runs/pretraining/elm/stage_0/0)
                        or a direct checkpoint file path (e.g., .../epoch_best.pt)

    Restores:
    - Model weights only

    Does NOT restore:
    - Optimizer state (fresh momentum for each stage)
    - LR schedule (fresh warmup for each stage)
    - Epoch counter (each stage trains for its own epochs)
    """
    checkpoint_path = Path(checkpoint_path)

    # Handle both run directory and direct checkpoint file paths
    if checkpoint_path.suffix == ".pt":
        # Direct checkpoint file path
        checkpoint_file = checkpoint_path
    else:
        # Run directory - look for best checkpoint
        checkpoint_file = checkpoint_path / "checkpoints" / "epoch_best.pt"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    if is_main():
        print(f"Loading checkpoint from {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    # Load model state only
    if args.distributed:
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    if is_main():
        print(f"Loaded model weights from epoch {checkpoint['epoch']}")


def build_dataloader(args, tokenizer, mode="train"):
    """Build dataloader for pretraining."""
    dataset = PretrainDataset(
        data_path=args.pretrain_data,
        mode=mode,
        llm_tokenizer_components={"llm_tokenizer": tokenizer},
        args=args,
    )

    # Determine if we should shuffle (disabled for curriculum learning)
    use_curriculum = getattr(args, "curriculum", True) and not getattr(args, "no_curriculum", False)
    should_shuffle = (mode == "train") and not use_curriculum

    # Use DistributedSampler for multi-GPU training
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=getattr(args, "seed", 1337),
            shuffle=should_shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = should_shuffle

    if is_main() and use_curriculum:
        print("Curriculum learning enabled: preserving task order (no shuffling)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
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

    # Create run directory with stage suffix if training specific stage
    base_run_dir = f"{RUNS_DIR}/pretraining/elm"
    if args.stage is not None:
        base_run_dir = f"{RUNS_DIR}/pretraining/elm/stage_{args.stage}"

    run_dir = setup_experiment_folders(base_run_dir, args)

    if is_main():
        print(f"Run dir: {run_dir}")
        if args.stage is not None:
            print(f"Training stage: {args.stage}")
        if args.resume_from:
            print(f"Resuming from: {args.resume_from}")
        if args.wandb:
            setup_wandb(args)
            setup_pretrain_metrics()

    set_seed(getattr(args, "seed", 1337))

    # Build tokenizer first (will load from checkpoint if resuming)
    tokenizer = build_tokenizer(args)

    # Build dataloader (will filter by stage if specified)
    dataloader = build_dataloader(args, tokenizer, mode="train")
    if is_main():
        print(f"Training samples: {len(dataloader.dataset)}")

    # Build model
    llm_components = build_model(args, tokenizer)

    # Setup GPU (key is "elm" when encoder=None)
    # Stage 1 freezes LoRA, so we need find_unused_parameters=True when starting from Stage 1
    # to avoid DDP errors about unused parameters not receiving gradients
    starting_stage = getattr(args, "stage", None)
    use_ecg_warmup = not getattr(args, "no_ecg_warmup", False)
    will_freeze_lora = use_ecg_warmup and (starting_stage is None or starting_stage == 1)
    find_unused = llm_components["find_unused_parameters"] or will_freeze_lora

    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(llm_components["elm"], find_unused)

    if args.dev:
        gpu_setup.print_model_device(elm, f"{args.llm}_pretrain")

    # Setup optimizer
    optimizer = get_optimizer(args, elm)

    # Load checkpoint if resuming from previous stage (model weights only)
    if args.resume_from:
        load_checkpoint(elm, args.resume_from, args)
        barrier()  # Ensure all processes have loaded

    checkpoint_manager = CheckpointManager(run_dir, args)

    # Save tokenizer with ECG tokens for seamless use in downstream training
    if is_main():
        tokenizer_save_path = os.path.join(run_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_save_path)
        print(f"Saved tokenizer with ECG tokens to {tokenizer_save_path}")

    # Training loop
    stage1_mode_initialized = False  # Track if Stage 1 mode has been set up
    for epoch in range(args.epochs):
        train_result = train_with_scheduled_sampling(
            elm,
            dataloader,
            optimizer,
            epoch,
            args,
            checkpoint_manager,
            ss_horizon_start=getattr(args, "ss_horizon_start", 1),
            ss_horizon_end=getattr(args, "ss_horizon_end", 10),
            max_grad_norm=getattr(args, "max_grad_norm", 0),
            stage1_mode_initialized=stage1_mode_initialized,
        )
        stage1_mode_initialized = train_result.get("stage1_mode_initialized", False)

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

    # Print next stage instructions
    if is_main() and args.stage is not None and args.stage < 8:
        print(f"\n{'='*60}")
        print(f"Stage {args.stage} complete!")
        print(f"To continue with stage {args.stage + 1}, run:")
        print(f"  --stage {args.stage + 1} --resume_from {run_dir}")
        print(f"{'='*60}\n")

    if args.distributed:
        cleanup()

    if is_main() and args.wandb:
        cleanup_wandb()


if __name__ == "__main__":
    main()
