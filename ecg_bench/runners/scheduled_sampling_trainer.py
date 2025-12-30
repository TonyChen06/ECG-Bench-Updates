"""
Scheduled Sampling Trainer for ECG Pretraining.

Implements scheduled sampling for Tasks 5 and 6 where the model must generate
long ECG sequences. During training, some ground truth tokens are replaced with
the model's own predictions to teach the model to handle error accumulation.

The sampling probability starts low (mostly teacher forcing) and increases
over training to gradually expose the model to its own predictions.
"""

import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from typing import Optional, Callable

from ecg_bench.utils.gpu_setup import is_main, train_dev_break


def set_ecg_embedding_only_mode(elm, enable: bool, is_distributed: bool = False):
    """
    Configure training mode for ECG pretraining.

    embed_tokens and lm_head are in modules_to_save, so they are trained directly
    (not via LoRA). This means ALL embedding rows are trainable, including both
    original tokens and ECG tokens.

    Stage 1 (enable=True):
    - embed_tokens/lm_head: TRAINABLE (via modules_to_save)
    - LoRA (attention/MLP): FROZEN

    Stages 2+ (enable=False):
    - embed_tokens/lm_head: TRAINABLE (via modules_to_save)
    - LoRA (attention/MLP): TRAINABLE

    Args:
        elm: The model (may be DDP wrapped)
        enable: True for Stage 1 (freeze LoRA), False for later stages
        is_distributed: Whether using DDP
    """
    # Get the underlying model
    model = elm.module if is_distributed else elm
    llm = model.llm

    for name, param in llm.named_parameters():
        # Check if this is an embedding-related parameter (modules_to_save)
        is_embedding_param = "embed_tokens" in name or "lm_head" in name

        if is_embedding_param:
            # Embedding weights: always trainable (via modules_to_save)
            param.requires_grad = True
        elif "lora" in name.lower():
            # LoRA (attention/MLP): frozen in Stage 1, unfrozen later
            param.requires_grad = not enable
        else:
            # Other params (base model weights): frozen (we only train via LoRA)
            param.requires_grad = False


def linear_schedule(step: int, total_steps: int, start: float = 0.0, end: float = 0.5) -> float:
    """
    Linear schedule for sampling probability.

    Args:
        step: Current training step
        total_steps: Total training steps
        start: Starting probability (usually 0 = full teacher forcing)
        end: Ending probability (usually 0.5 = 50% self-generated)

    Returns:
        Current sampling probability
    """
    if total_steps == 0:
        return start
    progress = min(step / total_steps, 1.0)
    return start + (end - start) * progress


def exponential_schedule(step: int, total_steps: int, start: float = 0.0, end: float = 0.5, k: float = 5.0) -> float:
    """
    Exponential schedule for sampling probability (slower initial increase).

    Args:
        step: Current training step
        total_steps: Total training steps
        start: Starting probability
        end: Ending probability
        k: Exponential factor (higher = slower initial increase)

    Returns:
        Current sampling probability
    """
    if total_steps == 0:
        return start
    progress = min(step / total_steps, 1.0)
    # Exponential interpolation: slower at start, faster at end
    exp_progress = (1 - torch.exp(torch.tensor(-k * progress))) / (1 - torch.exp(torch.tensor(-k)))
    return start + (end - start) * exp_progress.item()


def horizon_schedule(step: int, total_steps: int, start: int = 1, end: int = 10) -> int:
    """
    Linear schedule for prediction horizon.

    Starts with next-1 token prediction and gradually increases to next-N.

    Args:
        step: Current training step
        total_steps: Total training steps
        start: Starting horizon (1 = standard next token prediction)
        end: Ending horizon (e.g., 10 = predict 10 tokens ahead)

    Returns:
        Current prediction horizon (integer)
    """
    if total_steps == 0:
        return start
    progress = min(step / total_steps, 1.0)
    return int(start + (end - start) * progress)


def apply_scheduled_sampling(
    model,
    batch: dict,
    horizon: int,
    device: torch.device,
) -> dict:
    """
    Apply horizon-based scheduled sampling to a batch.

    For samples with scheduled sampling enabled, generate `horizon` tokens
    autoregressively at regular intervals throughout the ECG sequence.
    The horizon starts at 1 (next token prediction) and increases throughout
    training to teach longer-range dependencies.

    Args:
        model: The LLM model
        batch: Batch dictionary with input_ids, labels, etc.
        horizon: Number of tokens to generate autoregressively (1 = standard next token)
        device: Device to run on

    Returns:
        Modified batch with segments replaced by model predictions
    """
    if horizon <= 1:
        return batch  # Standard next token prediction, no modification needed

    input_ids = batch["elm_input_ids"].clone()
    use_ss = batch["use_scheduled_sampling"]  # (batch_size, 1)
    ecg_positions = batch["ecg_token_positions"]  # (batch_size, num_positions)

    batch_size = input_ids.shape[0]

    for b in range(batch_size):
        # Skip if this sample doesn't use scheduled sampling
        if not use_ss[b, 0]:
            continue

        positions = ecg_positions[b]
        positions = positions[positions >= 0].tolist()  # Filter out -1 padding

        if len(positions) < horizon:
            continue

        # Generate a SMALL fixed number of segments to avoid excessive slowdown
        # Each segment requires `horizon` forward passes, so we limit to 3 segments max
        # This provides exposure to autoregressive errors without massive overhead
        max_segments = 3
        num_segments = min(max_segments, max(1, len(positions) // (horizon * 4)))
        segment_spacing = len(positions) // (num_segments + 1)  # +1 to avoid edges

        for seg_idx in range(num_segments):
            # Start segments at 1/4, 2/4, 3/4 through the sequence
            start_idx = (seg_idx + 1) * segment_spacing
            if start_idx + horizon > len(positions):
                break

            segment_positions = positions[start_idx:start_idx + horizon]

            # Generate this segment autoregressively
            input_ids[b] = _generate_segment_autoregressive(
                model, input_ids[b], segment_positions,
                batch["elm_attention_mask"][b], device
            )

    # Update batch with modified input_ids
    batch = dict(batch)  # Make a copy
    batch["elm_input_ids"] = input_ids

    return batch


def _generate_segment_autoregressive(
    model,
    input_ids: torch.Tensor,
    segment_positions: list,
    attention_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a contiguous segment of tokens autoregressively.

    The model generates each token based on its own previous predictions,
    allowing errors to accumulate naturally.

    Args:
        model: The LLM model (may be wrapped in DDP)
        input_ids: Input token IDs (1D tensor)
        segment_positions: List of positions to generate
        attention_mask: Attention mask (1D tensor)
        device: Device to run on

    Returns:
        Modified input_ids with segment replaced by model predictions
    """
    input_ids = input_ids.clone()

    # Handle DDP wrapper - get underlying module if wrapped
    unwrapped_model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for pos in segment_positions:
            if pos <= 0 or pos >= len(input_ids):
                continue

            # Forward pass up to current position
            outputs = unwrapped_model.llm(
                input_ids=input_ids[:pos].unsqueeze(0).to(device),
                attention_mask=attention_mask[:pos].unsqueeze(0).to(device),
            )

            # Get prediction for current position
            logits = outputs.logits[0, -1, :]  # Last position predicts next token
            pred_token = torch.argmax(logits).item()

            # Replace ground truth with model prediction
            input_ids[pos] = pred_token

    return input_ids


def train_with_scheduled_sampling(
    elm,
    dataloader,
    optimizer,
    epoch: int,
    args,
    checkpoint_manager=None,
    global_step: int = 0,
    total_training_steps: int = None,
    ss_horizon_start: int = 1,
    ss_horizon_end: int = 10,
    max_grad_norm: float = 1.0,
    stage1_mode_initialized: bool = False,
):
    """
    Training loop with horizon-based scheduled sampling for ECG pretraining.

    The prediction horizon starts at 1 (standard next token prediction) and
    gradually increases to ss_horizon_end throughout training. This teaches
    the model to handle increasingly long autoregressive generation.

    Args:
        elm: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer with step_and_update_lr method
        epoch: Current epoch number
        args: Training arguments
        checkpoint_manager: Optional checkpoint manager
        global_step: Current global step (for scheduling)
        total_training_steps: Total steps for the full training run
        ss_horizon_start: Starting prediction horizon (1 = next token)
        ss_horizon_end: Ending prediction horizon (e.g., 10 = predict 10 tokens)
        max_grad_norm: Maximum gradient norm for clipping (default 1.0, 0 to disable)
        stage1_mode_initialized: Whether Stage 1 training mode has been initialized

    Returns:
        Dict with training metrics, updated global_step, and stage1_mode_initialized
    """
    # Only call set_epoch if shuffling is enabled (not curriculum mode)
    # set_epoch changes the random seed for shuffling; skip when preserving order
    use_curriculum = getattr(args, "curriculum", True) and not getattr(args, "no_curriculum", False)
    if not use_curriculum and getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    show_progress = is_main()
    elm.train()
    total_loss = 0
    total_steps = 0
    ss_samples = 0  # Count samples that used scheduled sampling
    total_samples = 0
    device = next(elm.parameters()).device
    total_steps_per_epoch = len(dataloader)

    if total_training_steps is None:
        total_training_steps = len(dataloader) * getattr(args, "epochs", 1)

    progress = tqdm(
        dataloader,
        desc=f"Training Epoch {epoch}",
        disable=not show_progress,
        leave=False,
    )

    # Track SS steps separately for proper horizon scheduling
    ss_step_count = 0
    current_horizon = 0  # Current prediction horizon for scheduled sampling
    # Track current phase (max stage seen) for LR schedule resets
    # This is different from individual sample task_ids which can be review samples
    current_phase = 0
    # Track per-stage step counts for wandb (stages 1-9)
    task_step_counts = {i: 0 for i in range(1, 10)}
    # Track Stage 1 sub-task losses (task_type_id 10-11: values_to_tokens, tokens_to_values)
    STAGE1_SUBTYPE_NAMES = {10: "val2tok", 11: "tok2val"}
    stage1_subtype_counts = {10: 0, 11: 0}
    # Track Stage 2 sub-task losses (task_type_id 20-21: magnitude, value_cmp)
    STAGE2_SUBTYPE_NAMES = {20: "magnitude", 21: "value_cmp"}
    stage2_subtype_counts = {20: 0, 21: 0}
    # Track Stage 3 sub-task losses (task_type_id 30-31: sequence, halfway)
    STAGE3_SUBTYPE_NAMES = {30: "sequence", 31: "halfway"}
    stage3_subtype_counts = {30: 0, 31: 0}

    # Training mode setup
    # ECG token training is handled by PEFT's trainable_token_indices (set in LoraConfig)
    # Here we just control which LoRA modules are frozen/unfrozen based on stage
    is_distributed = getattr(args, "distributed", False)
    skip_ecg_warmup = getattr(args, "no_ecg_warmup", False)

    # Determine starting stage (if --stage is specified, we know what phase we're in)
    starting_stage = getattr(args, "stage", None)
    should_start_with_frozen_lora = (starting_stage is None or starting_stage == 1)

    # Initialize training mode on first epoch
    if not stage1_mode_initialized and not skip_ecg_warmup:
        # Set initial training mode based on starting stage
        # Stage 1: ECG tokens (direct) + embedding LoRA (attention/MLP LoRA frozen)
        # Stages 2+: ECG tokens (direct) + all LoRA
        set_ecg_embedding_only_mode(elm, should_start_with_frozen_lora, is_distributed)
        stage1_mode_initialized = True
        if is_main():
            if should_start_with_frozen_lora:
                print(f"[Initial] Training mode: ECG tokens + embedding LoRA (attention/MLP LoRA frozen)")
            else:
                print(f"[Initial] Training mode: ECG tokens + all LoRA")

    for step, batch in enumerate(progress):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Detect phase transitions and reset LR schedule (curriculum mode only)
        # Phase = max task_id seen so far. Review samples have lower task_ids, so we
        # only trigger a phase transition when we see a NEW higher task_id.
        batch_task_ids = batch.get("task_id", None)
        phase_just_changed = False
        if batch_task_ids is not None and use_curriculum:
            batch_max_task = batch_task_ids[:, 0].max().item()  # Max task in batch
            if batch_max_task > current_phase:
                if current_phase >= 0:  # Not the first phase
                    # Reset LR schedule for new phase
                    if hasattr(optimizer, "reset_schedule"):
                        optimizer.reset_schedule()
                        if is_main():
                            print(f"\n[Phase {current_phase} → {batch_max_task}] Resetting LR schedule (warmup restart)")
                current_phase = batch_max_task
                phase_just_changed = True

                # Toggle training mode based on stage
                # Stage 1: ECG tokens + embedding LoRA (attention/MLP LoRA frozen)
                # Stages 2+: ECG tokens + all LoRA
                if not skip_ecg_warmup:
                    should_freeze_nonembedding_lora = (current_phase == 1)
                    set_ecg_embedding_only_mode(elm, should_freeze_nonembedding_lora, is_distributed)
                    if is_main():
                        if should_freeze_nonembedding_lora:
                            print(f"[Stage {current_phase}] Training: ECG tokens + embedding LoRA (attention/MLP LoRA frozen)")
                        else:
                            print(f"[Stage {current_phase}] Training: ECG tokens + all LoRA")

        # Scheduled sampling is disabled for now (too slow due to autoregressive generation)
        # Tasks 5 & 6 will use standard teacher forcing like all other tasks
        applied_ss = False

        total_samples += batch["elm_input_ids"].shape[0]

        # Standard training step
        optimizer.zero_grad()
        outputs = elm(batch)
        loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1
        loss.backward()

        # Gradient clipping to prevent explosion
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(elm.parameters(), max_grad_norm)

        optimizer.step_and_update_lr()

        # Logging
        current_step = global_step + step
        if getattr(args, "wandb", False) and is_main():
            # Global metrics
            log_dict = {
                "train/step_loss": loss.item(),
                "epoch": epoch,
                "global_step": current_step,
                "train/phase": current_phase,
                "train/lr": optimizer.learning_rate if hasattr(optimizer, "learning_rate") else 0,
            }
            if applied_ss:
                log_dict["train/ss_horizon"] = current_horizon
                log_dict["train/ss_step"] = ss_step_count

            # Per-stage metrics: only log when batch is pure (all samples same stage)
            # This avoids mixing review samples into the wrong stage graph
            if batch_task_ids is not None:
                batch_stages = batch_task_ids[:, 0].tolist()  # Get all stage IDs in batch
                unique_stages = set(batch_stages)
                if len(unique_stages) == 1:  # Pure batch - all same stage
                    batch_stage = batch_stages[0]
                    if 1 <= batch_stage <= 9:
                        task_step_counts[batch_stage] += 1
                        log_dict[f"stage{batch_stage}/step"] = task_step_counts[batch_stage]
                        log_dict[f"stage{batch_stage}/loss"] = loss.item()
                        log_dict[f"stage{batch_stage}/lr"] = optimizer.learning_rate if hasattr(optimizer, "learning_rate") else 0

                    # For Stage 1, log sub-task type (val2tok, tok2val)
                    if batch_stage == 1:
                        batch_task_type_ids = batch.get("task_type_id", None)
                        if batch_task_type_ids is not None:
                            task_type_id = batch_task_type_ids[0, 0].item()
                            if task_type_id in STAGE1_SUBTYPE_NAMES:
                                subtype_name = STAGE1_SUBTYPE_NAMES[task_type_id]
                                stage1_subtype_counts[task_type_id] += 1
                                log_dict[f"stage1_{subtype_name}/step"] = stage1_subtype_counts[task_type_id]
                                log_dict[f"stage1_{subtype_name}/loss"] = loss.item()

                    # For Stage 2, log sub-task type (magnitude, value_cmp)
                    if batch_stage == 2:
                        batch_task_type_ids = batch.get("task_type_id", None)
                        if batch_task_type_ids is not None:
                            task_type_id = batch_task_type_ids[0, 0].item()
                            if task_type_id in STAGE2_SUBTYPE_NAMES:
                                subtype_name = STAGE2_SUBTYPE_NAMES[task_type_id]
                                stage2_subtype_counts[task_type_id] += 1
                                log_dict[f"stage2_{subtype_name}/step"] = stage2_subtype_counts[task_type_id]
                                log_dict[f"stage2_{subtype_name}/loss"] = loss.item()

                    # For Stage 3, log sub-task type (sequence, halfway)
                    if batch_stage == 3:
                        batch_task_type_ids = batch.get("task_type_id", None)
                        if batch_task_type_ids is not None:
                            task_type_id = batch_task_type_ids[0, 0].item()
                            if task_type_id in STAGE3_SUBTYPE_NAMES:
                                subtype_name = STAGE3_SUBTYPE_NAMES[task_type_id]
                                stage3_subtype_counts[task_type_id] += 1
                                log_dict[f"stage3_{subtype_name}/step"] = stage3_subtype_counts[task_type_id]
                                log_dict[f"stage3_{subtype_name}/loss"] = loss.item()

                # When phase changes, also log the new phase to ensure it appears in wandb
                # This handles cases where rank 0 might not see the new stage's samples immediately
                elif phase_just_changed and 1 <= current_phase <= 9:
                    task_step_counts[current_phase] += 1
                    log_dict[f"stage{current_phase}/step"] = task_step_counts[current_phase]
                    log_dict[f"stage{current_phase}/loss"] = loss.item()
                    log_dict[f"stage{current_phase}/lr"] = optimizer.learning_rate if hasattr(optimizer, "learning_rate") else 0

            wandb.log(log_dict)

        # Update progress bar - only show horizon when SS is actually applied
        postfix = {"loss": f"{loss.item():.4f}", "phase": current_phase}
        if applied_ss:
            postfix["SS"] = current_horizon
        progress.set_postfix(postfix)

        # Checkpointing
        if checkpoint_manager and checkpoint_manager.save_step(step, total_steps_per_epoch):
            checkpoint_manager.save_checkpoint(elm, optimizer, epoch, step, prefix="step_")

        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break

    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    ss_ratio = ss_samples / total_samples if total_samples > 0 else 0

    if is_main():
        print(f"Epoch {epoch}: avg_loss={average_loss:.4f}, ss_ratio={ss_ratio:.2%}, final_phase={current_phase}")
        # Print per-task step counts
        task_summary = ", ".join([f"T{t}:{c}" for t, c in task_step_counts.items() if c > 0])
        print(f"  Task steps: {task_summary}")

    return {
        "average_loss": average_loss,
        "total_steps": total_steps,
        "global_step": global_step + total_steps,
        "ss_ratio": ss_ratio,
        "stage1_mode_initialized": stage1_mode_initialized,  # Pass back across epochs
    }
