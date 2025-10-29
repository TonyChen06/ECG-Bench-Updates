import torch
import wandb
from tqdm import tqdm
from ecg_bench.utils.gpu_setup import is_main, train_dev_break


def train(encoder, dataloader, optimizer, epoch, args, checkpoint_manager=None):
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    show_progress = is_main()
    encoder.train()
    total_loss = 0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Training Encoder; Epoch {epoch}",
        disable=not show_progress,
        leave=False,
    )
    device = next(encoder.parameters()).device
    total_steps_per_epoch = len(dataloader)

    for step, batch in enumerate(progress):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = encoder(batch)
        loss = outputs.loss
        total_loss += loss.item()
        total_steps += 1
        loss.backward()
        optimizer.step_and_update_lr()
        if getattr(args, "wandb", False) and is_main():
            wandb.log({"train/step_loss": loss.item(), "epoch": epoch})
        if checkpoint_manager and checkpoint_manager.save_step(step, total_steps_per_epoch):
            checkpoint_manager.save_checkpoint(encoder, optimizer, epoch, step, prefix="step_")
        if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
            break
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    return {"average_loss": average_loss, "total_steps": total_steps}


def validate(encoder, dataloader, epoch, args):
    show_progress = is_main()
    encoder.eval()
    total_loss = 0
    total_steps = 0
    skipped_batches = 0
    progress = tqdm(
        dataloader,
        desc=f"Validating Encoder; Epoch {epoch}",
        disable=not show_progress,
        leave=False,
    )
    device = next(encoder.parameters()).device

    with torch.no_grad():
        for step, batch in enumerate(progress):
            # Check if batch is empty (collate_fn returned empty tensors when all items were None)
            if "ecg_signal" in batch and batch["ecg_signal"].numel() == 0:
                skipped_batches += 1
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = encoder(batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
            if train_dev_break(getattr(args, "dev", False), batch, loss.item()):
                break
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")

    if is_main():
        print(f"Validation: processed {total_steps} batches, skipped {skipped_batches} empty batches, avg loss: {average_loss:.4f}")

    return {"average_loss": average_loss, "total_steps": total_steps}
