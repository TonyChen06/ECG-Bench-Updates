import torch
import os
from ecg_bench.utils.gpu_setup import is_main


class CheckpointManager:
    def __init__(self, run_dir, args):
        self.run_dir = run_dir
        self.args = args
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.best_loss = float("inf")
        self.epoch_losses = []
        if is_main():
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step, is_best=False, prefix=""):
        if not is_main():
            return

        # Only save if it's the best model
        if not is_best:
            return

        # Handle DDP-wrapped models
        if self.args.distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
        }

        # Only save best.pt (overwrites previous best)
        best_path = os.path.join(self.checkpoint_dir, f"{prefix}best.pt")
        torch.save(checkpoint, best_path)

    def save_epoch(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.epoch_losses.append(loss)
            return True
        self.epoch_losses.append(loss)
        return False

    def save_step(self, step, total_steps_per_epoch):
        # Disabled: don't save checkpoints during steps
        return False

    def stop_early(self):
        if len(self.epoch_losses) < self.args.patience + 1:
            return False
        best_loss = min(self.epoch_losses[: -self.args.patience])
        current_loss = self.epoch_losses[-1]
        return current_loss > best_loss + self.args.patience_delta
