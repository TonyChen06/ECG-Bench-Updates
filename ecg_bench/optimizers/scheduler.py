import numpy as np
import argparse
from torch.optim import Adam, AdamW

from ecg_bench.utils.gpu_setup import get_world_size, is_main


def get_optimizer(args, model):
    """Build optimizer with learning rate scheduling.

    If embed_skip_warmup is enabled, creates separate param groups for
    embeddings (skip warmup) and other params (normal warmup).
    """
    optimizers = {
        "adam": Adam,
        "adamw": AdamW,
    }
    if args.optimizer.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}. Supported: {list(optimizers.keys())}")
    optimizer_class = optimizers[args.optimizer.lower()]

    embed_skip_warmup = getattr(args, "embed_skip_warmup", False)

    if embed_skip_warmup:
        # Separate embedding params from other params
        embed_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Match embedding layers (embed_tokens, lm_head, wte, wpe, etc.)
            if any(emb_name in name.lower() for emb_name in ['embed', 'wte', 'wpe', 'lm_head']):
                embed_params.append(param)
            else:
                other_params.append(param)

        if is_main():
            print(f"[embed_skip_warmup] Embedding params: {len(embed_params)}, Other params: {len(other_params)}")

        param_groups = [
            {"params": other_params, "skip_warmup": False},
            {"params": embed_params, "skip_warmup": True},
        ]

        optimizer = ScheduledOptim(
            optimizer_class(
                param_groups,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            args,
            embed_skip_warmup=True,
        )
    else:
        optimizer = ScheduledOptim(
            optimizer_class(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(args.beta1, args.beta2),
                eps=args.eps,
                lr=args.lr,
                weight_decay=args.weight_decay,
            ),
            args,
        )
    return optimizer


class ScheduledOptim:
    @staticmethod
    def world_size(args):
        ws = get_world_size()
        gpus = getattr(args, "gpus", None)
        if ws == 1 and getattr(args, "distributed", False) and gpus is not None:
            return len(str(gpus).split(","))
        return ws

    @staticmethod
    def effective_global_bs(args):
        ws = ScheduledOptim.world_size(args)
        return args.batch_size * int(ws) * args.grad_accum_steps

    def __init__(
        self,
        optimizer,
        args: argparse.Namespace,
        embed_skip_warmup: bool = False,
    ):
        self.optimizer = optimizer
        self.n_warmup_steps = args.warmup
        self.n_current_steps = 0
        self.embed_skip_warmup = embed_skip_warmup
        eff_bs = self.effective_global_bs(args)

        if args.ref_global_bs is None:
            args.ref_global_bs = args.batch_size * args.grad_accum_steps
        args.ref_global_bs = max(args.ref_global_bs, 1)
        scale = max(eff_bs / args.ref_global_bs, 1e-8)

        if args.encoder and not args.llm:
            self.peak_lr = float(getattr(args, "lr", 1e-3)) * scale
        else:
            self.peak_lr = float(getattr(args, "lr", 3e-4)) * scale

        self.init_lr = self.peak_lr * (self.n_warmup_steps**0.5 if self.n_warmup_steps > 0 else 1.0)

        wd_scale = 1.0 if args.scale_wd == "none" else (1.0 / (scale**0.5) if args.scale_wd == "inv_sqrt" else 1.0 / scale)

        for g in self.optimizer.param_groups:
            if "weight_decay" in g and g["weight_decay"] is not None:
                g["weight_decay"] = float(g["weight_decay"]) * wd_scale

        if is_main():
            print(
                f"[scale] eff_bs={eff_bs}, ref_global_bs={args.ref_global_bs}, scale={scale:.4g}, wd_mode={args.scale_wd}, init_lr={self.init_lr:.3e}, peak_lr={self.peak_lr:.3e}"
            )
            if embed_skip_warmup:
                print(f"[embed_skip_warmup] Embeddings will use peak_lr={self.peak_lr:.3e} from step 0")

    def step_and_update_lr(self):
        self.update_learning_rate()
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_lr_scale(self):
        step = max(1, self.n_current_steps)
        d_step = 1.0 / np.sqrt(step)
        return min(d_step, step * (self.n_warmup_steps**-1.5)) if self.n_warmup_steps > 0 else d_step

    def update_learning_rate(self):
        warmup_lr = max(self.init_lr * self.get_lr_scale(), 1e-8)

        for g in self.optimizer.param_groups:
            if self.embed_skip_warmup and g.get("skip_warmup", False):
                # Embeddings skip warmup - use peak LR with decay after warmup
                if self.n_current_steps < self.n_warmup_steps:
                    g["lr"] = self.peak_lr
                else:
                    # After warmup, follow the same decay schedule
                    g["lr"] = warmup_lr
            else:
                g["lr"] = warmup_lr

        self.n_current_steps += 1

    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def reset_schedule(self):
        """Reset the step counter to restart warmup from the beginning."""
        self.n_current_steps = 0
