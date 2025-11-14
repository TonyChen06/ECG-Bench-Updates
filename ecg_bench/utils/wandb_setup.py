import wandb


def setup_wandb(args):
    print("Initializing Wandb")
    wandb.init(
        project="ELFtables",
        config=args,
    )


def cleanup_wandb():
    wandb.finish()
