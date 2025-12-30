import wandb


def setup_wandb(args):
    print("Initializing Wandb")
    wandb.init(
        project="AHRI",
        config=args,
    )


def setup_pretrain_metrics():
    """
    Define per-task metrics for pretraining curriculum.

    This creates separate x-axes for each task so their losses
    are plotted independently in wandb.
    """
    # Define per-task step counters as x-axes
    for task_id in range(9):
        wandb.define_metric(f"task{task_id}/step")
        wandb.define_metric(f"task{task_id}/*", step_metric=f"task{task_id}/step")

    # Also define global metrics
    wandb.define_metric("global_step")
    wandb.define_metric("train/*", step_metric="global_step")


def cleanup_wandb():
    wandb.finish()
