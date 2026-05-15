"""
Wandb logging — all wandb calls go through this module.

# HINT: never call wandb directly from train.py. If wandb is disabled
# (e.g., offline run), this module no-ops gracefully.
"""

from typing import Dict, Optional

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    wandb = None
    _HAS_WANDB = False

_run = None


def init_wandb(global_config: Dict, experiment_config: Dict) -> None:
    """Initialize a wandb run if wandb is available."""
    global _run
    if not _HAS_WANDB:
        print("[wandb] not installed; logs go to stdout only")
        return

    wandb_cfg = global_config.get("wandb", {})
    project = wandb_cfg.get("project", "vjepa2-x-lora")
    entity = wandb_cfg.get("entity") or None

    # Combine configs for traceability — wandb will show all hyperparams.
    full_config = {"global": global_config, "experiment": experiment_config}

    _run = wandb.init(
        project=project,
        entity=entity,
        name=experiment_config.get("experiment_name", "unnamed"),
        config=full_config,
        resume="allow",
    )
    print(f"[wandb] initialized: {_run.url}")


def _log(d: Dict, step: int) -> None:
    if _HAS_WANDB and _run is not None:
        wandb.log(d, step=step)


def log_step(
    loss_dict: Dict,
    lr: float,
    global_step: int,
    grad_norm: Optional[float] = None,
    collapse_metrics: Optional[Dict] = None,
) -> None:
    payload = {
        "train/total_loss":  loss_dict["total_loss"],
        "train/verb_loss":   loss_dict["verb_loss"],
        "train/noun_loss":   loss_dict["noun_loss"],
        "train/action_loss": loss_dict["action_loss"],
        "train/lr":          lr,
    }
    if grad_norm is not None:
        payload["train/grad_norm"] = grad_norm
    if collapse_metrics is not None:
        payload.update(collapse_metrics)
    _log(payload, global_step)


def log_epoch(
    epoch: int,
    avg_loss: float,
    epoch_time_seconds: float,
    peak_gpu_mem_bytes: int,
    global_step: int,
) -> None:
    _log({
        "epoch/avg_loss":      avg_loss,
        "epoch/seconds":       epoch_time_seconds,
        "epoch/peak_mem_GB":   peak_gpu_mem_bytes / (1024 ** 3),
        "epoch":               epoch,
    }, global_step)


def log_eval(results: Dict, global_step: int, prefix: str = "val") -> None:
    """
    Log evaluation results. Only scalars go to wandb; per-class arrays
    are saved to disk by evaluate() itself.
    """
    payload = {}
    for k, v in results.items():
        # Skip arrays
        if hasattr(v, "shape"):
            continue
        payload[f"{prefix}/{k}"] = v
    _log(payload, global_step)


def finish() -> None:
    if _HAS_WANDB and _run is not None:
        wandb.finish()
        print("[wandb] run finished")
