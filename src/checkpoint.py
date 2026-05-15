"""
Checkpoint save / load / resume.

# Saves everything needed to:
#   1. Resume training after a crash
#   2. Reproduce results (config, seeds, git hash)
#   3. Fill the paper's experiment table (params, memory, time, metrics)
"""

import json
import os
from datetime import datetime
from typing import Dict, Optional

import torch


def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def save_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: Dict,
    epoch: int,
    global_step: int,
    history: list,
    best_action_mR5: float = 0.0,
    peak_gpu_mem_bytes: int = 0,
    total_train_time: float = 0.0,
    git_hash: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> None:
    """
    Save full checkpoint plus a human-readable summary JSON.

    Args:
        save_dir:             directory to save into
        model:                V-JEPA 2 (frozen or LoRA)
        probe:                AttentiveProbe
        optimizer / scheduler: training state
        config:               full experiment config (top-level YAML merged)
        epoch:                last completed epoch
        global_step:          total optimization steps so far
        history:              list of per-epoch dicts (losses, eval, time, mem)
        best_action_mR5:      best action mR@5 seen during this run
        peak_gpu_mem_bytes:   peak CUDA memory across the run
        total_train_time:     wall-clock seconds spent training
        git_hash:             optional, set in run.py
        extra:                anything else worth saving
    """
    os.makedirs(save_dir, exist_ok=True)

    # HINT: for frozen baseline, no encoder state to save (it's untouched).
    # For LoRA, save only the trainable (adapter) params.
    if config.get("use_lora", False):
        model_state = {
            k: v for k, v in model.state_dict().items()
            if any(n == k and p.requires_grad
                   for n, p in model.named_parameters())
        }
    else:
        model_state = None

    checkpoint = {
        "model_state":         model_state,
        "probe_state":         probe.state_dict(),
        "optimizer_state":     optimizer.state_dict(),
        "scheduler_state":     scheduler.state_dict(),
        "config":              config,
        "epoch":               epoch,
        "global_step":         global_step,
        "history":             history,
        "best_action_mR5":     best_action_mR5,
        "peak_gpu_mem_bytes":  peak_gpu_mem_bytes,
        "total_train_time":    total_train_time,
        "git_hash":            git_hash,
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "extra":               extra or {},
    }

    ckpt_path = os.path.join(save_dir, f"epoch{epoch}.pth")
    latest_path = os.path.join(save_dir, "latest.pth")
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)

    # Also save a human-readable summary JSON alongside for quick inspection.
    summary = {
        "experiment_name":    config.get("experiment_name", "unknown"),
        "epoch":              epoch,
        "global_step":        global_step,
        "best_action_mR5":    round(best_action_mR5, 3),
        "peak_gpu_mem_GB":    round(peak_gpu_mem_bytes / (1024 ** 3), 2),
        "total_train_time":   _format_time(total_train_time),
        "timestamp":          checkpoint["timestamp"],
        "git_hash":           git_hash,
    }
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ckpt] saved → {ckpt_path}")
    print(f"[ckpt]   best action mR@5: {best_action_mR5:.2f}%  "
          f"| peak GPU: {summary['peak_gpu_mem_GB']:.2f}GB  "
          f"| time: {summary['total_train_time']}")


def load_checkpoint(
    save_dir: str,
    model: torch.nn.Module,
    probe: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Load latest checkpoint if it exists. Returns None if not found.
    """
    latest_path = os.path.join(save_dir, "latest.pth")
    if not os.path.exists(latest_path):
        print("[ckpt] no checkpoint found, starting fresh")
        return None

    print(f"[ckpt] loading from {latest_path}")
    ckpt = torch.load(latest_path, map_location=device)

    # Model (LoRA weights only; missing keys for frozen base are expected)
    if ckpt.get("model_state") is not None:
        msg = model.load_state_dict(ckpt["model_state"], strict=False)
        if msg.missing_keys:
            # HINT: massive number of missing keys = frozen-base params that
            # weren't saved. That's normal for LoRA. Don't alarm.
            print(f"[ckpt] missing_keys (likely frozen base): {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"[ckpt] WARN unexpected_keys: {msg.unexpected_keys[:5]}…")

    probe.load_state_dict(ckpt["probe_state"])

    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])

    print(
        f"[ckpt] resumed: epoch {ckpt['epoch']}, step {ckpt['global_step']}, "
        f"best mR@5 {ckpt.get('best_action_mR5', 0):.2f}%, "
        f"saved {ckpt.get('timestamp', 'unknown')}"
    )
    return ckpt
