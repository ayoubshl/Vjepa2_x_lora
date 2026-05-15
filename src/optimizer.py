"""
Optimizer and LR scheduler.

# AdamW + linear warmup → cosine decay over the full training run.
# Total steps = len(train_loader) × num_epochs, computed once at start.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    probe: nn.Module,
    config: Dict,
) -> Tuple[torch.optim.Optimizer, List[torch.nn.Parameter]]:
    """
    AdamW over all trainable parameters.

    Args:
        model:  V-JEPA 2 (frozen or LoRA-wrapped)
        probe:  AttentiveProbe (all params trainable)
        config: experiment config

    Returns:
        optimizer
        trainable_params: list (used later for grad clipping)
    """
    opt_cfg = config["optimizer"]
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))

    trainable_params = [
        p for p in list(model.parameters()) + list(probe.parameters())
        if p.requires_grad
    ]
    n = sum(p.numel() for p in trainable_params)
    print(f"[optim] trainable params: {n / 1e6:.3f}M | lr={lr} wd={weight_decay}")

    # HINT: if you want different LR for probe vs LoRA params, split into
    # param groups here. Common practice: probe gets 2× the LoRA LR.
    # Default: same LR for both, simpler and reproducible.
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    return optimizer, trainable_params


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear warmup → cosine decay scheduler.

    Args:
        optimizer:       AdamW
        config:          experiment config
        steps_per_epoch: len(train_loader)

    Returns:
        LambdaLR scheduler
    """
    opt_cfg = config["optimizer"]
    train_cfg = config["training"]

    num_epochs = int(train_cfg["num_epochs"])
    total_steps = steps_per_epoch * num_epochs
    warmup_frac = float(opt_cfg.get("warmup_fraction", 0.05))
    warmup_steps = max(1, int(total_steps * warmup_frac))

    print(
        f"[optim] schedule: {warmup_steps} warmup → cosine decay "
        f"over {total_steps} steps ({num_epochs} epochs × {steps_per_epoch} batches)"
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay from 1.0 to 0.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
