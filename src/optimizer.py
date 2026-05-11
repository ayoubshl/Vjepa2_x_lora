"""
Optimizer and learning rate scheduler.

# AdamW + linear warmup then cosine decay.
# Continuous across all participants. Warmup happens once
# at the start, not per participant.
"""

import math
import torch


def build_optimizer(model, probe, config):
    """
    Builds AdamW optimizer for all trainable parameters.

    For frozen encoder: only probe params.
    For LoRA: probe + LoRA adapter params.

    Args:
        model:  V-JEPA 2 model
        probe:  AttentiveProbe
        config: experiment config dict

    Returns:
        optimizer:  AdamW instance
        params:     list of trainable params (for grad clipping)
    """
    params = [p for p in list(model.parameters()) + list(probe.parameters())
              if p.requires_grad]

    n_trainable = sum(p.numel() for p in params)
    print(f"Trainable parameters: {n_trainable / 1e6:.2f}M")

    lr = float(config['lr'])
    weight_decay = float(config.get('weight_decay', 0.05))

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay
    )

    return optimizer, params


def build_scheduler(optimizer, config, steps_per_epoch, num_participants):
    """
    Builds a warmup + cosine decay scheduler.

    Total steps = steps_per_epoch × epochs_per_participant × num_participants
    Warmup is a fraction of the total, happens once at the start.

    Args:
        optimizer:          AdamW optimizer
        config:             experiment config dict
        steps_per_epoch:    batches per epoch (estimated from first participant)
        num_participants:   total number of participants

    Returns:
        scheduler: LambdaLR instance
    """
    epochs_per_participant = int(config['epochs_per_participant'])
    warmup_steps = int(config.get('warmup_steps', 500))
    total_steps = steps_per_epoch * epochs_per_participant * num_participants

    print(f"Scheduler: {warmup_steps} warmup → cosine decay "
          f"over {total_steps} total steps")

    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return scheduler
