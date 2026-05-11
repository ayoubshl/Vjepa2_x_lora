"""
Checkpoint save/load/resume.

Saves everything needed to:
  1. Resume training exactly where it left off
  2. Track full experiment history
  3. Reproduce results
"""

import os
import torch
from datetime import datetime


def save_checkpoint(save_dir, model, probe, optimizer, scheduler,
                    config, participant_id, epoch, global_step,
                    results=None, best_action_r5=0.0,
                    participants_trained=None,
                    total_train_time=0.0,
                    history=None):
    """
    Saves full checkpoint to disk.

    Args:
        save_dir:              checkpoints directory
        model:                 V-JEPA 2 model
        probe:                 AttentiveProbe
        optimizer:             AdamW
        scheduler:             LR scheduler
        config:                experiment config dict
        participant_id:        current participant (e.g. 'P01')
        epoch:                 current epoch within participant
        global_step:           total training steps so far
        results:               eval results dict or None
        best_action_r5:        best Action R@5 seen so far
        participants_trained:  list of completed participants
        total_train_time:      total seconds spent training
        history:               full training history dict
    """
    os.makedirs(save_dir, exist_ok=True)

    # LoRA: save only trainable params. Frozen: nothing to save.
    if config.get('use_lora', False):
        model_state = {
            k: v for k, v in model.state_dict().items()
            if any(p.requires_grad
                   for n, p in model.named_parameters() if n == k)
        }
    else:
        model_state = None

    checkpoint = {
        # Model + training state
        'model_state': model_state,
        'probe_state': probe.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),

        # Progress tracking
        'participant_id': participant_id,
        'epoch': epoch,
        'global_step': global_step,
        'participants_trained': participants_trained or [],

        # Results
        'results': results,
        'best_action_r5': best_action_r5,

        # Full history — every epoch, every participant
        # Structure:
        # {
        #   'P01': {
        #     'epoch_1': {
        #       'avg_loss': 2.31,
        #       'verb_loss': 0.8, 'noun_loss': 0.9, 'action_loss': 0.6,
        #       'verb_r5': 45.2, 'noun_r5': 32.1, 'action_r5': 12.5,
        #       'lr': 5e-5, 'num_clips': 1200
        #     },
        #     'epoch_2': { ... },
        #   },
        #   'P02': { ... },
        # }
        'history': history or {},

        # Metadata
        'config': config,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_train_time': total_train_time,
    }

    # Save numbered + latest
    ckpt_path = os.path.join(
        save_dir, f'{participant_id}_epoch{epoch}.pth'
    )
    latest_path = os.path.join(save_dir, 'latest.pth')

    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)

    time_str = _format_time(total_train_time)
    print(f"Checkpoint saved: {ckpt_path}")
    print(f"  best Action R@5: {best_action_r5:.2f}% | "
          f"trained: {len(participants_trained or [])} participants | "
          f"total time: {time_str}")


def load_checkpoint(save_dir, model, probe, optimizer=None,
                    scheduler=None, device='cpu'):
    """
    Loads latest checkpoint if it exists.

    Args:
        save_dir:   checkpoints directory
        model:      V-JEPA 2 model
        probe:      AttentiveProbe
        optimizer:  AdamW (optional, for resume)
        scheduler:  LR scheduler (optional, for resume)
        device:     torch device

    Returns:
        dict with all checkpoint info, or None if no checkpoint found
    """
    latest_path = os.path.join(save_dir, 'latest.pth')

    if not os.path.exists(latest_path):
        print("No checkpoint found — starting from scratch")
        return None

    print(f"Loading checkpoint from {latest_path}...")
    checkpoint = torch.load(latest_path, map_location=device)

    # Load model state (LoRA weights)
    if checkpoint.get('model_state') is not None:
        model.load_state_dict(checkpoint['model_state'], strict=False)

    # Load probe
    probe.load_state_dict(checkpoint['probe_state'])

    # Load optimizer and scheduler if provided
    if optimizer is not None and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if scheduler is not None and 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])

    info = {
        'participant_id': checkpoint.get('participant_id'),
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'results': checkpoint.get('results'),
        'best_action_r5': checkpoint.get('best_action_r5', 0.0),
        'participants_trained': checkpoint.get('participants_trained', []),
        'total_train_time': checkpoint.get('total_train_time', 0.0),
        'history': checkpoint.get('history', {}),
        'timestamp': checkpoint.get('timestamp', 'unknown'),
    }

    time_str = _format_time(info['total_train_time'])
    print(f"Resumed from {info['participant_id']} epoch {info['epoch']} "
          f"(step {info['global_step']})")
    print(f"  best Action R@5: {info['best_action_r5']:.2f}% | "
          f"trained: {info['participants_trained']} | "
          f"saved at: {info['timestamp']} | "
          f"total time: {time_str}")

    return info


def _format_time(seconds):
    """Formats seconds into human readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"