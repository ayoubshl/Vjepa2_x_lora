
"""
Per-participant training loop.

Trains for N epochs on one participant's data.
Model, optimizer, scheduler carry over between participants.
"""

import os
import time
import torch
import pandas as pd
from tqdm import tqdm

from src.model import load_model, extract_features, get_feature_dim
from src.probe import build_probe
from src.lora import setup_lora
from src.losses import build_loss
from src.optimizer import build_optimizer, build_scheduler
from src.monitor import CollapseMonitor
from src.evaluate import evaluate
from src.checkpoint import save_checkpoint, load_checkpoint
from src.dataset import build_dataloader
from src.vocabulary import load_action_vocabulary
from src import logger


# Module-level cache so we don't reload the model every participant
_cached_model = None
_cached_probe = None
_cached_optimizer = None
_cached_scheduler = None
_cached_loss_fn = None
_cached_monitor = None
_cached_processor = None
_cached_global_step = 0
_cached_best_action_r5 = 0.0
_cached_history = {}
_cached_participants_trained = []
_cached_total_train_time = 0.0
_cached_initialized = False


def _count_participants(train_csv):
    """Counts unique training participants without touching pipeline state."""
    df = pd.read_csv(train_csv, usecols=['participant_id'])
    return df['participant_id'].nunique()


def _initialize(global_config, experiment_config, action_to_id,
                num_actions):
    """
    One-time initialization: loads model, builds probe, optimizer,
    scheduler, loss, and restores from checkpoint if available.
    Called once on the first participant, then cached.
    """
    global _cached_model, _cached_probe, _cached_optimizer
    global _cached_scheduler, _cached_loss_fn, _cached_monitor
    global _cached_processor, _cached_global_step
    global _cached_best_action_r5, _cached_history
    global _cached_participants_trained, _cached_total_train_time
    global _cached_initialized

    if _cached_initialized:
        return

    paths = global_config['paths']
    pipeline = global_config['pipeline']
    dataset_cfg = global_config['dataset']
    device = torch.device(pipeline['device'])

    # Initialize wandb
    logger.init_wandb(global_config, experiment_config)

    # Load model
    hf_repo = global_config['model']['hf_repo']
    model, processor = load_model(hf_repo, device)

    # Apply LoRA if configured
    model = setup_lora(model, experiment_config)

    # Build probe
    feature_dim = get_feature_dim(model)
    probe = build_probe(
        feature_dim=feature_dim,
        num_action_classes=num_actions,
        num_verb_classes=int(dataset_cfg['num_verb_classes']),
        num_noun_classes=int(dataset_cfg['num_noun_classes']),
        num_layers=int(experiment_config.get('probe_layers', 2)),
        num_heads=int(experiment_config.get('probe_heads', 8)),
        dropout=float(experiment_config.get('probe_dropout', 0.1)),
    ).to(device)

    # Build optimizer
    optimizer, trainable_params = build_optimizer(
        model, probe, experiment_config
    )

    # Estimate scheduler steps
    # We use a rough estimate here — exact count comes from dataloader
    # but we need the scheduler before building the first dataloader
    num_participants = _count_participants(
        os.path.expanduser(paths['train_csv'])
    )

    # Rough estimate: ~1000 clips per participant, / batch_size
    batch_size = int(experiment_config['batch_size'])
    estimated_steps_per_epoch = 1000 // batch_size
    scheduler = build_scheduler(
        optimizer, experiment_config,
        estimated_steps_per_epoch, num_participants
    )

    # Build loss
    loss_fn = build_loss(experiment_config)

    # Monitor
    monitor = CollapseMonitor(log_every=50)

    # Try to load checkpoint
    checkpoints_dir = os.path.expanduser(
        os.path.join(paths['checkpoints_dir'],
                     experiment_config['experiment_name'])
    )
    ckpt_info = load_checkpoint(
        checkpoints_dir, model, probe, optimizer, scheduler, device
    )

    global_step = 0
    best_action_r5 = 0.0
    history = {}
    participants_trained = []
    total_train_time = 0.0

    if ckpt_info is not None:
        global_step = ckpt_info['global_step']
        best_action_r5 = ckpt_info['best_action_r5']
        history = ckpt_info['history']
        participants_trained = ckpt_info['participants_trained']
        total_train_time = ckpt_info['total_train_time']

    # Cache everything
    _cached_model = model
    _cached_probe = probe
    _cached_optimizer = optimizer
    _cached_scheduler = scheduler
    _cached_loss_fn = loss_fn
    _cached_monitor = monitor
    _cached_processor = processor
    _cached_global_step = global_step
    _cached_best_action_r5 = best_action_r5
    _cached_history = history
    _cached_participants_trained = participants_trained
    _cached_total_train_time = total_train_time
    _cached_initialized = True

    print("\nInitialization complete")


def train_on_participant(participant_id, global_config,
                         experiment_config, action_to_id,
                         num_actions):
    """
    Trains on a single participant's data.

    The model, optimizer, and scheduler carry over from
    previous participants — this is a continuous training run.

    Args:
        participant_id:     e.g. 'P01'
        global_config:      parsed global.yaml
        experiment_config:  parsed experiment yaml
        action_to_id:       vocabulary dict
        num_actions:        number of action classes
    """
    global _cached_global_step, _cached_best_action_r5
    global _cached_history, _cached_participants_trained
    global _cached_total_train_time

    # Initialize on first call
    _initialize(global_config, experiment_config, action_to_id,
                num_actions)

    paths = global_config['paths']
    pipeline = global_config['pipeline']
    dataset_cfg = global_config['dataset']
    device = torch.device(pipeline['device'])

    model = _cached_model
    probe = _cached_probe
    optimizer = _cached_optimizer
    scheduler = _cached_scheduler
    loss_fn = _cached_loss_fn
    monitor = _cached_monitor
    processor = _cached_processor

    epochs = int(experiment_config['epochs_per_participant'])
    eval_every = int(experiment_config.get('eval_every_n_epochs', 1))
    max_grad_norm = float(experiment_config.get('max_grad_norm', 1.0))
    use_bf16 = bool(experiment_config.get('use_bf16', True))
    batch_size = int(experiment_config['batch_size'])
    fps = int(dataset_cfg['fps'])
    anticipation_s = float(dataset_cfg['anticipation_seconds'])
    num_frames = int(dataset_cfg['num_frames'])

    checkpoints_dir = os.path.expanduser(
        os.path.join(paths['checkpoints_dir'],
                     experiment_config['experiment_name'])
    )

    # Build train dataloader for this participant only
    train_loader = build_dataloader(
        csv_path=os.path.expanduser(paths['train_csv']),
        frames_dir=os.path.expanduser(paths['frames_dir']),
        action_to_id=action_to_id,
        participants=[participant_id],
        processor=processor,
        batch_size=batch_size,
        fps=fps,
        anticipation_s=anticipation_s,
        num_frames=num_frames,
        split='train',
    )

    # Build val dataloader — uses ALL participants whose frames exist
    val_participants = [
        p for p in os.listdir(os.path.expanduser(paths['frames_dir']))
        if os.path.isdir(os.path.join(
            os.path.expanduser(paths['frames_dir']), p
        ))
    ]
    val_loader = build_dataloader(
        csv_path=os.path.expanduser(paths['val_csv']),
        frames_dir=os.path.expanduser(paths['frames_dir']),
        action_to_id=action_to_id,
        participants=val_participants if val_participants else None,
        processor=processor,
        batch_size=batch_size,
        fps=fps,
        anticipation_s=anticipation_s,
        num_frames=num_frames,
        split='validation',
    )

    # Collect trainable params for grad clipping
    trainable_params = [p for p in list(model.parameters())
                        + list(probe.parameters()) if p.requires_grad]

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=use_bf16)

    # Initialize history for this participant
    if participant_id not in _cached_history:
        _cached_history[participant_id] = {}

    participant_start_time = time.time()

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        probe.train()

        epoch_loss = 0.0
        epoch_verb_loss = 0.0
        epoch_noun_loss = 0.0
        epoch_action_loss = 0.0
        num_batches = 0

        progress = tqdm(
            train_loader,
            desc=f"{participant_id} epoch {epoch}/{epochs}"
        )

        for batch in progress:
            frames = batch['frames'].to(device)
            verb_labels = batch['verb_label'].to(device)
            noun_labels = batch['noun_label'].to(device)
            action_labels = batch['action_label'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                     enabled=use_bf16):
                features = extract_features(model, frames)
                verb_logits, noun_logits, action_logits = probe(features)

                total_loss, loss_dict = loss_fn(
                    verb_logits, noun_logits, action_logits,
                    verb_labels, noun_labels, action_labels
                )

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params,
                                            max_norm=max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Collapse monitoring
            collapse_metrics = monitor.update(
                features.detach().float().mean(dim=1)
            )

            # Logging
            _cached_global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            logger.log_step(loss_dict, current_lr,
                           _cached_global_step, collapse_metrics)

            epoch_loss += loss_dict['total_loss']
            epoch_verb_loss += loss_dict['verb_loss']
            epoch_noun_loss += loss_dict['noun_loss']
            epoch_action_loss += loss_dict['action_loss']
            num_batches += 1

            progress.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        # End of epoch
        avg_loss = epoch_loss / max(1, num_batches)
        avg_verb = epoch_verb_loss / max(1, num_batches)
        avg_noun = epoch_noun_loss / max(1, num_batches)
        avg_action = epoch_action_loss / max(1, num_batches)

        logger.log_epoch(participant_id, epoch, avg_loss,
                        _cached_global_step)

        print(f"{participant_id} epoch {epoch} — "
              f"avg loss: {avg_loss:.4f}")

        # Evaluation
        results = None
        if epoch % eval_every == 0 or epoch == epochs:
            print(f"\n{participant_id} epoch {epoch} evaluation:")
            results = evaluate(model, probe, val_loader, device)
            results['avg_loss'] = avg_loss

            logger.log_eval(results, participant_id,
                           _cached_global_step)

            if results['action_r5'] > _cached_best_action_r5:
                _cached_best_action_r5 = results['action_r5']
                print(f"New best Action R@5: "
                      f"{_cached_best_action_r5:.2f}%")

        # Save epoch history
        _cached_history[participant_id][f'epoch_{epoch}'] = {
            'avg_loss': avg_loss,
            'verb_loss': avg_verb,
            'noun_loss': avg_noun,
            'action_loss': avg_action,
            'lr': current_lr,
            'num_clips': len(train_loader.dataset),
            'verb_r5': results['verb_r5'] if results else None,
            'noun_r5': results['noun_r5'] if results else None,
            'action_r5': results['action_r5'] if results else None,
        }

        # Update total training time
        _cached_total_train_time += time.time() - participant_start_time
        participant_start_time = time.time()

        # Save checkpoint
        save_checkpoint(
            save_dir=checkpoints_dir,
            model=model,
            probe=probe,
            optimizer=optimizer,
            scheduler=scheduler,
            config=experiment_config,
            participant_id=participant_id,
            epoch=epoch,
            global_step=_cached_global_step,
            results=results,
            best_action_r5=_cached_best_action_r5,
            participants_trained=_cached_participants_trained,
            total_train_time=_cached_total_train_time,
            history=_cached_history,
        )

    # Mark participant as trained
    _cached_participants_trained.append(participant_id)

    logger.log_participant_summary(
        participant_id=participant_id,
        results=results,
        best_action_r5=_cached_best_action_r5,
        participants_done=len(_cached_participants_trained),
        total_participants=_count_participants(
            os.path.expanduser(paths['train_csv'])
        ),
        global_step=_cached_global_step,
    )

    print(f"\n{participant_id} training complete "
          f"({epochs} epochs)")
