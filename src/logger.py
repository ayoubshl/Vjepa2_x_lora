"""
Logging — wandb initialization and structured logging.

All wandb calls go through this module so the rest of
the codebase never imports wandb directly.
"""

import wandb


def init_wandb(global_config, experiment_config):
    """
    Initializes a wandb run.

    Args:
        global_config:      global.yaml contents
        experiment_config:  experiment yaml contents (baseline, lora, etc.)

    Returns:
        wandb run object
    """
    run = wandb.init(
        project=global_config['wandb']['project'],
        entity=global_config['wandb'].get('entity'),
        name=experiment_config['experiment_name'],
        config=experiment_config,
        resume="allow",
    )

    print(f"wandb initialized: {run.url}")
    return run


def log_step(loss_dict, lr, global_step, collapse_metrics=None):
    """
    Logs per-batch metrics.

    Args:
        loss_dict:         dict from MultiHeadLoss
        lr:                current learning rate
        global_step:       total steps so far
        collapse_metrics:  dict from CollapseMonitor or None
    """
    log = {
        'train/total_loss': loss_dict['total_loss'],
        'train/verb_loss': loss_dict['verb_loss'],
        'train/noun_loss': loss_dict['noun_loss'],
        'train/action_loss': loss_dict['action_loss'],
        'train/lr': lr,
    }

    if collapse_metrics is not None:
        log.update(collapse_metrics)

    wandb.log(log, step=global_step)


def log_epoch(participant_id, epoch, avg_loss, global_step):
    """
    Logs end-of-epoch summary.

    Args:
        participant_id:  current participant
        epoch:           epoch number within participant
        avg_loss:        average loss over the epoch
        global_step:     total steps so far
    """
    wandb.log({
        'train/epoch_avg_loss': avg_loss,
        'train/participant': participant_id,
        'train/epoch': epoch,
    }, step=global_step)


def log_eval(results, participant_id, global_step):
    """
    Logs evaluation results.

    Args:
        results:         dict from evaluate()
        participant_id:  current participant
        global_step:     total steps so far
    """
    wandb.log({
        'val/verb_r5': results['verb_r5'],
        'val/noun_r5': results['noun_r5'],
        'val/action_r5': results['action_r5'],
        'val/action_n_valid': results['action_n_valid'],
        'val/action_n_total': results['action_n_total'],
        'val/participant': participant_id,
    }, step=global_step)


def log_participant_summary(participant_id, results, best_action_r5,
                            participants_done, total_participants,
                            global_step):
    """
    Logs summary when a participant finishes training.

    Args:
        participant_id:      completed participant
        results:             final eval results for this participant
        best_action_r5:      best Action R@5 across all participants
        participants_done:   number of participants completed
        total_participants:  total number of participants
        global_step:         total steps so far
    """
    wandb.log({
        'progress/participant': participant_id,
        'progress/participants_done': participants_done,
        'progress/total_participants': total_participants,
        'progress/best_action_r5': best_action_r5,
    }, step=global_step)


def finish_wandb():
    """Closes the wandb run."""
    wandb.finish()
    print("wandb run finished")