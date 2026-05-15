"""
Main training loop.

# Single training run over a fixed participant subset. No participant
# streaming. Same data, same probe, same loss across all experiments —
# only the encoder treatment changes (frozen / LoRA / QLoRA).
#
# Pipeline:
#   1. Load config, set seed, init wandb
#   2. Build vocabulary (or load if cached)
#   3. Build dataloaders for train and validation
#   4. Load V-JEPA 2 (frozen by default; QLoRA quantizes here)
#   5. Apply LoRA if configured (paper claim: trainable param count > 0)
#   6. Build probe (paper-matched: 4 blocks, 16 heads, 3 query tokens)
#   7. Build optimizer + scheduler + loss
#   8. Train for N epochs; checkpoint each epoch
#   9. Final mean-class R@5 evaluation on the fixed validation subset
"""

import os
import subprocess
import time
from typing import Dict, List

import torch
import yaml
from tqdm import tqdm

from src.seed import set_seed
from src.vocabulary import build_action_vocabulary, load_action_vocabulary
from src.dataset import build_dataloader
from src.model import load_vjepa2, extract_features, get_feature_dims
from src.probe import build_probe
from src.lora import setup_encoder_treatment
from src.losses import build_loss
from src.optimizer import build_optimizer, build_scheduler
from src.monitor import CollapseMonitor
from src.evaluate import evaluate
from src.checkpoint import save_checkpoint, load_checkpoint
from src import logger


def _git_hash() -> str:
    """Best-effort current git hash for reproducibility."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _read_participants_file(path: str) -> List[str]:
    """Read participants.txt — one ID per line. Empty / commented lines ok."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"participants_file not found: {path}\n"
            f"Create this file with one participant ID per line (e.g. P01)."
        )
    parts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts.append(line)
    return parts


def train(global_config: Dict, experiment_config: Dict) -> None:
    """
    Main training entry point.

    Args:
        global_config:     parsed configs/global.yaml
        experiment_config: parsed configs/<experiment>.yaml
    """
    paths = {k: os.path.expanduser(v) for k, v in global_config["paths"].items()}
    runtime = global_config["runtime"]
    dataset_cfg = global_config["dataset"]
    train_cfg = experiment_config["training"]
    probe_cfg = experiment_config["probe"]

    exp_name = experiment_config["experiment_name"]
    checkpoints_dir = os.path.join(paths["checkpoints_dir"], exp_name)
    predictions_dir = os.path.join(paths["predictions_dir"], exp_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # ----- 1. Seed + wandb -----
    seed = int(runtime.get("seed", 42))
    deterministic = bool(runtime.get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    logger.init_wandb(global_config, experiment_config)
    git_hash = _git_hash()
    print(f"[run] git_hash={git_hash} seed={seed} deterministic={deterministic}")

    device = torch.device(
        runtime["device"] if torch.cuda.is_available() else "cpu"
    )
    use_cuda = device.type == "cuda"

    # ----- 2. Vocabulary (built from FULL train CSV, not subset) -----
    if os.path.exists(paths["vocabulary_path"]):
        action_to_id, num_actions = load_action_vocabulary(paths["vocabulary_path"])
    else:
        action_to_id, num_actions = build_action_vocabulary(
            train_csv=paths["train_csv"],
            save_path=paths["vocabulary_path"],
        )

    # ----- 3. Participant subset -----
    participants = _read_participants_file(paths["participants_file"])
    print(f"[run] participants ({len(participants)}): {participants}")
    print(
        "[run] protocol: fixed participant subset from participants.txt; "
        "do not compare directly to full EK-100 unless this list is complete"
    )

    # ----- 4. Load model (FIRST, before dataloaders, so processor is ready) -----
    use_qlora = bool(experiment_config.get("use_qlora", False))
    model, processor = load_vjepa2(
        global_config["model"]["hf_repo"], device, use_qlora=use_qlora,
    )
    enc_dim, pred_dim = get_feature_dims(model)

    # ----- 5. Encoder treatment (frozen / LoRA / QLoRA) -----
    model = setup_encoder_treatment(model, experiment_config)

    # ----- 6. Probe -----
    probe = build_probe(
        encoder_dim=enc_dim,
        predictor_dim=pred_dim,
        num_action_classes=num_actions,
        num_verb_classes=int(dataset_cfg["num_verb_classes"]),
        num_noun_classes=int(dataset_cfg["num_noun_classes"]),
        depth=int(probe_cfg["depth"]),
        num_heads=int(probe_cfg["num_heads"]),
        mlp_ratio=float(probe_cfg.get("mlp_ratio", 4.0)),
        dropout=float(probe_cfg.get("dropout", 0.0)),
    ).to(device)

    # ----- 7. Dataloaders -----
    train_loader = build_dataloader(
        csv_path=paths["train_csv"],
        videos_dir=paths["videos_dir"],
        action_to_id=action_to_id,
        participants=participants,
        processor=processor,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(runtime["num_workers"]),
        fps_source=int(dataset_cfg["fps_source"]),
        fps_target=int(dataset_cfg["fps_target"]),
        num_frames=int(dataset_cfg["num_frames"]),
        anticipation_s=float(dataset_cfg["anticipation_seconds"]),
        split="train",
        cache_dir=paths["annotation_cache_dir"],
        allow_decode_errors=False,
    )

    # Validation uses the same fixed participant subset because storage limits
    # prevent keeping the full EK-100 validation set locally.
    val_loader = build_dataloader(
        csv_path=paths["val_csv"],
        videos_dir=paths["videos_dir"],
        action_to_id=action_to_id,
        participants=participants,
        processor=processor,
        batch_size=int(train_cfg["batch_size"]),
        num_workers=int(runtime["num_workers"]),
        fps_source=int(dataset_cfg["fps_source"]),
        fps_target=int(dataset_cfg["fps_target"]),
        num_frames=int(dataset_cfg["num_frames"]),
        anticipation_s=float(dataset_cfg["anticipation_seconds"]),
        split="validation",
        cache_dir=paths["annotation_cache_dir"],
        allow_decode_errors=False,
    )

    # ----- 8. Optimizer + scheduler + loss -----
    optimizer, trainable_params = build_optimizer(model, probe, experiment_config)
    scheduler = build_scheduler(optimizer, experiment_config, len(train_loader))
    loss_fn = build_loss(experiment_config["loss"])
    monitor = CollapseMonitor(log_every=50)

    use_bf16 = bool(train_cfg.get("use_bf16", True))
    max_grad_norm = float(experiment_config["optimizer"].get("max_grad_norm", 1.0))
    num_epochs = int(train_cfg["num_epochs"])
    eval_mid = bool(train_cfg.get("eval_mid_training", False))
    eval_at_end = bool(train_cfg.get("eval_at_end", True))

    # ----- 9. Resume if checkpoint exists -----
    ckpt = load_checkpoint(checkpoints_dir, model, probe, optimizer, scheduler, device=device)
    start_epoch = 1
    global_step = 0
    history: List[Dict] = []
    best_action_mR5 = 0.0
    total_train_time = 0.0
    if ckpt is not None:
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        history = ckpt.get("history", [])
        best_action_mR5 = ckpt.get("best_action_mR5", 0.0)
        total_train_time = ckpt.get("total_train_time", 0.0)
        print(f"[run] resumed: starting at epoch {start_epoch}")

    # ----- 10. Training loop -----
    print(f"\n{'=' * 70}\nTRAINING: {exp_name}\n{'=' * 70}")
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    overall_start = time.time()

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        probe.train()

        epoch_loss = 0.0
        epoch_v = 0.0
        epoch_n = 0.0
        epoch_a = 0.0
        num_batches = 0
        epoch_start = time.time()

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs}")
        for batch in pbar:
            frames = batch["frames"].to(device, non_blocking=True)
            v_lab = batch["verb_label"].to(device, non_blocking=True)
            n_lab = batch["noun_label"].to(device, non_blocking=True)
            a_lab = batch["action_label"].to(device, non_blocking=True)

            with torch.amp.autocast(
                device.type if use_cuda else "cpu",
                dtype=torch.bfloat16,
                enabled=bool(use_bf16 and use_cuda),
            ):
                enc_feat, pred_feat = extract_features(model, frames)
                v_log, n_log, a_log = probe(enc_feat, pred_feat)
                total_loss, loss_dict = loss_fn(
                    v_log, n_log, a_log, v_lab, n_lab, a_lab,
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=max_grad_norm
            ).item()

            optimizer.step()
            scheduler.step()

            # Collapse monitor (mean-pool encoder features over tokens)
            with torch.no_grad():
                features_for_monitor = enc_feat.detach().float().mean(dim=1)
            collapse_metrics = monitor.update(features_for_monitor)

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            logger.log_step(
                loss_dict=loss_dict,
                lr=current_lr,
                global_step=global_step,
                grad_norm=grad_norm,
                collapse_metrics=collapse_metrics,
            )

            epoch_loss += loss_dict["total_loss"]
            epoch_v += loss_dict["verb_loss"]
            epoch_n += loss_dict["noun_loss"]
            epoch_a += loss_dict["action_loss"]
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss_dict['total_loss']:.3f}"})

        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time
        avg_loss = epoch_loss / max(1, num_batches)
        peak_mem = torch.cuda.max_memory_allocated(device) if use_cuda else 0
        print(
            f"[epoch {epoch}] avg_loss={avg_loss:.4f}  "
            f"v={epoch_v / max(1, num_batches):.4f}  "
            f"n={epoch_n / max(1, num_batches):.4f}  "
            f"a={epoch_a / max(1, num_batches):.4f}  "
            f"time={epoch_time:.0f}s  peak_mem={peak_mem / 1e9:.2f}GB"
        )
        logger.log_epoch(
            epoch=epoch,
            avg_loss=avg_loss,
            epoch_time_seconds=epoch_time,
            peak_gpu_mem_bytes=peak_mem,
            global_step=global_step,
        )

        # Mid-training eval (optional, at midpoint and end)
        results = None
        is_midpoint = eval_mid and epoch == max(1, num_epochs // 2)
        is_end = epoch == num_epochs
        if is_midpoint or (is_end and eval_at_end):
            print(f"\n[epoch {epoch}] running validation…")
            save_path = os.path.join(predictions_dir, f"epoch{epoch}.pt")
            results = evaluate(
                model=model,
                probe=probe,
                dataloader=val_loader,
                device=device,
                use_bf16=use_bf16,
                save_predictions_path=save_path,
                log_prefix=f"val_e{epoch}",
            )
            logger.log_eval(results, global_step=global_step, prefix="val")
            if results["action_mR5"] > best_action_mR5:
                best_action_mR5 = results["action_mR5"]
                print(f"[epoch {epoch}] NEW BEST action mR@5: {best_action_mR5:.2f}%")

        # Per-epoch history entry
        history_entry = {
            "epoch":            epoch,
            "avg_loss":         avg_loss,
            "avg_verb_loss":    epoch_v / max(1, num_batches),
            "avg_noun_loss":    epoch_n / max(1, num_batches),
            "avg_action_loss":  epoch_a / max(1, num_batches),
            "lr":               current_lr,
            "epoch_time":       epoch_time,
            "peak_mem_bytes":   peak_mem,
            "num_clips":        len(train_loader.dataset),
        }
        if results is not None:
            history_entry["verb_mR5"]   = results["verb_mR5"]
            history_entry["noun_mR5"]   = results["noun_mR5"]
            history_entry["action_mR5"] = results["action_mR5"]
        history.append(history_entry)

        # Checkpoint at end of every epoch
        save_checkpoint(
            save_dir=checkpoints_dir,
            model=model,
            probe=probe,
            optimizer=optimizer,
            scheduler=scheduler,
            config=experiment_config,
            epoch=epoch,
            global_step=global_step,
            history=history,
            best_action_mR5=best_action_mR5,
            peak_gpu_mem_bytes=peak_mem,
            total_train_time=total_train_time,
            git_hash=git_hash,
            extra={"participants": participants, "seed": seed},
        )

    # ----- 11. Done -----
    total_wall = time.time() - overall_start
    print(f"\n{'=' * 70}")
    print(f"DONE: {exp_name}")
    print(f"  best action mR@5  : {best_action_mR5:.2f}%")
    print(f"  total train time  : {total_train_time:.0f}s ({total_train_time / 60:.1f}m)")
    print(f"  wall-clock total  : {total_wall:.0f}s ({total_wall / 60:.1f}m)")
    peak_gpu = torch.cuda.max_memory_allocated(device) if use_cuda else 0
    print(f"  peak GPU memory   : {peak_gpu / 1e9:.2f}GB")
    print(f"{'=' * 70}\n")
    logger.finish()
