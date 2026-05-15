"""
Evaluation — mean-class Recall@5, paper-matched protocol.

# CRITICAL FIX vs old code: this is MEAN-CLASS recall, not sample-level.
# For each class c that appears in val, compute R@5 among samples whose
# ground truth is c, then average across classes. This is the official
# EK-100 metric (Damen et al. 2022).
#
# On long-tailed EK-100, sample-level R@5 over-weights frequent classes;
# mean-class R@5 gives each class equal weight. The two can differ by
# 5–15 percentage points. Using the wrong one will get your paper rejected.
#
# We also save raw predictions and clip metadata to disk so subset results
# can be audited and metrics can be recomputed without rerunning inference.
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

# HINT: `extract_features` is imported lazily inside `evaluate()` so this
# module can be used (e.g. for `mean_class_recall_at_k` alone) without
# requiring transformers / HF to be installed.


# ---------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------

@torch.no_grad()
def mean_class_recall_at_k(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int = 5,
    ignore_index: Optional[int] = None,
) -> Dict[str, float]:
    """
    Mean-class Recall@K.

    For each class c that appears in `labels`:
        R_c = (# samples with label c whose top-k contains c) / (# samples with label c)
    Return mean of R_c over classes that appear.

    Args:
        logits:       [N, C] raw logits or probabilities (order doesn't matter for top-k)
        labels:       [N] long
        k:            top-k
        ignore_index: label value to exclude (e.g. -1 for action's unseen pairs)

    Returns:
        dict with:
            mean_class_recall:  float (the headline number, 0..1)
            per_class_recall:   numpy array [num_classes_present]
            classes_evaluated:  numpy array [num_classes_present] of class IDs
            num_samples_used:   int
            num_classes_used:   int
    """
    if ignore_index is not None:
        valid = labels != ignore_index
        logits = logits[valid]
        labels = labels[valid]

    n = labels.numel()
    if n == 0:
        return {
            "mean_class_recall": 0.0,
            "per_class_recall": np.array([]),
            "classes_evaluated": np.array([]),
            "num_samples_used": 0,
            "num_classes_used": 0,
        }

    # Top-k indices per sample, shape [N, k]
    topk = torch.topk(logits, k=k, dim=-1).indices
    # Per-sample correctness: is the true label in the top-k?
    correct = (topk == labels.unsqueeze(1)).any(dim=1)  # [N] bool

    # Group by class
    unique_classes, inverse = labels.unique(return_inverse=True)
    num_classes = unique_classes.numel()

    per_class_recall = torch.zeros(num_classes, dtype=torch.float64)
    per_class_count = torch.zeros(num_classes, dtype=torch.long)

    # Accumulate per-class sums via scatter
    correct_f = correct.to(torch.float64)
    per_class_recall.scatter_add_(0, inverse, correct_f)
    per_class_count.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.long))

    # Avoid div-by-zero (shouldn't happen since we only include classes present)
    per_class_count = per_class_count.clamp(min=1).to(torch.float64)
    per_class_recall = per_class_recall / per_class_count

    return {
        "mean_class_recall": float(per_class_recall.mean()),
        "per_class_recall":  per_class_recall.cpu().numpy(),
        "classes_evaluated": unique_classes.cpu().numpy(),
        "num_samples_used":  int(n),
        "num_classes_used":  int(num_classes),
    }


# ---------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    probe: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_bf16: bool = True,
    save_predictions_path: Optional[str] = None,
    log_prefix: str = "val",
) -> Dict:
    """
    Full eval pass: compute mean-class R@5 for verb, noun, action.

    Args:
        model:                 V-JEPA 2 (frozen / LoRA)
        probe:                 AttentiveProbe
        dataloader:            validation DataLoader
        device:                target device
        use_bf16:              autocast bfloat16 for inference speed
        save_predictions_path: if given, save raw logits, labels, and clip
                               metadata to .pt here
        log_prefix:            for tqdm display

    Returns:
        dict with verb/noun/action mR@5, per-class breakdowns, sample-level
        R@5 (as sanity check), counts.
    """
    # Lazy import — keeps src.evaluate importable without transformers
    from src.model import extract_features

    model.eval()
    probe.eval()

    all_v_logits, all_n_logits, all_a_logits = [], [], []
    all_v_labels, all_n_labels, all_a_labels = [], [], []
    all_participants, all_video_ids, all_start_frames = [], [], []

    autocast_device = device.type if device.type == "cuda" else "cpu"
    autocast_enabled = bool(use_bf16 and device.type == "cuda")

    for batch in tqdm(dataloader, desc=f"{log_prefix} eval"):
        frames = batch["frames"].to(device, non_blocking=True)
        v_label = batch["verb_label"]
        n_label = batch["noun_label"]
        a_label = batch["action_label"]

        with torch.amp.autocast(
            autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            enc, pred = extract_features(model, frames)
            v_log, n_log, a_log = probe(enc, pred)

        # Move to float32 on CPU for accumulation (avoids fp drift)
        all_v_logits.append(v_log.float().cpu())
        all_n_logits.append(n_log.float().cpu())
        all_a_logits.append(a_log.float().cpu())
        all_v_labels.append(v_label.cpu())
        all_n_labels.append(n_label.cpu())
        all_a_labels.append(a_label.cpu())
        all_participants.extend(list(batch["participant_id"]))
        all_video_ids.extend(list(batch["video_id"]))
        all_start_frames.extend([int(x) for x in batch["start_frame"]])

    all_v_logits = torch.cat(all_v_logits)
    all_n_logits = torch.cat(all_n_logits)
    all_a_logits = torch.cat(all_a_logits)
    all_v_labels = torch.cat(all_v_labels)
    all_n_labels = torch.cat(all_n_labels)
    all_a_labels = torch.cat(all_a_labels)

    # Mean-class R@5 (the paper metric)
    verb_mcr = mean_class_recall_at_k(all_v_logits, all_v_labels, k=5)
    noun_mcr = mean_class_recall_at_k(all_n_logits, all_n_labels, k=5)
    action_mcr = mean_class_recall_at_k(all_a_logits, all_a_labels, k=5, ignore_index=-1)

    # Sample-level R@5 (sanity check + comparison with old code)
    def _sample_level_recall(logits, labels, k=5, ignore_index=None):
        if ignore_index is not None:
            valid = labels != ignore_index
            logits = logits[valid]
            labels = labels[valid]
        if labels.numel() == 0:
            return 0.0
        topk = torch.topk(logits, k=k, dim=-1).indices
        return float((topk == labels.unsqueeze(1)).any(dim=1).float().mean())

    verb_sr = _sample_level_recall(all_v_logits, all_v_labels, 5)
    noun_sr = _sample_level_recall(all_n_logits, all_n_labels, 5)
    action_sr = _sample_level_recall(all_a_logits, all_a_labels, 5, ignore_index=-1)

    results = {
        # Headline numbers — paper metric
        "verb_mR5":   verb_mcr["mean_class_recall"] * 100,
        "noun_mR5":   noun_mcr["mean_class_recall"] * 100,
        "action_mR5": action_mcr["mean_class_recall"] * 100,

        # Sample-level numbers (sanity check)
        "verb_sR5":   verb_sr * 100,
        "noun_sR5":   noun_sr * 100,
        "action_sR5": action_sr * 100,

        # Counts
        "verb_n_samples":   verb_mcr["num_samples_used"],
        "verb_n_classes":   verb_mcr["num_classes_used"],
        "noun_n_samples":   noun_mcr["num_samples_used"],
        "noun_n_classes":   noun_mcr["num_classes_used"],
        "action_n_samples": action_mcr["num_samples_used"],
        "action_n_classes": action_mcr["num_classes_used"],

        # Per-class breakdowns (for error analysis in the paper)
        "verb_per_class":   verb_mcr["per_class_recall"],
        "noun_per_class":   noun_mcr["per_class_recall"],
        "action_per_class": action_mcr["per_class_recall"],
        "verb_classes":     verb_mcr["classes_evaluated"],
        "noun_classes":     noun_mcr["classes_evaluated"],
        "action_classes":   action_mcr["classes_evaluated"],
    }

    # Save raw predictions if requested
    # HINT: this is what you should use for your paper's final numbers.
    # Run scripts/compute_metrics_offline.py on these .pt files to recompute
    # metrics without re-running inference.
    if save_predictions_path is not None:
        save_dir = os.path.dirname(save_predictions_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            "verb_logits":   all_v_logits,
            "noun_logits":   all_n_logits,
            "action_logits": all_a_logits,
            "verb_labels":   all_v_labels,
            "noun_labels":   all_n_labels,
            "action_labels": all_a_labels,
            "participant_id": all_participants,
            "video_id": all_video_ids,
            "start_frame": all_start_frames,
        }, save_predictions_path)
        print(f"[eval] Saved raw predictions to {save_predictions_path}")

    # Headline print
    print(f"[{log_prefix}] mean-class R@5  | verb {results['verb_mR5']:.2f}  "
          f"noun {results['noun_mR5']:.2f}  action {results['action_mR5']:.2f}")
    print(f"[{log_prefix}] sample-level R@5| verb {results['verb_sR5']:.2f}  "
          f"noun {results['noun_sR5']:.2f}  action {results['action_sR5']:.2f}")
    print(f"[{log_prefix}] action: {results['action_n_samples']} samples "
          f"across {results['action_n_classes']} classes")

    return results
