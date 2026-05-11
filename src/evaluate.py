"""
Evaluation — Recall@5 for verb, noun, and action.

Following V-JEPA 2 paper protocol:
  - verb R@5:   all clips evaluated
  - noun R@5:   all clips evaluated
  - action R@5: clips with action_label=-1 excluded
"""

import torch
from src.model import extract_features


def recall_at_k(logits, labels, k=5, ignore_index=None):
    """
    Computes Recall@K.

    Args:
        logits:       [N, C] raw logits
        labels:       [N] ground truth
        k:            top-k to consider
        ignore_index: label value to exclude (e.g. -1)

    Returns:
        float between 0 and 1, or 0.0 if no valid samples
    """
    if ignore_index is not None:
        valid = labels != ignore_index
        if valid.sum() == 0:
            return 0.0
        logits = logits[valid]
        labels = labels[valid]

    topk = torch.topk(logits, k=k, dim=-1).indices    # [N, k]
    expanded = labels.unsqueeze(1).expand_as(topk)     # [N, k]
    correct = topk.eq(expanded).any(dim=1)             # [N]

    return correct.float().mean().item()


def evaluate(model, probe, dataloader, device):
    """
    Full evaluation pass on a dataloader.

    Args:
        model:      V-JEPA 2 model
        probe:      AttentiveProbe
        dataloader: validation DataLoader
        device:     torch device

    Returns:
        dict with verb_r5, noun_r5, action_r5 (0-100%),
        action_n_valid, action_n_total
    """
    model.eval()
    probe.eval()

    all_verb_logits = []
    all_noun_logits = []
    all_action_logits = []
    all_verb_labels = []
    all_noun_labels = []
    all_action_labels = []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch['frames'].to(device)
            verb_labels = batch['verb_label'].to(device)
            noun_labels = batch['noun_label'].to(device)
            action_labels = batch['action_label'].to(device)

            features = extract_features(model, frames)
            verb_logits, noun_logits, action_logits = probe(features)

            all_verb_logits.append(verb_logits.cpu())
            all_noun_logits.append(noun_logits.cpu())
            all_action_logits.append(action_logits.cpu())
            all_verb_labels.append(verb_labels.cpu())
            all_noun_labels.append(noun_labels.cpu())
            all_action_labels.append(action_labels.cpu())

    all_verb_logits = torch.cat(all_verb_logits)
    all_noun_logits = torch.cat(all_noun_logits)
    all_action_logits = torch.cat(all_action_logits)
    all_verb_labels = torch.cat(all_verb_labels)
    all_noun_labels = torch.cat(all_noun_labels)
    all_action_labels = torch.cat(all_action_labels)

    verb_r5 = recall_at_k(all_verb_logits, all_verb_labels, k=5)
    noun_r5 = recall_at_k(all_noun_logits, all_noun_labels, k=5)
    action_r5 = recall_at_k(
        all_action_logits, all_action_labels, k=5, ignore_index=-1
    )

    n_total = all_action_labels.size(0)
    n_valid = (all_action_labels != -1).sum().item()

    results = {
        'verb_r5': verb_r5 * 100,
        'noun_r5': noun_r5 * 100,
        'action_r5': action_r5 * 100,
        'action_n_valid': n_valid,
        'action_n_total': n_total,
    }

    print(f"  Verb R@5:   {results['verb_r5']:.2f}%")
    print(f"  Noun R@5:   {results['noun_r5']:.2f}%")
    print(f"  Action R@5: {results['action_r5']:.2f}% "
          f"({n_valid}/{n_total} valid)")

    return results