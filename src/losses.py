"""
Losses for 3-head action anticipation.

# Paper uses focal loss (Lin et al. 2017) on each head independently,
# then sums them before backprop. We support CE too for ablation.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    Standard form:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the predicted probability of the true class.

    Args:
        gamma:         focusing parameter (paper uses gamma=2)
        alpha:         per-class weight tensor [num_classes] or scalar.
                       None = uniform alpha=1 (most common)
        ignore_index:  label to skip entirely (used for action head -1)
        reduction:     'mean' or 'sum' or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = ignore_index
        self.reduction = reduction
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.register_buffer(
            "alpha", alpha if alpha is not None else torch.tensor(1.0),
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Mask out ignored samples
        valid = target != self.ignore_index
        if valid.sum() == 0:
            # No valid samples → contribute zero loss but keep gradient graph
            return logits.sum() * 0.0
        logits = logits[valid]
        target = target[valid]

        # log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        # Gather the log-prob of the true class per sample
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        # alpha factor (per-class or scalar)
        if self.alpha.numel() == 1:
            alpha_t = self.alpha.to(logits.device)
        else:
            alpha_t = self.alpha.to(logits.device)[target]

        loss = -alpha_t * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------
# Multi-head wrapper
# ---------------------------------------------------------------------

class MultiHeadLoss(nn.Module):
    """
    Weighted sum of verb + noun + action losses.

    Verb and noun losses never ignore samples (all labels are valid 0..N-1).
    Action loss uses ignore_index=-1 for unseen (verb,noun) pairs.
    """

    def __init__(
        self,
        loss_type: str = "focal",
        gamma: float = 2.0,
        w_verb: float = 1.0,
        w_noun: float = 1.0,
        w_action: float = 1.0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.w_verb = float(w_verb)
        self.w_noun = float(w_noun)
        self.w_action = float(w_action)

        if loss_type == "focal":
            # ignore_index for action; verb/noun never have -1
            self.verb_loss = FocalLoss(gamma=gamma, ignore_index=-100)
            self.noun_loss = FocalLoss(gamma=gamma, ignore_index=-100)
            self.action_loss = FocalLoss(gamma=gamma, ignore_index=-1)
        elif loss_type == "ce":
            self.verb_loss = nn.CrossEntropyLoss()
            self.noun_loss = nn.CrossEntropyLoss()
            self.action_loss = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            raise ValueError(f"unknown loss type: {loss_type}")

        print(
            f"[loss] type={loss_type} gamma={gamma if loss_type == 'focal' else 'N/A'} "
            f"weights: verb={self.w_verb} noun={self.w_noun} action={self.w_action}"
        )

    def forward(
        self,
        verb_logits, noun_logits, action_logits,
        verb_labels, noun_labels, action_labels,
    ) -> Tuple[torch.Tensor, Dict]:
        lv = self.verb_loss(verb_logits, verb_labels)
        ln = self.noun_loss(noun_logits, noun_labels)
        la = self.action_loss(action_logits, action_labels)

        total = self.w_verb * lv + self.w_noun * ln + self.w_action * la

        return total, {
            "verb_loss":   float(lv.detach()),
            "noun_loss":   float(ln.detach()),
            "action_loss": float(la.detach()),
            "total_loss":  float(total.detach()),
        }


def build_loss(loss_cfg: Dict) -> MultiHeadLoss:
    return MultiHeadLoss(
        loss_type=str(loss_cfg.get("type", "focal")),
        gamma=float(loss_cfg.get("gamma", 2.0)),
        w_verb=float(loss_cfg.get("w_verb", 1.0)),
        w_noun=float(loss_cfg.get("w_noun", 1.0)),
        w_action=float(loss_cfg.get("w_action", 1.0)),
    )
