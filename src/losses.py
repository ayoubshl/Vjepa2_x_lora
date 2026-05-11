"""
Loss functions for 3-head action anticipation.

Three CrossEntropyLoss instances:
  - verb:   no masking, all samples contribute
  - noun:   no masking, all samples contribute
  - action: ignore_index=-1, unseen (verb,noun) pairs skipped
"""

import torch.nn as nn


class MultiHeadLoss(nn.Module):
    """
    Weighted sum of verb + noun + action losses.

    Args:
        w_verb:   weight for verb CE loss
        w_noun:   weight for noun CE loss
        w_action: weight for action CE loss
    """

    def __init__(self, w_verb=1.0, w_noun=1.0, w_action=1.0):
        super().__init__()

        self.verb_criterion = nn.CrossEntropyLoss()
        self.noun_criterion = nn.CrossEntropyLoss()
        self.action_criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.w_verb = w_verb
        self.w_noun = w_noun
        self.w_action = w_action

    def forward(self, verb_logits, noun_logits, action_logits,
                verb_labels, noun_labels, action_labels):
        """
        Args:
            verb_logits:    [B, 97]
            noun_logits:    [B, 300]
            action_logits:  [B, N]
            verb_labels:    [B] long (0-96)
            noun_labels:    [B] long (0-299)
            action_labels:  [B] long (0 to N-1 or -1)

        Returns:
            total_loss:  scalar
            loss_dict:   individual losses for logging
        """
        verb_loss = self.verb_criterion(verb_logits, verb_labels)
        noun_loss = self.noun_criterion(noun_logits, noun_labels)
        action_loss = self.action_criterion(action_logits, action_labels)

        total = (self.w_verb * verb_loss
                 + self.w_noun * noun_loss
                 + self.w_action * action_loss)

        loss_dict = {
            'verb_loss': verb_loss.item(),
            'noun_loss': noun_loss.item(),
            'action_loss': action_loss.item(),
            'total_loss': total.item(),
        }

        return total, loss_dict


def build_loss(config):
    """
    Builds MultiHeadLoss from config.

    Args:
        config: experiment config dict

    Returns:
        MultiHeadLoss instance
    """
    loss = MultiHeadLoss(
        w_verb=float(config.get('w_verb', 1.0)),
        w_noun=float(config.get('w_noun', 1.0)),
        w_action=float(config.get('w_action', 1.0)),
    )

    print(f"Loss weights: verb={loss.w_verb}, "
          f"noun={loss.w_noun}, action={loss.w_action}")

    return loss
