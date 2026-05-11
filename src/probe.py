"""
Attentive Probe — 3-head classification on top of V-JEPA 2 features.

Following V-JEPA 2 paper protocol:
  - TransformerEncoder processes patch tokens
  - Mean pool over token dimension
  - Three parallel Linear heads: verb, noun, action

Input:  [B, num_patches, feature_dim]  (e.g. [B, 8192, 2048])
Output: (verb_logits, noun_logits, action_logits)
"""

import torch.nn as nn


class AttentiveProbe(nn.Module):

    def __init__(self, input_dim, num_verb_classes, num_noun_classes,
                 num_action_classes, num_heads=8, num_layers=2,
                 dropout=0.1):
        super().__init__()

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        self.norm = nn.LayerNorm(input_dim)

        self.verb_head = nn.Linear(input_dim, num_verb_classes)
        self.noun_head = nn.Linear(input_dim, num_noun_classes)
        self.action_head = nn.Linear(input_dim, num_action_classes)

    def forward(self, x):
        # x: [B, num_patches, input_dim]
        x = self.transformer(x)       # [B, num_patches, input_dim]
        x = self.norm(x)              # [B, num_patches, input_dim]
        x = x.mean(dim=1)             # [B, input_dim]

        verb_logits = self.verb_head(x)      # [B, num_verb_classes]
        noun_logits = self.noun_head(x)      # [B, num_noun_classes]
        action_logits = self.action_head(x)  # [B, num_action_classes]

        return verb_logits, noun_logits, action_logits


def build_probe(feature_dim, num_action_classes,
                num_verb_classes=97, num_noun_classes=300,
                num_layers=2, num_heads=8, dropout=0.1):
    """
    Builds the 3-head probe.

    Args:
        feature_dim:        input dimension (e.g. 2048)
        num_action_classes: from vocabulary (N unique pairs)
        num_verb_classes:   97 for EK-100
        num_noun_classes:   300 for EK-100
        num_layers:         transformer encoder layers
        num_heads:          attention heads
        dropout:            dropout rate

    Returns:
        AttentiveProbe instance
    """
    if num_action_classes is None:
        raise ValueError(
            "num_action_classes required. "
            "Call build/load_action_vocabulary() first."
        )

    probe = AttentiveProbe(
        input_dim=feature_dim,
        num_verb_classes=num_verb_classes,
        num_noun_classes=num_noun_classes,
        num_action_classes=num_action_classes,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )

    n_params = sum(p.numel() for p in probe.parameters())
    print(f"Probe: {n_params / 1e6:.2f}M params "
          f"(in={feature_dim}, layers={num_layers}, heads={num_heads})")
    print(f"  verb_head:   {feature_dim} -> {num_verb_classes}")
    print(f"  noun_head:   {feature_dim} -> {num_noun_classes}")
    print(f"  action_head: {feature_dim} -> {num_action_classes}")

    return probe