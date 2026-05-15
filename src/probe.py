"""
Paper-matched attentive probe for V-JEPA 2 EK-100 anticipation.

# Architecture (V-JEPA 2 paper Section 6 / Appendix 13.1):
#   - Stack of `depth` transformer encoder blocks (self-attention + MLP)
#     over the (encoder ⊕ predictor) token sequence
#   - Final cross-attention layer: 3 learnable query tokens attend to the
#     refined token sequence; each query token gets its own linear head
#     (verb, noun, action)
#
# This differs from the old code in three important ways:
#   1. Operates on token-dim concat of encoder+predictor (not feature-dim)
#   2. Uses learnable query tokens (not mean pooling)
#   3. Depth = 4, num_heads = 16 (paper hyperparameters)
#
# We also include a learned predictor→encoder projection because the HF
# V-JEPA 2 model has separate hidden sizes for encoder vs predictor.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# Standard transformer encoder block (self-attention + MLP)
# ---------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LN transformer block (LayerNorm before each sublayer)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention over the full token sequence
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------
# Cross-attention pooling: N learned queries attend to the token sequence
# ---------------------------------------------------------------------

class QueryPool(nn.Module):
    """
    Final cross-attention layer with `num_queries` learnable query tokens.

    For EK-100 anticipation: num_queries=3 (one each for verb, noun, action).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_queries: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # HINT: trunc_normal init at std=0.02 is the ViT-family default.
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Per-query MLP refinement
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, D] refined token sequence from the probe blocks

        Returns:
            queries_out: [B, num_queries, D]
        """
        B = tokens.size(0)
        # Expand learned queries to batch
        q = self.queries.expand(B, -1, -1)         # [B, Q, D]
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(tokens)

        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)
        q = q + attn_out
        q = q + self.mlp(self.norm2(q))
        return q                                    # [B, Q, D]


# ---------------------------------------------------------------------
# Full attentive probe
# ---------------------------------------------------------------------

class AttentiveProbe(nn.Module):
    """
    Paper-matched attentive probe for EK-100 anticipation.

    Forward:
        1. Project predictor features to encoder dim (if dims differ)
        2. Concatenate encoder + predictor along the TOKEN dim → [B, N_e+N_p, D]
        3. Pass through `depth` transformer encoder blocks
        4. Pool with 3 query tokens via cross-attention → [B, 3, D]
        5. Apply three independent linear classifiers (verb / noun / action)
           to the corresponding query outputs

    Args:
        encoder_dim:        encoder hidden size (e.g. 1024 for ViT-L)
        predictor_dim:      predictor hidden size from model.config
        num_verb_classes:   97 for EK-100
        num_noun_classes:   300 for EK-100
        num_action_classes: from vocabulary (~3000+)
        depth:              transformer blocks (paper = 4)
        num_heads:          attention heads (paper = 16)
        mlp_ratio:          MLP expansion ratio (paper default 4.0)
        dropout:            dropout in attention + MLP
    """

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        num_verb_classes: int,
        num_noun_classes: int,
        num_action_classes: int,
        depth: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.predictor_dim = predictor_dim

        # HINT: only build the projection if dims actually differ. If they
        # match (rare for HF V-JEPA 2 but possible), the projection is a
        # no-op identity to avoid wasted parameters.
        if predictor_dim != encoder_dim:
            self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)
            # Initialize close to identity-ish: small random init is fine
            # for a learned projection. No special initialization needed.
        else:
            self.predictor_proj = nn.Identity()

        # Probe body: stack of transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=encoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        # Final cross-attention with 3 query tokens
        self.pool = QueryPool(
            dim=encoder_dim,
            num_heads=num_heads,
            num_queries=3,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Three independent classifiers, one per query
        self.verb_head = nn.Linear(encoder_dim, num_verb_classes)
        self.noun_head = nn.Linear(encoder_dim, num_noun_classes)
        self.action_head = nn.Linear(encoder_dim, num_action_classes)

        # HINT: print param count so user can see if probe is reasonable
        n = sum(p.numel() for p in self.parameters())
        print(
            f"[probe] AttentiveProbe: {n / 1e6:.2f}M params | "
            f"depth={depth} heads={num_heads} "
            f"enc_dim={encoder_dim} pred_dim={predictor_dim}"
        )

    def forward(
        self,
        encoder_feats: torch.Tensor,
        predictor_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_feats:   [B, N_e, D_e]
            predictor_feats: [B, N_p, D_p]

        Returns:
            verb_logits:   [B, num_verb_classes]
            noun_logits:   [B, num_noun_classes]
            action_logits: [B, num_action_classes]
        """
        # 1. Project predictor to encoder dim if needed
        predictor_feats = self.predictor_proj(predictor_feats)

        # 2. Concatenate along the TOKEN dimension (paper-matched)
        tokens = torch.cat([encoder_feats, predictor_feats], dim=1)  # [B, N_e+N_p, D]

        # 3. Probe blocks (shared trunk for all three heads)
        for blk in self.blocks:
            tokens = blk(tokens)

        # 4. Query-token pooling
        pooled = self.pool(tokens)  # [B, 3, D]

        verb_q   = pooled[:, 0, :]
        noun_q   = pooled[:, 1, :]
        action_q = pooled[:, 2, :]

        # 5. Per-head linear classifiers
        verb_logits   = self.verb_head(verb_q)
        noun_logits   = self.noun_head(noun_q)
        action_logits = self.action_head(action_q)

        return verb_logits, noun_logits, action_logits


def build_probe(
    encoder_dim: int,
    predictor_dim: int,
    num_action_classes: int,
    num_verb_classes: int = 97,
    num_noun_classes: int = 300,
    depth: int = 4,
    num_heads: int = 16,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
) -> AttentiveProbe:
    """Construct the probe from config + vocab."""
    return AttentiveProbe(
        encoder_dim=encoder_dim,
        predictor_dim=predictor_dim,
        num_verb_classes=num_verb_classes,
        num_noun_classes=num_noun_classes,
        num_action_classes=num_action_classes,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )
