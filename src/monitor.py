"""
Collapse Monitor — detects latent space collapse during training.

# Tracks three signals every N steps on the encoder's mean-pooled features:
#   1. Embedding variance     (healthy: high; collapse: < 0.01)
#   2. Mean cosine similarity (healthy: low; collapse: > 0.99)
#   3. Effective rank         (healthy: ~ feature_dim; collapse: tiny)
#
# This is your novel-finding angle: no one has applied LoRA to a JEPA model
# before, so it's unknown whether fine-tuning adapters causes the
# self-supervised latent space to degrade. If LoRA at high ranks collapses
# but low ranks don't, that's a result. If nothing collapses, also a result.
"""

from typing import Dict, Optional

import numpy as np
import torch


class CollapseMonitor:
    def __init__(self, log_every: int = 50):
        self.log_every = int(log_every)
        self.step = 0

    def update(self, features: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        Args:
            features: [B, D] tensor (mean-pooled over tokens, detached, float32)

        Returns:
            dict of metrics, or None on non-logging steps.
        """
        self.step += 1
        if self.step % self.log_every != 0:
            return None

        with torch.no_grad():
            # 1. Per-feature variance, averaged
            variance = features.var(dim=0).mean().item()

            # 2. Mean pairwise cosine similarity (off-diagonal)
            normed = torch.nn.functional.normalize(features, dim=-1)
            sim = normed @ normed.T
            B = sim.size(0)
            if B > 1:
                mask = ~torch.eye(B, dtype=torch.bool, device=features.device)
                cosine_sim = sim[mask].mean().item()
            else:
                cosine_sim = 0.0

            # 3. Effective rank = exp(entropy of normalized eigenvalues)
            # HINT: eigvalsh on a [D, D] cov matrix (D=1024) is ~few ms.
            # If D grows much larger and this slows training, downsample
            # the batch or compute every 200 steps instead.
            try:
                cov = torch.cov(features.T)
                eig = torch.linalg.eigvalsh(cov).abs().clamp(min=1e-8)
                probs = eig / eig.sum()
                entropy = -(probs * probs.log()).sum().item()
                effective_rank = float(np.exp(entropy))
            except Exception:
                effective_rank = float("nan")

        return {
            "collapse/variance":       variance,
            "collapse/cosine_sim":     cosine_sim,
            "collapse/effective_rank": effective_rank,
        }
