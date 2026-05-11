"""
Collapse Monitor — detects latent space collapse during training.

Tracks three signals every N steps:
  1. Embedding variance     (healthy: high, collapse: <0.01)
  2. Mean cosine similarity (healthy: low, collapse: >0.99)
  3. Effective rank          (healthy: high, collapse: low)
"""

import torch
import numpy as np


class CollapseMonitor:

    def __init__(self, log_every=50):
        self.log_every = log_every
        self.step = 0

    def update(self, features):
        """
        Args:
            features: [B, D] tensor (mean-pooled, detached)

        Returns:
            dict of metrics or None (only returns every log_every steps)
        """
        self.step += 1
        if self.step % self.log_every != 0:
            return None

        with torch.no_grad():
            # 1. Embedding variance
            variance = features.var(dim=0).mean().item()

            # 2. Mean pairwise cosine similarity
            normed = torch.nn.functional.normalize(features, dim=-1)
            sim_matrix = normed @ normed.T
            mask = ~torch.eye(
                sim_matrix.size(0),
                dtype=torch.bool,
                device=features.device
            )
            cosine_sim = sim_matrix[mask].mean().item()

            # 3. Effective rank via eigenvalue entropy
            cov = torch.cov(features.T)
            eigenvalues = torch.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues.abs().clamp(min=1e-8)
            probs = eigenvalues / eigenvalues.sum()
            entropy = -(probs * probs.log()).sum().item()
            effective_rank = np.exp(entropy)

        metrics = {
            'collapse/variance': variance,
            'collapse/cosine_sim': cosine_sim,
            'collapse/effective_rank': effective_rank,
        }

        # Warn if collapse detected (after warmup)
        if self.step > 500:
            if variance < 0.01:
                print(f"WARNING: Low variance {variance:.6f} "
                      f"at step {self.step}")
            if cosine_sim > 0.99:
                print(f"WARNING: High cosine sim {cosine_sim:.4f} "
                      f"at step {self.step}")

        return metrics