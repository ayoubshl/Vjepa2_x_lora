"""
Seeding for reproducibility.

"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the seed across Python random, NumPy, PyTorch, and CUDA.

    Args:
        seed:           the master seed
        deterministic:  if True, makes cuDNN deterministic (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PYTHONHASHSEED for hash-based randomness (dict iteration etc.)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # CUBLAS workspace must be set for fully deterministic matmul
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # PyTorch ≥1.8: also enforces deterministic algorithms
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
