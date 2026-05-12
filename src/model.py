"""
V-JEPA 2 model loading and feature extraction.

Handles loading from HuggingFace, freezing all parameters,
and extracting encoder features for downstream tasks.
LoRA wrapping is handled separately in src/lora.py.
"""

import torch
from transformers import AutoModel, AutoVideoProcessor


def load_model(hf_repo, device):
    """
    Loads V-JEPA 2 model and processor from HuggingFace.
    All parameters are frozen by default.
    LoRA is applied separately via src/lora.py.

    Args:
        hf_repo:  HuggingFace repo ID, e.g. "facebook/vjepa2-vitl-fpc64-256"
        device:   torch device

    Returns:
        model:     V-JEPA 2 model (all params frozen)
        processor: AutoVideoProcessor
    """
    print(f"Loading V-JEPA 2 from {hf_repo}...")
    model = AutoModel.from_pretrained(hf_repo)
    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params / 1e6:.1f}M params (all frozen)")
    print(f"  encoder hidden_size: {model.config.hidden_size}")

    return model, processor


def extract_features(model, pixel_values):
    """
    Forward pass through encoder only.
    Predictor is discarded — it's a pretraining artifact.

    Input:  pixel_values  [B, T, C, H, W]
    Output: features      [B, num_patches, hidden_size]

    For vitl-fpc64-256:
      encoder: [B, 8192, 1024]
    """
    outputs = model(pixel_values_videos=pixel_values)
    return outputs.last_hidden_state


def get_feature_dim(model):
    """
    Returns the encoder output dimension.
    """
    return model.config.hidden_size  # 1024
