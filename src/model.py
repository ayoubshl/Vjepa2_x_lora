"""
V-JEPA 2 model loading and feature extraction.

Handles loading from HuggingFace, freezing all parameters,
and extracting concatenated encoder + predictor features.
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
    print(f"  predictor hidden_size: {model.config.pred_hidden_size}")

    return model, processor


def extract_features(model, pixel_values):
    """
    Forward pass through encoder and predictor.
    Returns concatenated features.

    Input:  pixel_values  [B, T, C, H, W]
    Output: features      [B, num_patches, hidden_size + pred_hidden_size]

    For vitl-fpc64-256:
      encoder:     [B, 8192, 1024]
      predictor:   [B, 8192, 1024]
      concatenated: [B, 8192, 2048]
    """
    outputs = model(pixel_values_videos=pixel_values)
    encoder_features = outputs.last_hidden_state
    predictor_features = outputs.predictor_output.last_hidden_state

    features = torch.cat(
        [encoder_features, predictor_features], dim=-1
    )
    return features


def get_feature_dim(model):
    """
    Returns the concatenated feature dimension.
    hidden_size + pred_hidden_size (e.g. 1024 + 1024 = 2048)
    """
    return model.config.hidden_size + model.config.pred_hidden_size