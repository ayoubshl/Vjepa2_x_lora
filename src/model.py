"""
V-JEPA 2 model loading and feature extraction.

# CRITICAL FIX vs old code: encoder + predictor are concatenated along
# the TOKEN dimension (dim=1), NOT the feature dimension (dim=-1).
# The paper says "concatenated along the token dimension" and the demo
# notebook builds AttentiveClassifier(embed_dim=model.embed_dim), which
# matches the encoder dim (not 2× it).
"""

from typing import Optional, Tuple

import torch
from transformers import AutoModel, AutoVideoProcessor


def load_vjepa2(
    hf_repo: str,
    device: torch.device,
    use_qlora: bool = False,
) -> Tuple[torch.nn.Module, "AutoVideoProcessor"]:
    """
    Load V-JEPA 2 model and processor from HuggingFace.

    All parameters frozen by default. LoRA application happens separately
    in src/lora.py.

    Args:
        hf_repo:    e.g. "facebook/vjepa2-vitl-fpc64-256"
        device:     target device (ignored if use_qlora=True; bnb handles it)
        use_qlora:  if True, load with 4-bit quantization via bitsandbytes

    Returns:
        model:     V-JEPA 2 model, all params frozen, on `device`
        processor: AutoVideoProcessor
    """
    print(f"[model] Loading V-JEPA 2 from {hf_repo} (use_qlora={use_qlora})")

    if use_qlora:
        # HINT: QLoRA requires bitsandbytes. The 4-bit quantization config
        # must be applied at load time, before LoRA wrapping.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError(
                "QLoRA requires `transformers` with bitsandbytes. "
                "`pip install bitsandbytes accelerate`"
            ) from e

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            hf_repo,
            quantization_config=bnb_config,
            device_map="auto",  # bnb places quantized layers
        )
    else:
        model = AutoModel.from_pretrained(hf_repo)
        model = model.to(device)

    processor = AutoVideoProcessor.from_pretrained(hf_repo)

    # Freeze everything; LoRA / probe will mark their own params trainable.
    for p in model.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in model.parameters())
    print(f"[model] Loaded: {total / 1e6:.1f}M params (all frozen)")
    print(f"[model]   encoder hidden_size   = {model.config.hidden_size}")
    print(f"[model]   predictor hidden_size = {model.config.pred_hidden_size}")
    return model, processor


def extract_features(
    model: torch.nn.Module,
    pixel_values_videos: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward through encoder + predictor; return BOTH outputs.

    The probe (see src/probe.py) handles:
      1. Projecting predictor features to encoder dim (if needed)
      2. Concatenating along the TOKEN dim
      3. Pooling with query tokens

    Paper Section 6: "The outputs of the predictor and encoder are
    concatenated along the token dimension and fed to an attentive probe."

    Input:
        pixel_values_videos: [B, T, C, H, W] (processor output, T=32)

    Output:
        encoder_features:   [B, N_enc, D_enc]
        predictor_features: [B, N_pred, D_pred]

    For vitl-fpc64-256 with T=32:
        encoder out:    [B, N_enc, 1024]
        predictor out:  [B, N_pred, D_pred]   — D_pred may differ from D_enc

    # CAVEAT: the V-JEPA 2 predictor has its OWN hidden_size (e.g. 384 by
    # default per HF config), separate from the encoder's 1024.
    # If D_pred != D_enc we cannot directly concatenate along the token
    # dimension — they'd have different feature widths.
    #
    # Two ways to handle this:
    #   (a) project predictor output to D_enc with a small linear layer
    #       (this becomes another trainable parameter in the probe)
    #   (b) only use encoder features, ignore predictor (loses the
    #       "anticipation" signal from the predictor)
    #
    # The official V-JEPA 2 EK-100 probe was trained with the paper's
    # internal codebase where encoder and predictor shared dimension. The
    # HF model has a separate `pred_hidden_size` (often 384 vs 1024).
    #
    # We default to (a): the probe owns a `predictor_proj` linear layer
    # that maps predictor_dim → encoder_dim before concatenation.
    """
    outputs = model(pixel_values_videos=pixel_values_videos)
    encoder_features = outputs.last_hidden_state                          # [B, N_e, D_e]
    predictor_features = outputs.predictor_output.last_hidden_state        # [B, N_p, D_p]
    return encoder_features, predictor_features


def get_feature_dims(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Return (encoder_hidden_size, predictor_hidden_size).

    The probe needs both to construct its predictor-to-encoder projection.
    """
    return int(model.config.hidden_size), int(model.config.pred_hidden_size)
