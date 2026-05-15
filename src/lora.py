"""
LoRA / QLoRA wrapping for V-JEPA 2 encoder.

# CRITICAL: this module FAILS LOUDLY if LoRA ends up with zero trainable
# parameters. The most common LoRA bug is target_modules names that don't
# match the actual encoder's parameter names — silent and disastrous.
"""

from typing import Dict, List

import torch.nn as nn
from peft import LoraConfig, get_peft_model


def _list_attention_layer_indices(model: nn.Module) -> List[int]:
    """
    Discover which integer indices correspond to encoder layers.

    HuggingFace ViT-style encoders have parameters like:
        encoder.layer.0.attention.attention.query.weight
        encoder.layer.0.attention.attention.value.weight
        ...
        encoder.layer.N-1.attention.attention.query.weight

    We scan named_parameters to find unique layer indices.
    """
    indices = set()
    for name, _ in model.named_parameters():
        # Look for ".layer.N." patterns
        if ".layer." in name:
            parts = name.split(".")
            try:
                i = parts.index("layer")
                idx = int(parts[i + 1])
                indices.add(idx)
            except (ValueError, IndexError):
                continue
    return sorted(indices)


def apply_lora(model: nn.Module, config: Dict) -> nn.Module:
    """
    Apply LoRA to the V-JEPA 2 encoder.

    Args:
        model:  V-JEPA 2 model (loaded, all params frozen)
        config: experiment config dict (the top-level YAML)

    Returns:
        Same model with LoRA adapters injected.

    # HINT: PEFT's `target_modules` accepts:
    #   - a list of suffix names (matched against the end of each module name)
    #   - a regex string
    #
    # We use the suffix list form for simplicity. e.g. ["query", "value"]
    # matches any module whose full name ends with .query or .value.
    """
    if not config.get("use_lora", False):
        print("[lora] use_lora=False — encoder stays frozen")
        return model

    lora_cfg = config["lora"]
    r = int(lora_cfg["r"])
    alpha = int(lora_cfg.get("alpha", r * 2))
    dropout = float(lora_cfg.get("dropout", 0.1))
    target_modules = list(lora_cfg.get("target_modules", ["query", "value"]))
    bias = str(lora_cfg.get("bias", "none"))
    upper_only = bool(lora_cfg.get("upper_layers_only", False))
    num_upper = int(lora_cfg.get("num_upper_layers", 6))

    # Upper-layers-only: restrict to the last N encoder layers using a regex.
    if upper_only:
        all_layer_idx = _list_attention_layer_indices(model)
        if not all_layer_idx:
            raise RuntimeError(
                "[lora] upper_layers_only=True but no encoder layers detected "
                "via .layer.N. pattern. Run scripts/inspect_model.py."
            )
        total = max(all_layer_idx) + 1
        start = max(0, total - num_upper)
        upper_idx = list(range(start, total))
        # Build a regex that matches ".layer.{i}." for i in upper_idx,
        # followed by anything, ending in any of the target_modules.
        idx_alt = "|".join(str(i) for i in upper_idx)
        mod_alt = "|".join(target_modules)
        target_modules = rf".*\.layer\.({idx_alt})\..*\.({mod_alt})$"
        print(
            f"[lora] upper_layers_only: layers {start}..{total - 1} "
            f"({num_upper}/{total}) | regex={target_modules}"
        )

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
    )

    model = get_peft_model(model, lora_config)

    # CRITICAL CHECK: trainable params MUST be > 0.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * n_train / n_total

    print(f"[lora] r={r} alpha={alpha} dropout={dropout}")
    print(f"[lora] trainable: {n_train / 1e6:.3f}M / {n_total / 1e6:.1f}M ({pct:.3f}%)")

    if n_train == 0:
        # Failed silently — dump the encoder's parameter names so the user
        # can see what to put in target_modules.
        print("[lora] ERROR — zero trainable params! Likely causes:")
        print("  1. target_modules names don't match this model's modules")
        print("  2. Encoder uses a different naming scheme (q_proj, qkv, etc.)")
        print("")
        print("[lora] Sample of encoder parameter names (look for q/k/v):")
        names = [n for n, _ in model.named_parameters()]
        attn_names = [n for n in names if "attn" in n.lower() or "query" in n.lower()
                      or "q_proj" in n or "qkv" in n]
        for n in attn_names[:30]:
            print(f"        {n}")
        raise RuntimeError("LoRA produced zero trainable parameters. See above.")

    return model


def setup_encoder_treatment(model: nn.Module, config: Dict) -> nn.Module:
    """
    Top-level entry point used by train.py.

    Decides: frozen (no LoRA), LoRA, or QLoRA.
    QLoRA quantization is applied at MODEL LOAD time (see src/model.py).
    Here we just attach LoRA adapters on top of the (possibly quantized) model.
    """
    if config.get("use_qlora", False):
        # Quantization already done at load time; LoRA on top works the same.
        # bitsandbytes-compatible PEFT integration is automatic.
        print("[encoder] applying LoRA on top of 4-bit quantized base (QLoRA)")
        return apply_lora(model, config)

    if config.get("use_lora", False):
        print("[encoder] applying standard LoRA")
        return apply_lora(model, config)

    print("[encoder] no adapter — encoder remains fully frozen")
    return model
