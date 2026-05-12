"""
LoRA and QLoRA setup for V-JEPA 2 encoder.

Applies low-rank adapters to attention layers by manually
replacing Linear layers with LoRALinear wrappers.

Does NOT use peft's get_peft_model (which fails on VJEPA2Encoder).
Instead walks the encoder's module tree and injects LoRA directly.

Supports:
  - Full LoRA (all encoder layers)
  - Upper-layer LoRA (last N layers only)
  - QLoRA (4-bit quantized base + LoRA)
"""

import re
import torch
import torch.nn as nn


# ─── LoRA Layer ─────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with a parallel low-rank branch.

    out = W_frozen @ x + (B @ A @ x) * (alpha / r)

    Only A and B are trainable. W_frozen is kept intact.
    B is initialized to zero so LoRA starts as identity.
    """

    def __init__(self, linear, r, alpha, dropout=0.0):
        super().__init__()
        self.linear = linear
        self.r = r
        self.scaling = alpha / r

        in_features = linear.in_features
        out_features = linear.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Init: A ~ kaiming, B = 0 → LoRA output is zero at start
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze original weights
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        base = self.linear(x)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + lora


# ─── Module Injection ───────────────────────────────────────────────

def _find_and_replace(model, target_names, lora_r, lora_alpha,
                      lora_dropout, layer_indices=None):
    """
    Walks the encoder module tree and replaces matching Linear
    layers with LoRALinear.

    Args:
        model:         V-JEPA 2 model
        target_names:  list of module name suffixes to target
                       e.g. ['query', 'value']
        lora_r:        LoRA rank
        lora_alpha:    LoRA alpha scaling
        lora_dropout:  dropout before LoRA branch
        layer_indices: set of layer indices to target, or None for all

    Returns:
        count of replaced modules
    """
    replaced = 0

    # Walk all named modules in the encoder
    for name, module in model.encoder.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check if this module's name ends with a target name
        short_name = name.split('.')[-1]
        if short_name not in target_names:
            continue

        # Check layer index if upper-layers-only mode
        if layer_indices is not None:
            layer_match = re.search(r'layer\.(\d+)', name)
            if layer_match:
                idx = int(layer_match.group(1))
                if idx not in layer_indices:
                    continue

        # Get device of the original module
        device = next(module.parameters()).device

        # Replace the module — move LoRA weights to same device
        lora_module = LoRALinear(
            linear=module,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        ).to(device)

        # Navigate to parent and replace
        parts = name.split('.')
        parent = model.encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora_module)

        replaced += 1

    return replaced


# ─── Public API ─────────────────────────────────────────────────────

def apply_lora(model, config):
    """
    Applies LoRA to V-JEPA 2 encoder attention layers.

    Args:
        model:  V-JEPA 2 model (already frozen)
        config: experiment config dict with lora settings

    Returns:
        model with LoRA applied to encoder
    """
    lora_r = int(config['lora_r'])
    lora_alpha = int(config.get('lora_alpha', lora_r * 2))
    lora_dropout = float(config.get('lora_dropout', 0.1))
    target_modules = config.get('lora_target_modules', ['query', 'value'])

    # Upper layers only mode
    layer_indices = None
    upper_only = config.get('lora_upper_layers_only', False)
    if upper_only:
        num_upper = int(config.get('lora_num_upper_layers', 6))
        total_layers = model.config.num_hidden_layers
        start_layer = total_layers - num_upper
        layer_indices = set(range(start_layer, total_layers))

        print(f"LoRA upper layers only: layers {start_layer}-"
              f"{total_layers - 1} ({num_upper}/{total_layers} layers)")

    count = _find_and_replace(
        model=model,
        target_names=target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        layer_indices=layer_indices,
    )

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"LoRA applied (r={lora_r}, alpha={lora_alpha}):")
    print(f"  Replaced {count} Linear layers with LoRALinear")
    print(f"  Trainable: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M "
          f"({100 * trainable / total:.2f}%)")

    return model


def apply_qlora(model, config):
    """
    Applies QLoRA: 4-bit quantized base model + LoRA adapters.
    Requires bitsandbytes library.

    Note: the model must be loaded with quantization config
    BEFORE calling this. This function handles the LoRA part.

    Args:
        model:  V-JEPA 2 model (loaded with 4-bit quantization)
        config: experiment config dict

    Returns:
        model with QLoRA applied
    """
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        raise ImportError(
            "QLoRA requires bitsandbytes. "
            "Install with: pip install bitsandbytes"
        )

    # QLoRA uses the same LoRA injection on a quantized model
    return apply_lora(model, config)


def setup_lora(model, config):
    """
    Main entry point. Decides which LoRA variant to apply.

    Args:
        model:  V-JEPA 2 model
        config: experiment config dict

    Returns:
        model with appropriate LoRA variant applied
    """
    if not config.get('use_lora', False):
        print("No LoRA — encoder stays frozen")
        return model

    if config.get('use_qlora', False):
        print("Applying QLoRA...")
        return apply_qlora(model, config)
    else:
        print("Applying LoRA...")
        return apply_lora(model, config)
