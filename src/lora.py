"""
LoRA and QLoRA setup for V-JEPA 2 encoder.

Applies low-rank adapters to attention layers.
Supports:
  - Full LoRA (all encoder layers)
  - Upper-layer LoRA (last N layers only)
  - QLoRA (4-bit quantized base + LoRA)
"""

from peft import LoraConfig, get_peft_model


def apply_lora(model, config):
    """
    Wraps the encoder with LoRA adapters.

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
    upper_only = config.get('lora_upper_layers_only', False)
    if upper_only:
        num_upper = int(config.get('lora_num_upper_layers', 6))
        total_layers = model.config.num_hidden_layers
        start_layer = total_layers - num_upper

        # Build layer-specific target patterns
        # Only target layers from start_layer to total_layers-1
        layer_targets = []
        for layer_idx in range(start_layer, total_layers):
            for module in target_modules:
                layer_targets.append(f"layer.{layer_idx}.*.{module}")

        target_modules = layer_targets
        print(f"LoRA upper layers only: layers {start_layer}-{total_layers - 1} "
              f"({num_upper}/{total_layers} layers)")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    model.encoder = get_peft_model(model.encoder, lora_config)

    print(f"LoRA applied (r={lora_r}, alpha={lora_alpha}):")
    model.encoder.print_trainable_parameters()

    # Gradient checkpointing to save VRAM
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

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

    # QLoRA uses the same LoRA application on a quantized model
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
