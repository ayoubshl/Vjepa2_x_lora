"""
Inspect the V-JEPA 2 model: print named parameters, shapes, and the
exact attention projection module names so you can put them in your
LoRA config.

# Run this ONCE before anything else:
#   python scripts/inspect_model.py
#
# Look at the output and verify your lora_*.yaml `target_modules` matches
# what shows up in the "ATTENTION PROJECTIONS" section. If it doesn't,
# LoRA will silently apply to zero parameters.
"""

import argparse
import os
import sys

# Make `src` importable when this script is run from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import yaml

from src.model import load_vjepa2, get_feature_dims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default="configs/global.yaml")
    args = parser.parse_args()

    with open(args.global_config) as f:
        gcfg = yaml.safe_load(f)

    device = torch.device(gcfg["runtime"]["device"]
                          if torch.cuda.is_available() else "cpu")
    model, processor = load_vjepa2(gcfg["model"]["hf_repo"], device, use_qlora=False)
    enc_dim, pred_dim = get_feature_dims(model)

    print("\n" + "=" * 70)
    print("CONFIG")
    print("=" * 70)
    print(f"hidden_size      (encoder) = {enc_dim}")
    print(f"pred_hidden_size (pred)    = {pred_dim}")
    print(f"num_hidden_layers          = {model.config.num_hidden_layers}")
    print(f"num_attention_heads        = {model.config.num_attention_heads}")
    try:
        print(f"pred_num_hidden_layers     = {model.config.pred_num_hidden_layers}")
        print(f"pred_num_attention_heads   = {model.config.pred_num_attention_heads}")
    except AttributeError:
        pass

    print("\n" + "=" * 70)
    print("ATTENTION PROJECTIONS (USE THESE NAMES IN LORA target_modules)")
    print("=" * 70)
    name_set = set()
    for name, _ in model.named_modules():
        # Common attention projection name patterns
        if any(name.endswith(s) for s in
               (".query", ".key", ".value", ".q_proj", ".k_proj", ".v_proj",
                ".qkv", ".to_q", ".to_k", ".to_v")):
            suffix = name.rsplit(".", 1)[-1]
            name_set.add(suffix)
            print(f"  {name}")

    print("\n" + "=" * 70)
    print(f"UNIQUE PROJECTION SUFFIXES: {sorted(name_set)}")
    print("=" * 70)
    print("Put these (or a subset like ['query','value']) in lora.target_modules")

    print("\n" + "=" * 70)
    print("ALL NAMED PARAMETERS (first 60, last 20)")
    print("=" * 70)
    names = [(n, tuple(p.shape)) for n, p in model.named_parameters()]
    for n, s in names[:60]:
        print(f"  {n}  {s}")
    if len(names) > 80:
        print(f"  ... [{len(names) - 80} more] ...")
    for n, s in names[-20:]:
        print(f"  {n}  {s}")

    print("\n" + "=" * 70)
    print("FORWARD PASS SHAPE CHECK")
    print("=" * 70)
    print("Running a single dummy forward to check output shapes…")
    model.eval()

    # Dummy input matching the expected shape: [B=1, T=32, C=3, H=256, W=256]
    dummy = torch.zeros(1, 32, 3, 256, 256, dtype=torch.float32, device=device)

    # Some processors normalize, but raw zeros are fine for shape check.
    with torch.no_grad():
        out = model(pixel_values_videos=dummy)
    enc = out.last_hidden_state
    pred = out.predictor_output.last_hidden_state
    print(f"  encoder.last_hidden_state          : {tuple(enc.shape)}")
    print(f"  predictor_output.last_hidden_state : {tuple(pred.shape)}")
    print()
    print("Token-dim concat will produce:")
    print(f"  shape = [B, {enc.shape[1] + pred.shape[1]}, D]")
    print(f"  encoder dim = {enc.shape[-1]}, predictor dim = {pred.shape[-1]}")
    if enc.shape[-1] != pred.shape[-1]:
        print(f"  → probe will project predictor {pred.shape[-1]} → {enc.shape[-1]}")


if __name__ == "__main__":
    main()
