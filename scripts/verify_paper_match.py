"""
Verify your data loading + metric implementation against the released
V-JEPA 2 EK-100 attentive probe.

# This is your single most important sanity check. If, after loading
# Meta's released probe and running it on your validation set with YOUR
# data pipeline, you get close to the paper's 32.7 action mR@5 for ViT-L,
# then your data loading, frame sampling, feature extraction, and metric
# are all correct.
#
# If you get a number very far from 32.7%, something in your pipeline is
# wrong. Debug that before training your own probes.
#
# Note: the released probe is on Meta's CDN at
#   https://dl.fbaipublicfiles.com/vjepa2/evals/
# Look for the EK-100 ViT-L probe filename. The demo notebook references
# SSv2 explicitly; EK-100 may be named like "ek100-vitl-256-32x?.pt".
#
# Steps you need to do manually:
#   1. Download the ViT-L EK-100 probe weights from Meta's CDN
#   2. Put the .pt file somewhere accessible
#   3. Edit PROBE_CKPT_PATH below
#   4. Run this script
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

from src.model import load_vjepa2, extract_features, get_feature_dims
from src.probe import build_probe
from src.vocabulary import load_action_vocabulary, build_action_vocabulary
from src.dataset import build_dataloader
from src.evaluate import evaluate


PROBE_CKPT_PATH = os.environ.get(
    "VJEPA2_EK100_PROBE",
    "/path/to/ek100-vitl-256.pt",   # EDIT or set env var
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--probe-ckpt", default=PROBE_CKPT_PATH)
    args = parser.parse_args()

    with open(args.global_config) as f:
        gcfg = yaml.safe_load(f)

    paths = {k: os.path.expanduser(v) for k, v in gcfg["paths"].items()}
    dataset_cfg = gcfg["dataset"]
    device = torch.device(gcfg["runtime"]["device"] if torch.cuda.is_available() else "cpu")

    # 1. Load model
    model, processor = load_vjepa2(gcfg["model"]["hf_repo"], device, use_qlora=False)
    enc_dim, pred_dim = get_feature_dims(model)

    # 2. Load or build vocabulary
    if os.path.exists(paths["vocabulary_path"]):
        action_to_id, num_actions = load_action_vocabulary(paths["vocabulary_path"])
    else:
        action_to_id, num_actions = build_action_vocabulary(
            train_csv=paths["train_csv"],
            save_path=paths["vocabulary_path"],
        )

    # 3. Build probe with paper's architecture
    probe = build_probe(
        encoder_dim=enc_dim,
        predictor_dim=pred_dim,
        num_action_classes=num_actions,
        num_verb_classes=int(dataset_cfg["num_verb_classes"]),
        num_noun_classes=int(dataset_cfg["num_noun_classes"]),
        depth=4,
        num_heads=16,
        mlp_ratio=4.0,
        dropout=0.0,
    ).to(device)

    # 4. Load released probe weights
    if not os.path.exists(args.probe_ckpt):
        print(f"ERROR: probe checkpoint not found at {args.probe_ckpt}")
        print("Download it from https://dl.fbaipublicfiles.com/vjepa2/evals/")
        print("and set --probe-ckpt or VJEPA2_EK100_PROBE env var.")
        sys.exit(1)

    raw = torch.load(args.probe_ckpt, map_location="cpu", weights_only=True)
    # The released format wraps weights in dicts. Common structure:
    # raw["classifiers"][0] is the state_dict.
    # HINT: print(raw.keys()) and inspect if loading fails.
    if isinstance(raw, dict) and "classifiers" in raw:
        state = raw["classifiers"][0]
    else:
        state = raw

    # Strip DDP prefixes if any
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # CAVEAT: our probe architecture may not match key-for-key with Meta's
    # released AttentiveClassifier. If load_state_dict fails with mismatched
    # keys, you'll need to either:
    #   (a) write a key-remapping function from their state_dict to ours, OR
    #   (b) port their AttentiveClassifier verbatim into our probe.py
    #
    # This script will print the mismatch so you can decide.
    msg = probe.load_state_dict(state, strict=False)
    print(f"[verify] missing keys ({len(msg.missing_keys)}):")
    for k in msg.missing_keys[:10]:
        print(f"  - {k}")
    if len(msg.missing_keys) > 10:
        print(f"  ... and {len(msg.missing_keys) - 10} more")
    print(f"[verify] unexpected keys ({len(msg.unexpected_keys)}):")
    for k in msg.unexpected_keys[:10]:
        print(f"  - {k}")

    if msg.missing_keys or msg.unexpected_keys:
        print(
            "\n[verify] WARNING: our probe architecture does NOT match the "
            "released probe key-for-key. Either remap keys or port their "
            "AttentiveClassifier code. The eval below will use our "
            "(partially loaded) probe and will NOT match the paper number."
        )

    # 5. Validation dataloader (full official val set, optionally subset)
    participants = None  # use all from val CSV; if you only have a subset on
                         # disk, this would still work if missing-video clips
                         # are filtered out
    val_loader = build_dataloader(
        csv_path=paths["val_csv"],
        videos_dir=paths["videos_dir"],
        action_to_id=action_to_id,
        participants=participants,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=gcfg["runtime"]["num_workers"],
        fps_source=dataset_cfg["fps_source"],
        fps_target=dataset_cfg["fps_target"],
        num_frames=dataset_cfg["num_frames"],
        anticipation_s=dataset_cfg["anticipation_seconds"],
        split="validation",
        cache_dir=paths["annotation_cache_dir"],
    )

    # 6. Run evaluation
    results = evaluate(
        model=model,
        probe=probe,
        dataloader=val_loader,
        device=device,
        use_bf16=True,
        save_predictions_path=os.path.join(paths["predictions_dir"], "verify",
                                            "released_probe.pt"),
        log_prefix="verify",
    )

    print("\n" + "=" * 70)
    print("PAPER COMPARISON (V-JEPA 2 ViT-L on EK-100 from Table 5)")
    print("=" * 70)
    print(f"  Paper verb mR@5:   57.8 %   |   Yours: {results['verb_mR5']:.2f} %")
    print(f"  Paper noun mR@5:   53.8 %   |   Yours: {results['noun_mR5']:.2f} %")
    print(f"  Paper action mR@5: 32.7 %   |   Yours: {results['action_mR5']:.2f} %")
    print("=" * 70)
    if abs(results["action_mR5"] - 32.7) < 2.0:
        print("[verify] ✓ within 2% of paper — pipeline is correct")
    else:
        print("[verify] ✗ off by >2% — investigate before training your own probes")


if __name__ == "__main__":
    main()
