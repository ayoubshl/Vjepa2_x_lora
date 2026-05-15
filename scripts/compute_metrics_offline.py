"""
Recompute metrics from saved logits.

# Used for the paper's final numbers — you saved raw logits during training
# (see evaluate.py), now recompute mean-class R@5 from them. You can also
# format them in EK-100 official JSON for the official evaluation script.
#
# Usage:
#   python scripts/compute_metrics_offline.py predictions/baseline_frozen/final.pt
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

from src.evaluate import mean_class_recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_path", help="path to .pt with logits + labels")
    parser.add_argument("--save-json", default=None,
                        help="optional: save EK-100 official JSON format here")
    args = parser.parse_args()

    if not os.path.exists(args.predictions_path):
        raise FileNotFoundError(args.predictions_path)

    data = torch.load(args.predictions_path, map_location="cpu", weights_only=True)

    verb = mean_class_recall_at_k(data["verb_logits"], data["verb_labels"], k=5)
    noun = mean_class_recall_at_k(data["noun_logits"], data["noun_labels"], k=5)
    action = mean_class_recall_at_k(
        data["action_logits"], data["action_labels"], k=5, ignore_index=-1
    )

    participants = sorted(set(data.get("participant_id", [])))
    if participants:
        print(f"subset participants: {participants}")
        print(f"saved clips: {len(data.get('video_id', []))}")

    print(f"verb   mR@5:  {verb['mean_class_recall'] * 100:.2f}%  "
          f"({verb['num_samples_used']} samples, {verb['num_classes_used']} classes)")
    print(f"noun   mR@5:  {noun['mean_class_recall'] * 100:.2f}%  "
          f"({noun['num_samples_used']} samples, {noun['num_classes_used']} classes)")
    print(f"action mR@5:  {action['mean_class_recall'] * 100:.2f}%  "
          f"({action['num_samples_used']} samples, {action['num_classes_used']} classes)")

    if args.save_json is not None:
        # EK-100 official JSON format for action anticipation:
        # {
        #   "version": "0.2",
        #   "challenge": "action_anticipation",
        #   "results": {
        #       "<clip_id>": {
        #           "verb": {"<class_id>": score, ...},
        #           "noun": {"<class_id>": score, ...},
        #           "action": {"<verb>,<noun>": score, ...}
        #       }
        #   }
        # }
        #
        # HINT: this requires per-clip metadata (clip_id, mapping from
        # action_id back to "verb,noun"). If you want to use the official
        # script, you need to save those too in evaluate.py — currently we
        # only save logits and labels. Extend evaluate.py to also save
        # clip metadata, then implement the JSON formatting here.
        print(f"[offline] EK-100 JSON export not yet implemented")
        print(f"[offline] To enable: extend src/evaluate.py to save clip IDs,")
        print(f"[offline] then implement the formatting here.")


if __name__ == "__main__":
    main()
