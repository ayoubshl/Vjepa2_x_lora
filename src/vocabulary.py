"""
Action vocabulary.

# Action classes = unique (verb_class, noun_class) pairs from the FULL
# official EK-100 training CSV. Built once, saved as JSON, reused for
# every experiment.
#
# HINT: build from the full train CSV, not from your subset. Otherwise
# your action class space shrinks and you can't compare across experiments
# trained on different subsets. The (verb, noun) → id mapping must be
# stable across all experiments forever.
#
# Validation clips whose (verb, noun) pair was never seen in the full
# train set get action_class = -1, and are excluded from Action mR@5.
"""

import json
from typing import Dict, Tuple

import pandas as pd


def build_action_vocabulary(
    train_csv: str,
    save_path: str = None,
) -> Tuple[Dict[str, int], int]:
    """
    Build (verb, noun) → id mapping from the FULL EK-100 training CSV.

    Args:
        train_csv: path to EPIC_100_train.csv (full, no participant filter)
        save_path: if given, save the JSON here

    Returns:
        action_to_id: dict {"(verb,noun)" : id}
        num_actions:  number of unique pairs
    """
    df = pd.read_csv(train_csv)

    pairs = (
        df[["verb_class", "noun_class"]]
        .drop_duplicates()
        .sort_values(["verb_class", "noun_class"])
        .reset_index(drop=True)
    )

    # JSON cannot use tuple keys → encode as "(v,n)" strings.
    action_to_id = {
        f"({int(v)},{int(n)})": i for i, (v, n) in enumerate(pairs.values)
    }
    num_actions = len(action_to_id)

    print(
        f"[vocab] Built action vocabulary: {num_actions} unique (verb,noun) pairs "
        f"from {len(df)} training clips"
    )

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(action_to_id, f, indent=2)
        print(f"[vocab] Saved to {save_path}")

    return action_to_id, num_actions


def load_action_vocabulary(vocab_path: str) -> Tuple[Dict[str, int], int]:
    """Load a previously saved vocabulary JSON."""
    with open(vocab_path, "r") as f:
        action_to_id = json.load(f)
    num_actions = len(action_to_id)
    print(f"[vocab] Loaded {num_actions} action classes from {vocab_path}")
    return action_to_id, num_actions


def get_action_id(
    verb_class: int,
    noun_class: int,
    action_to_id: Dict[str, int],
) -> int:
    """Look up action id; return -1 if pair never seen in training."""
    key = f"({int(verb_class)},{int(noun_class)})"
    return action_to_id.get(key, -1)
