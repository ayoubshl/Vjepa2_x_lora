"""
Builds and loads the global action vocabulary.

Action classes = unique (verb_class, noun_class) pairs from the
FULL training set across ALL participants. Built once before any
training starts, saved as JSON, reused for every experiment.

Validation clips whose (verb, noun) pair was never seen in training
get action_class = -1 and are excluded from Action R@5.
"""

import json
import pandas as pd


def build_action_vocabulary(train_csv, save_path=None):
    """
    Reads the full training CSV (all participants) and builds
    a mapping from every unique (verb_class, noun_class) pair
    to a sequential integer ID.

    Args:
        train_csv:  path to EPIC_100_train.csv
        save_path:  if provided, saves vocabulary as JSON

    Returns:
        action_to_id : dict {str "(v,n)" : int id}
        num_actions  : total number of unique pairs
    """
    df = pd.read_csv(train_csv)

    pairs = df[['verb_class', 'noun_class']].drop_duplicates()
    pairs = pairs.sort_values(['verb_class', 'noun_class'])
    pairs = pairs.reset_index(drop=True)

    # Keys are strings "(v,n)" because JSON doesn't support tuple keys
    action_to_id = {
        f"({int(v)},{int(n)})": i
        for i, (v, n) in enumerate(pairs.values)
    }
    num_actions = len(action_to_id)

    print(f"Action vocabulary: {num_actions} unique (verb, noun) pairs "
          f"from {len(df)} training clips")

    if save_path is not None:
        with open(save_path, 'w') as f:
            json.dump(action_to_id, f, indent=2)
        print(f"Vocabulary saved to {save_path}")

    return action_to_id, num_actions


def load_action_vocabulary(vocab_path):
    """
    Loads a previously saved vocabulary JSON.

    Returns:
        action_to_id : dict {str "(v,n)" : int id}
        num_actions  : total number of unique pairs
    """
    with open(vocab_path, 'r') as f:
        action_to_id = json.load(f)

    num_actions = len(action_to_id)
    print(f"Loaded vocabulary: {num_actions} action classes")

    return action_to_id, num_actions


def get_action_id(verb_class, noun_class, action_to_id):
    """
    Looks up the action ID for a (verb, noun) pair.
    Returns -1 if the pair was never seen in training.
    """
    key = f"({int(verb_class)},{int(noun_class)})"
    return action_to_id.get(key, -1)