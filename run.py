"""
Single entry point for one experiment.

# Usage:
#   python run.py --config configs/baseline_frozen.yaml
#   python run.py --config configs/lora_r4.yaml
#   python run.py --config configs/lora_r16.yaml --global-config configs/global.yaml
"""

import argparse

import yaml

from train import train


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="V-JEPA 2 × LoRA training (paper-matched protocol)"
    )
    parser.add_argument(
        "--config", required=True,
        help="experiment YAML, e.g. configs/baseline_frozen.yaml",
    )
    parser.add_argument(
        "--global-config", default="configs/global.yaml",
        help="global YAML with paths and dataset constants",
    )
    args = parser.parse_args()

    global_config = load_yaml(args.global_config)
    experiment_config = load_yaml(args.config)

    print(f"[run] global_config:     {args.global_config}")
    print(f"[run] experiment_config: {args.config}")
    print(f"[run] experiment_name:   {experiment_config['experiment_name']}")

    train(global_config, experiment_config)


if __name__ == "__main__":
    main()
