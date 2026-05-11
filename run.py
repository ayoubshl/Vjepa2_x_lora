"""
Single entry point for the entire pipeline.

"""

import argparse
import os
import yaml

from pipeline.manager import PipelineManager


def load_config(config_path):
    """Loads a YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='V-JEPA 2 + LoRA Sequential Training Pipeline'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to experiment config '
    )
    parser.add_argument(
        '--global-config', type=str, default='configs/global.yaml',
        help='Path to global config '
    )
    args = parser.parse_args()

    # Load configs
    global_config = load_config(args.global_config)
    experiment_config = load_config(args.config)

    print(f"Global config:     {args.global_config}")
    print(f"Experiment config: {args.config}")
    print(f"Experiment name:   {experiment_config['experiment_name']}")

    # Run pipeline
    manager = PipelineManager(global_config, experiment_config)
    manager.run()


if __name__ == '__main__':
    main()