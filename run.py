"""
V-JEPA 2 × LoRA — Experiment Runner

Runs experiments 2 at a time on 2 GPUs, fully automated.
Pairs experiments in order and runs each pair through all
participants before moving to the next pair.

Usage:
    # Run a single pair:
    python run.py \\
        --configs configs/baseline.yaml configs/lora_r8.yaml \\
        --devices cuda:0 cuda:1

    # Run all experiments (paired automatically):
    python run.py \\
        --configs configs/baseline.yaml configs/lora_r8.yaml \\
                  configs/lora_r16.yaml configs/lora_r32.yaml \\
                  configs/lora_r4.yaml configs/qlora.yaml \\
        --devices cuda:0 cuda:1

    # Run from a list file:
    python run.py --config-list experiments.txt --devices cuda:0 cuda:1

    # Single experiment on one GPU (backwards compatible):
    python run.py --configs configs/baseline.yaml --devices cuda:0
"""

import argparse
import yaml
import sys
import os
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.manager import PipelineManager


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='V-JEPA 2 × LoRA experiment runner'
    )
    parser.add_argument('--configs', nargs='+',
                        help='Experiment config files (paired in order)')
    parser.add_argument('--config-list', type=str,
                        help='Text file with one config path per line')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1'],
                        help='GPUs to use (default: cuda:0 cuda:1)')
    parser.add_argument('--global-config', type=str,
                        default='configs/global.yaml',
                        help='Path to global config')

    # Legacy support: --config for single experiment
    parser.add_argument('--config', type=str,
                        help='(Legacy) Single experiment config')

    args = parser.parse_args()

    # ── Collect config paths ──
    config_paths = []

    if args.config:
        # Legacy single-config mode
        config_paths = [args.config]
    elif args.config_list:
        with open(args.config_list, 'r') as f:
            config_paths = [line.strip() for line in f
                          if line.strip() and not line.startswith('#')]
    elif args.configs:
        config_paths = args.configs

    if not config_paths:
        print("ERROR: No experiment configs provided.")
        print("Usage:")
        print("  python run.py --configs config1.yaml config2.yaml "
              "--devices cuda:0 cuda:1")
        print("  python run.py --config-list experiments.txt "
              "--devices cuda:0 cuda:1")
        print("  python run.py --config configs/baseline.yaml "
              "--devices cuda:0  (single)")
        sys.exit(1)

    # ── Load global config ──
    global_config = load_yaml(args.global_config)

    # ── Load all experiment configs ──
    experiments = []
    for path in config_paths:
        if not os.path.exists(path):
            print(f"ERROR: Config not found: {path}")
            sys.exit(1)
        experiments.append((path, load_yaml(path)))

    devices = args.devices
    num_gpus = len(devices)

    # ── Build pairs (chunk by number of GPUs) ──
    pairs = []
    for i in range(0, len(experiments), num_gpus):
        chunk = experiments[i:i + num_gpus]
        pairs.append(chunk)

    # ── Print execution plan ──
    start_time = datetime.now()
    print("\n" + "=" * 70)
    print("V-JEPA 2 × LoRA — BATCH RUNNER")
    print(f"Started:      {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Devices:      {', '.join(devices)}")
    print(f"Experiments:  {len(experiments)}")
    print(f"Batches:      {len(pairs)}")
    print("-" * 70)

    for idx, chunk in enumerate(pairs, 1):
        names = []
        for j, (path, cfg) in enumerate(chunk):
            dev = devices[j] if j < len(devices) else devices[0]
            names.append(f"{cfg['experiment_name']} ({dev})")
        print(f"  Batch {idx}: {' + '.join(names)}")

    print("=" * 70)

    # ── Run each pair ──
    results = []

    for idx, chunk in enumerate(pairs, 1):
        batch_start = datetime.now()
        print(f"\n{'#' * 70}")
        print(f"BATCH {idx}/{len(pairs)}")
        print(f"Started: {batch_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#' * 70}")

        # Build (experiment_config, device) tuples for this batch
        exp_device_pairs = []
        for j, (path, cfg) in enumerate(chunk):
            dev = devices[j] if j < len(devices) else devices[0]
            exp_device_pairs.append((cfg, dev))
            print(f"  {cfg['experiment_name']} on {dev}")

        try:
            manager = PipelineManager(
                global_config=global_config,
                experiments=exp_device_pairs,
            )
            manager.run()

            for cfg, dev in exp_device_pairs:
                results.append((cfg['experiment_name'], "OK"))

        except Exception as e:
            print(f"\nFATAL ERROR in batch {idx}:")
            traceback.print_exc()
            for cfg, dev in exp_device_pairs:
                results.append((cfg['experiment_name'], f"FAILED: {e}"))

        batch_end = datetime.now()
        print(f"\nBatch {idx} finished in {batch_end - batch_start}")

    # ── Final summary ──
    end_time = datetime.now()
    total_duration = end_time - start_time

    print("\n" + "=" * 70)
    print("ALL BATCHES COMPLETE")
    print(f"Total time: {total_duration}")
    print("-" * 70)
    for name, status in results:
        marker = "OK" if status == "OK" else "FAILED"
        print(f"  [{marker}] {name}: {status}")
    print("=" * 70)


if __name__ == '__main__':
    main()
