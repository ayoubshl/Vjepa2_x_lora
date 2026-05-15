"""
Single unattended entry point for storage-limited EK-100 LoRA experiments.

Run inside tmux:
    python setup_and_run.py

The script runs:
  1. scripts/download_until_full.py
  2. scripts/run_all_experiments.py

Both phases write persistent logs and can be resumed by rerunning this file.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, log_file):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_phase(name, command, log_file):
    log(f"starting phase: {name}", log_file)
    log(f"command: {' '.join(shlex.quote(x) for x in command)}", log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] ===== {name} =====\n")
        f.flush()
        proc = subprocess.Popen(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        rc = proc.wait()
    log(f"finished phase: {name} returncode={rc}", log_file)
    return rc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--buffer-gb", type=float, default=50.0)
    parser.add_argument("--max-download-gb", type=float, default=750.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--summary", default="result_summary.json")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    project_root = Path.cwd()
    log_dir = project_root / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    master_log = log_dir / "setup_and_run.log"

    log("setup_and_run started", master_log)
    log(f"project_root={project_root}", master_log)
    log(f"global_config={args.global_config}", master_log)

    if not args.skip_download:
        download_cmd = [
            sys.executable,
            "scripts/download_until_full.py",
            "--global-config",
            args.global_config,
            "--buffer-gb",
            str(args.buffer_gb),
            "--max-download-gb",
            str(args.max_download_gb),
            "--log-file",
            str(log_dir / "download.log"),
            "--manifest",
            "download_manifest.json",
        ]
        rc = run_phase("download", download_cmd, master_log)
        if rc != 0:
            log(
                "download phase failed; continuing to training only if participants.txt exists",
                master_log,
            )
    else:
        log("download phase skipped by flag", master_log)

    if not args.skip_training:
        train_cmd = [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--global-config",
            args.global_config,
            "--summary",
            args.summary,
            "--log-dir",
            str(log_dir),
        ]
        rc = run_phase("training", train_cmd, master_log)
        if rc != 0:
            log(f"training phase exited with returncode={rc}", master_log)
            return rc
    else:
        log("training phase skipped by flag", master_log)

    log("setup_and_run finished", master_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
