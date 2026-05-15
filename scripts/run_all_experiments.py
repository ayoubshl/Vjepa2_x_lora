"""
Run all V-JEPA2 x LoRA experiments sequentially and summarize results.

Each experiment runs in its own subprocess. Failures are recorded, but do not
stop the remaining experiments.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import torch
import yaml


EXPERIMENTS = [
    ("baseline_frozen", "configs/baseline.yaml"),
    ("lora_r4", "configs/lora_r4.yaml"),
    ("lora_r8", "configs/lora_r8.yaml"),
    ("lora_r16", "configs/lora_r16.yaml"),
    ("lora_r32", "configs/lora_r32.yaml"),
    ("lora_upper_layers", "configs/lora_upper_layers.yaml"),
    ("qlora", "configs/qlora.yaml"),
]


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, log_file):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_paths(global_config):
    return {k: os.path.expanduser(v) for k, v in global_config["paths"].items()}


def read_participants_file(path):
    p = Path(path)
    if not p.exists():
        return []
    participants = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                participants.append(line)
    return participants


def run_command(command, cwd, log_path):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n[{now()}] COMMAND: {' '.join(shlex.quote(x) for x in command)}\n")
        f.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        return proc.wait()


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint_summary(checkpoint_dir):
    latest = Path(checkpoint_dir) / "latest.pth"
    summary_json = Path(checkpoint_dir) / "summary.json"
    out = {}

    if summary_json.exists():
        with summary_json.open(encoding="utf-8") as f:
            out.update(json.load(f))

    if latest.exists():
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        history = ckpt.get("history", [])
        final_eval = None
        for item in reversed(history):
            if "action_mR5" in item:
                final_eval = item
                break
        out.update({
            "epoch": ckpt.get("epoch"),
            "global_step": ckpt.get("global_step"),
            "best_action_mR5": ckpt.get("best_action_mR5"),
            "peak_gpu_mem_bytes": ckpt.get("peak_gpu_mem_bytes"),
            "total_train_time_seconds": ckpt.get("total_train_time"),
            "checkpoint_path": str(latest),
        })
        if final_eval is not None:
            out.update({
                "final_eval_epoch": final_eval.get("epoch"),
                "final_verb_mR5": final_eval.get("verb_mR5"),
                "final_noun_mR5": final_eval.get("noun_mR5"),
                "final_action_mR5": final_eval.get("action_mR5"),
            })
    return out


def save_summary(path, summary):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    tmp.replace(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--summary", default="result_summary.json")
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    project_root = Path.cwd()
    log_dir = project_root / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log = log_dir / "train_all.log"

    global_config = load_yaml(args.global_config)
    paths = expand_paths(global_config)
    participants = read_participants_file(paths["participants_file"])
    if not participants:
        raise RuntimeError(
            f"No participants found in {paths['participants_file']}; "
            "download phase did not produce a usable subset."
        )

    summary = {
        "started_at": now(),
        "participants": participants,
        "num_participants": len(participants),
        "experiments": {},
    }
    save_summary(project_root / args.summary, summary)

    log(f"training phase started | participants={participants}", runner_log)
    for name, config_path in EXPERIMENTS:
        exp_log = log_dir / f"train_{name}.log"
        started = now()
        log(f"starting {name} with {config_path}", runner_log)

        if not Path(config_path).exists():
            msg = f"config not found: {config_path}"
            log(f"failed {name}: {msg}", runner_log)
            summary["experiments"][name] = {
                "status": "failed",
                "config": config_path,
                "started_at": started,
                "finished_at": now(),
                "error": msg,
            }
            save_summary(project_root / args.summary, summary)
            continue

        command = [
            sys.executable,
            "run.py",
            "--config",
            config_path,
            "--global-config",
            args.global_config,
        ]

        rc = run_command(command, project_root, exp_log)
        finished = now()
        record = {
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
            "config": config_path,
            "log_path": str(exp_log),
            "started_at": started,
            "finished_at": finished,
        }

        try:
            exp_cfg = load_yaml(config_path)
            exp_name = exp_cfg.get("experiment_name", name)
            ckpt_dir = Path(paths["checkpoints_dir"]) / exp_name
            pred_dir = Path(paths["predictions_dir"]) / exp_name
            record["experiment_name"] = exp_name
            record["predictions_dir"] = str(pred_dir)
            record.update(load_checkpoint_summary(ckpt_dir))
        except Exception as exc:
            record["summary_error"] = f"{type(exc).__name__}: {exc}"
            record["summary_traceback"] = traceback.format_exc()

        summary["experiments"][name] = record
        save_summary(project_root / args.summary, summary)

        if rc == 0:
            log(f"completed {name}", runner_log)
        else:
            log(f"failed {name} returncode={rc}; continuing", runner_log)

    summary["finished_at"] = now()
    save_summary(project_root / args.summary, summary)
    log(f"training phase finished | summary={project_root / args.summary}", runner_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
