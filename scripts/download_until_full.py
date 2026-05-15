"""
Download as many EK-100 participants as fit in the configured storage budget.

This script is designed for unattended tmux runs. It is resumable: participants
already listed in participants.txt are skipped, and new participants are added
only after the downloader exits successfully and at least one MP4 is found.
"""

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

try:
    import wandb
except ImportError:
    wandb = None


GB = 1024 ** 3


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, log_file):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def expand_paths(global_config):
    return {k: os.path.expanduser(v) for k, v in global_config["paths"].items()}


def read_participants_from_csv(train_csv):
    participants = set()
    with open(train_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            participants.add(row["participant_id"])
    return sorted(participants, key=lambda p: int(p[1:]) if p.startswith("P") else p)


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


def append_participant(path, participant):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = set(read_participants_file(path))
    if participant in existing:
        return
    with p.open("a", encoding="utf-8") as f:
        f.write(participant + "\n")


def nearest_existing_path(path):
    p = Path(path).expanduser()
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def free_gb(path):
    usage = shutil.disk_usage(nearest_existing_path(path))
    return usage.free / GB


def directory_size_gb(path):
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0
    for root, _, files in os.walk(p):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total / GB


def participant_has_mp4(videos_dir, participant):
    roots = [
        Path(videos_dir) / participant,
        Path(videos_dir) / participant / "videos",
        Path(videos_dir) / "EPIC-KITCHENS" / participant,
        Path(videos_dir) / "EPIC-KITCHENS" / participant / "videos",
    ]
    for root in roots:
        if root.exists() and any(root.rglob("*.MP4")):
            return True
        if root.exists() and any(root.rglob("*.mp4")):
            return True
    return False


def build_download_command(project_root, downloader_path, participant, output_path):
    template = os.environ.get("EPIC_DOWNLOAD_COMMAND")
    values = {
        "participant": participant,
        "output_path": str(output_path),
        "project_root": str(project_root),
        "downloader": str(downloader_path),
        "python": sys.executable,
    }
    if template:
        return shlex.split(template.format(**values))

    return [
        sys.executable,
        str(downloader_path),
        "--videos",
        "--participants",
        participant,
        "--output-path",
        str(output_path),
    ]


def load_manifest(path):
    p = Path(path)
    if not p.exists():
        return {"participants": {}, "events": []}
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path, manifest):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp.replace(p)


def run_command(command, cwd, log_file):
    with open(log_file, "a", encoding="utf-8") as f:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-config", default="configs/global.yaml")
    parser.add_argument("--buffer-gb", type=float, default=50.0)
    parser.add_argument("--max-download-gb", type=float, default=750.0)
    parser.add_argument("--log-file", default="logs/download.log")
    parser.add_argument("--manifest", default="download_manifest.json")
    args = parser.parse_args()

    project_root = Path.cwd()
    log_file = project_root / args.log_file
    manifest_path = project_root / args.manifest
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.global_config, encoding="utf-8") as f:
        global_config = yaml.safe_load(f)
    paths = expand_paths(global_config)

    train_csv = Path(paths["train_csv"])
    videos_dir = Path(paths["videos_dir"])
    participants_file = Path(paths["participants_file"])
    downloader_path = project_root / "downloader" / "epic_downloader.py"

    videos_dir.mkdir(parents=True, exist_ok=True)
    participants_file.parent.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        raise FileNotFoundError(f"train CSV not found: {train_csv}")
    if not downloader_path.exists() and "EPIC_DOWNLOAD_COMMAND" not in os.environ:
        raise FileNotFoundError(
            "downloader/epic_downloader.py not found. Set EPIC_DOWNLOAD_COMMAND "
            "to a non-interactive command template if your downloader differs."
        )

    all_participants = read_participants_from_csv(train_csv)
    completed = set(read_participants_file(participants_file))
    manifest = load_manifest(manifest_path)

    log(f"download phase started | participants_in_csv={len(all_participants)}", log_file)
    log(f"videos_dir={videos_dir}", log_file)
    log(f"participants_file={participants_file}", log_file)
    log(f"buffer_gb={args.buffer_gb} max_download_gb={args.max_download_gb}", log_file)

    wandb_run = None
    if wandb is not None:
        wandb_cfg = global_config.get("wandb", {})
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "vjepa2-x-lora"),
            entity=wandb_cfg.get("entity") or None,
            name="download_phase",
            config={
                "buffer_gb": args.buffer_gb,
                "max_download_gb": args.max_download_gb,
                "videos_dir": str(videos_dir),
                "participants_csv_count": len(all_participants),
            },
            resume="allow",
        )

    for participant in all_participants:
        current_free = free_gb(videos_dir)
        current_size = directory_size_gb(videos_dir)
        if wandb_run is not None:
            wandb.log({
                "download/free_gb": current_free,
                "download/videos_dir_gb": current_size,
                "download/completed_participants": len(completed),
            })
        if current_free < args.buffer_gb:
            log(
                f"stopping before {participant}: free={current_free:.2f}GB "
                f"< buffer={args.buffer_gb:.2f}GB",
                log_file,
            )
            break
        if current_size >= max(0.0, args.max_download_gb - args.buffer_gb):
            log(
                f"stopping before {participant}: videos_dir_size={current_size:.2f}GB "
                f"reached budget cap",
                log_file,
            )
            break

        if participant in completed:
            log(f"skip {participant}: already listed in participants.txt", log_file)
            continue

        if participant_has_mp4(videos_dir, participant):
            append_participant(participants_file, participant)
            completed.add(participant)
            log(f"mark {participant}: MP4s already present", log_file)
            if wandb_run is not None:
                wandb.log({"download/participant_completed": len(completed)})
            continue

        command = build_download_command(project_root, downloader_path, participant, videos_dir)
        before_free = free_gb(videos_dir)
        before_size = directory_size_gb(videos_dir)
        log(f"downloading {participant} | free_before={before_free:.2f}GB", log_file)
        rc = run_command(command, project_root, log_file)
        after_free = free_gb(videos_dir)
        after_size = directory_size_gb(videos_dir)
        ok = rc == 0 and participant_has_mp4(videos_dir, participant)

        manifest["participants"][participant] = {
            "status": "completed" if ok else "failed",
            "returncode": rc,
            "free_gb_before": round(before_free, 3),
            "free_gb_after": round(after_free, 3),
            "videos_dir_gb_before": round(before_size, 3),
            "videos_dir_gb_after": round(after_size, 3),
            "command": command,
            "timestamp": now(),
        }
        save_manifest(manifest_path, manifest)

        if ok:
            append_participant(participants_file, participant)
            completed.add(participant)
            log(
                f"completed {participant} | free_after={after_free:.2f}GB "
                f"downloaded_delta={after_size - before_size:.2f}GB",
                log_file,
            )
            if wandb_run is not None:
                wandb.log({
                    "download/participant_completed": len(completed),
                    "download/last_download_delta_gb": after_size - before_size,
                    "download/free_gb": after_free,
                    "download/videos_dir_gb": after_size,
                })
        else:
            log(
                f"failed {participant} | returncode={rc} | free_after={after_free:.2f}GB; "
                "continuing to next participant",
                log_file,
            )
            if wandb_run is not None:
                wandb.log({"download/participant_failed": 1, "download/free_gb": after_free})

    final_participants = read_participants_file(participants_file)
    manifest["final"] = {
        "participants": final_participants,
        "num_participants": len(final_participants),
        "free_gb": round(free_gb(videos_dir), 3),
        "videos_dir_gb": round(directory_size_gb(videos_dir), 3),
        "timestamp": now(),
    }
    save_manifest(manifest_path, manifest)
    log(f"download phase finished | participants={len(final_participants)}", log_file)
    if wandb_run is not None:
        wandb.log({
            "download/final_participants": len(final_participants),
            "download/final_free_gb": manifest["final"]["free_gb"],
            "download/final_videos_dir_gb": manifest["final"]["videos_dir_gb"],
        })
        wandb.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
