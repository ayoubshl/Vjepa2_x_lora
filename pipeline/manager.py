"""
Pipeline Manager — orchestrates download → train → delete.

Supports two modes:
  - Single experiment on one GPU
  - Two experiments in parallel on two GPUs, sharing frames

In parallel mode, both experiments train on the same participant
simultaneously. Download of participant N+1 starts as soon as
training on N begins, hiding download latency behind training time.

Flow:
  1. Build global vocabulary (once)
  2. Download participant N (blocking if not on disk)
  3. Start background download of participant N+1
  4. Train both experiments on N in parallel (separate GPUs)
  5. Wait for both to finish + N+1 download to finish
  6. Delete N, move to N+1
  7. Repeat until all participants done
"""

import os
import copy
import threading
import traceback

from pipeline.downloader import FrameDownloader
from pipeline.scheduler import PipelineScheduler
from src.vocabulary import build_action_vocabulary, load_action_vocabulary
from train import train_on_participant


class PipelineManager:
    """
    Args:
        global_config:      parsed global.yaml
        experiments:        list of (experiment_config, device_string) tuples
                            e.g. [(baseline_cfg, 'cuda:0'), (lora_cfg, 'cuda:1')]
                            or   [(baseline_cfg, 'cuda:0')]  for single mode
    """

    def __init__(self, global_config, experiments):
        self.global_config = global_config
        self.experiments = experiments

        paths = global_config['paths']
        pipeline = global_config['pipeline']

        self.frames_dir = os.path.expanduser(paths['frames_dir'])
        self.train_csv = os.path.expanduser(paths['train_csv'])
        self.val_csv = os.path.expanduser(paths['val_csv'])
        self.vocab_path = os.path.expanduser(paths['vocabulary_path'])

        # Create checkpoint/results dirs for each experiment
        for exp_config, _ in self.experiments:
            ckpt_dir = os.path.expanduser(
                os.path.join(paths['checkpoints_dir'],
                             exp_config['experiment_name'])
            )
            res_dir = os.path.expanduser(
                os.path.join(paths['results_dir'],
                             exp_config['experiment_name'])
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

        os.makedirs(self.frames_dir, exist_ok=True)

        # Shared downloader
        self.downloader = FrameDownloader(
            downloader_dir=os.path.expanduser(paths['downloader_dir']),
            frames_dir=self.frames_dir
        )

        # Shared scheduler — state file in first experiment's checkpoint dir
        first_ckpt = os.path.expanduser(
            os.path.join(paths['checkpoints_dir'],
                         experiments[0][0]['experiment_name'])
        )
        state_path = os.path.join(first_ckpt, 'pipeline_state.json')

        self.scheduler = PipelineScheduler(
            train_csv=self.train_csv,
            state_path=state_path,
            max_on_disk=int(pipeline.get('max_participants_on_disk', 3))
        )

    def run(self):
        """Main pipeline loop. Handles both single and parallel modes."""
        exp_names = [cfg['experiment_name'] for cfg, _ in self.experiments]
        devices = [dev for _, dev in self.experiments]

        print("\n" + "=" * 60)
        print("PIPELINE START")
        for name, dev in zip(exp_names, devices):
            print(f"  {name} on {dev}")
        print("=" * 60)

        # Step 1: Vocabulary
        self._ensure_vocabulary()
        action_to_id, num_actions = load_action_vocabulary(self.vocab_path)

        self.scheduler.summary()

        # Step 2: Main loop
        while not self.scheduler.is_done():
            current = self.scheduler.get_next_to_train()
            if current is None:
                break

            # Download current participant if needed (blocking)
            if not self.downloader.participant_ready(current):
                print(f"\n{current} not on disk — downloading (blocking)...")
                success = self.downloader.download_participant(current)
                if not success:
                    print(f"FATAL: Failed to download {current}")
                    break
                self.scheduler.mark_on_disk(current)

            self.scheduler.mark_training(current)
            self.scheduler.summary()

            # Prefetch next participant in background
            bg_thread, bg_participant = self._start_prefetch(current)

            # Train all experiments on current participant
            print(f"\n{'=' * 60}")
            print(f"TRAINING: {current}")
            for name, dev in zip(exp_names, devices):
                print(f"  {name} on {dev}")
            print(f"{'=' * 60}")

            errors = self._train_parallel(
                current, action_to_id, num_actions
            )

            for name, err in errors.items():
                if err is not None:
                    print(f"WARNING: {name} failed on {current}: {err}")

            # Mark trained
            self.scheduler.mark_trained(current)

            # Wait for prefetch before deleting
            if bg_thread is not None and bg_thread.is_alive():
                print(f"\nWaiting for {bg_participant} download...")
                bg_thread.join()
            if bg_participant is not None:
                self.scheduler.mark_on_disk(bg_participant)

            # Delete current participant
            self.downloader.delete_participant(current)
            self.scheduler.mark_deleted(current)

            self.scheduler.summary()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Trained on {len(self.scheduler.state['trained'])} "
              "participants")
        print("=" * 60)

    def _train_parallel(self, participant_id, action_to_id, num_actions):
        """
        Runs all experiments on one participant in parallel threads.
        Each experiment gets its own deep-copied global_config with
        the correct device.

        Returns:
            dict of {experiment_name: exception_or_None}
        """
        errors = {}
        threads = []

        for exp_config, device in self.experiments:
            name = exp_config['experiment_name']
            errors[name] = None

            # Each trainer gets its own config copy with correct device
            gcfg = copy.deepcopy(self.global_config)
            gcfg['pipeline']['device'] = device

            def run_trainer(n=name, gc=gcfg, ec=exp_config):
                try:
                    train_on_participant(
                        participant_id=participant_id,
                        global_config=gc,
                        experiment_config=ec,
                        action_to_id=action_to_id,
                        num_actions=num_actions,
                    )
                except Exception as e:
                    errors[n] = e
                    print(f"\nERROR in {n}: {e}")
                    traceback.print_exc()

            t = threading.Thread(target=run_trainer, name=name)
            threads.append(t)

        # Start all
        for t in threads:
            t.start()

        # Wait for all
        for t in threads:
            t.join()

        return errors

    def _start_prefetch(self, current_participant):
        """
        Starts background download of the next participant.

        Returns:
            (thread, participant_id) or (None, None)
        """
        remaining = self.scheduler.get_remaining_participants()
        next_p = None
        for p in remaining:
            if p != current_participant:
                next_p = p
                break

        if next_p is None:
            return None, None

        if self.downloader.participant_ready(next_p):
            self.scheduler.mark_on_disk(next_p)
            return None, next_p

        thread = self.downloader.download_in_background(next_p)
        print(f"Prefetch started: {next_p}")
        return thread, next_p

    def _ensure_vocabulary(self):
        """Builds vocabulary if it doesn't exist yet."""
        if os.path.exists(self.vocab_path):
            print(f"Vocabulary already exists: {self.vocab_path}")
            return

        print("Building global action vocabulary...")
        build_action_vocabulary(
            train_csv=self.train_csv,
            save_path=self.vocab_path
        )
