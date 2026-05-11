"""
Pipeline Manager — orchestrates the download → train → delete cycle.

This is the main loop that runs unattended in tmux.
Coordinates the scheduler, downloader, and training.

Flow:
  1. Build global vocabulary (once)
  2. Download first participant (blocking)
  3. Start background download of next
  4. Train on current participant
  5. Delete trained participant, shift queue
  6. Repeat until all participants done
"""

import os
import time
import yaml

from pipeline.downloader import FrameDownloader
from pipeline.scheduler import PipelineScheduler
from src.vocabulary import build_action_vocabulary, load_action_vocabulary
from train import train_on_participant


class PipelineManager:
    """
    Args:
        global_config:      parsed global.yaml
        experiment_config:  parsed experiment yaml (baseline, lora, etc.)
    """

    def __init__(self, global_config, experiment_config):
        self.global_config = global_config
        self.experiment_config = experiment_config

        paths = global_config['paths']
        pipeline = global_config['pipeline']

        self.frames_dir = os.path.expanduser(paths['frames_dir'])
        self.train_csv = os.path.expanduser(paths['train_csv'])
        self.val_csv = os.path.expanduser(paths['val_csv'])
        self.vocab_path = os.path.expanduser(paths['vocabulary_path'])
        self.checkpoints_dir = os.path.expanduser(
            os.path.join(paths['checkpoints_dir'],
                         experiment_config['experiment_name'])
        )
        self.results_dir = os.path.expanduser(
            os.path.join(paths['results_dir'],
                         experiment_config['experiment_name'])
        )

        # State file per experiment so experiments don't clash
        state_path = os.path.join(
            self.checkpoints_dir, 'pipeline_state.json'
        )

        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        # Initialize components
        self.downloader = FrameDownloader(
            downloader_dir=os.path.expanduser(paths['downloader_dir']),
            frames_dir=self.frames_dir
        )

        self.scheduler = PipelineScheduler(
            train_csv=self.train_csv,
            state_path=state_path,
            max_on_disk=pipeline.get('max_participants_on_disk', 3)
        )

    def run(self):
        """
        Main pipeline loop. Runs until all participants are trained.
        """
        print("\n" + "=" * 60)
        print("PIPELINE START")
        print(f"Experiment: {self.experiment_config['experiment_name']}")
        print("=" * 60)

        # Step 1: Build or load vocabulary
        self._ensure_vocabulary()

        # Step 2: Load vocabulary
        action_to_id, num_actions = load_action_vocabulary(self.vocab_path)

        self.scheduler.summary()

        # Track background download thread
        bg_download_thread = None
        bg_download_participant = None

        # Step 3: Main loop
        while not self.scheduler.is_done():

            # Get next participant to train
            next_train = self.scheduler.get_next_to_train()
            if next_train is None:
                break

            # Ensure this participant is on disk
            if not self.downloader.participant_ready(next_train):
                # Need to download it (blocking)
                print(f"\n{next_train} not on disk — downloading (blocking)...")
                success = self.downloader.download_participant(next_train)
                if not success:
                    print(f"FATAL: Failed to download {next_train}")
                    break
                self.scheduler.mark_on_disk(next_train)

            # Start background download of next participant
            bg_download_thread, bg_download_participant = (
                self._start_background_download(
                    bg_download_thread, bg_download_participant
                )
            )

            # Train on current participant
            self.scheduler.mark_training(next_train)
            self.scheduler.summary()

            print(f"\n{'=' * 60}")
            print(f"TRAINING: {next_train}")
            print(f"{'=' * 60}")

            train_on_participant(
                participant_id=next_train,
                global_config=self.global_config,
                experiment_config=self.experiment_config,
                action_to_id=action_to_id,
                num_actions=num_actions,
            )

            # Mark as trained
            self.scheduler.mark_trained(next_train)

            # Wait for background download before deleting
            if bg_download_thread is not None and bg_download_thread.is_alive():
                print(f"\nWaiting for {bg_download_participant} download...")
                bg_download_thread.join()
                self.scheduler.mark_on_disk(bg_download_participant)
                bg_download_thread = None
                bg_download_participant = None

            # Delete trained participant to free space
            self.downloader.delete_participant(next_train)
            self.scheduler.mark_deleted(next_train)

            # Start next background download if room
            bg_download_thread, bg_download_participant = (
                self._start_background_download(
                    bg_download_thread, bg_download_participant
                )
            )

            self.scheduler.summary()

        # Cleanup: wait for any remaining download
        if bg_download_thread is not None and bg_download_thread.is_alive():
            bg_download_thread.join()

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Trained on {len(self.scheduler.state['trained'])} participants")
        print("=" * 60)

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

    def _start_background_download(self, current_thread,
                                   current_participant):
        """
        Starts a background download if:
          - No download is currently running
          - There's room on disk
          - There are participants left to download

        Args:
            current_thread:      existing bg thread or None
            current_participant:  participant being downloaded or None

        Returns:
            (thread, participant_id) tuple
        """
        # Already downloading
        if current_thread is not None and current_thread.is_alive():
            return current_thread, current_participant

        # Check if finished download needs to be registered
        if current_thread is not None and current_participant is not None:
            self.scheduler.mark_on_disk(current_participant)

        # Check disk space
        if not self.scheduler.should_download_more():
            return None, None

        # Get next to download
        next_dl = self.scheduler.get_next_to_download()
        if next_dl is None:
            return None, None

        # Already on disk
        if self.downloader.participant_ready(next_dl):
            self.scheduler.mark_on_disk(next_dl)
            return None, None

        # Start download
        thread = self.downloader.download_in_background(next_dl)
        return thread, next_dl