"""
Pipeline Scheduler — tracks which participants are done,
downloading, or queued.

Reads all unique participant IDs from the training CSV
so it works regardless of how many participants exist.
# Saves state to JSON for resume support.
"""

import os
import json
import pandas as pd


class PipelineScheduler:
    """
    Manages the participant queue for sequential training.

    State on disk at any time:
      slot 1: currently training     (extracted frames)
      slot 2: downloaded and ready   (extracted frames)
      slot 3: currently downloading  (one at a time)
      max 3 participants on disk

    Args:
        train_csv:      path to EPIC_100_train.csv
        state_path:     path to pipeline_state.json
        max_on_disk:    max participants on disk at once (default 3)
    """

    def __init__(self, train_csv, state_path, max_on_disk=3):
        self.state_path = state_path
        self.max_on_disk = max_on_disk

        # Get all participants from CSV, sorted
        df = pd.read_csv(train_csv)
        self.all_participants = sorted(df['participant_id'].unique().tolist())
        print(f"Found {len(self.all_participants)} participants: "
              f"{self.all_participants[0]} — {self.all_participants[-1]}")

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self):
        """Loads state from JSON or creates fresh state."""
        if os.path.exists(self.state_path):
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            print(f"Pipeline state loaded from {self.state_path}")
            print(f"  trained: {state['trained']}")
            print(f"  current: {state['current']}")
            return state

        state = {
            'trained': [],
            'current': None,
            'on_disk': [],
        }
        self._save_state(state)
        return state

    def _save_state(self, state=None):
        """Saves current state to JSON."""
        if state is None:
            state = self.state
        with open(self.state_path, 'w') as f:
            json.dump(state, f, indent=2)

    def get_remaining_participants(self):
        """
        Returns list of participants not yet trained.

        Returns:
            list of participant IDs still in the queue
        """
        trained = set(self.state['trained'])
        return [p for p in self.all_participants if p not in trained]

    def get_next_to_train(self):
        """
        Returns the next participant to train.

        Returns:
            participant ID string or None if all done
        """
        remaining = self.get_remaining_participants()
        if not remaining:
            return None
        return remaining[0]

    def get_next_to_download(self):
        """
        Returns the next participant that should be downloading.
        Looks ahead in the queue past what's already on disk.

        Returns:
            participant ID string or None if nothing to download
        """
        remaining = self.get_remaining_participants()
        on_disk = set(self.state['on_disk'])

        for p in remaining:
            if p not in on_disk:
                return p
        return None

    def should_download_more(self):
        """
        Checks if we should start another download.
        True if on_disk count is below max_on_disk.

        Returns:
            bool
        """
        return len(self.state['on_disk']) < self.max_on_disk

    def mark_on_disk(self, participant_id):
        """Marks a participant as downloaded and on disk."""
        if participant_id not in self.state['on_disk']:
            self.state['on_disk'].append(participant_id)
        self._save_state()

    def mark_training(self, participant_id):
        """Marks a participant as currently training."""
        self.state['current'] = participant_id
        self._save_state()

    def mark_trained(self, participant_id):
        """Marks a participant as fully trained."""
        if participant_id not in self.state['trained']:
            self.state['trained'].append(participant_id)
        self.state['current'] = None
        self._save_state()

    def mark_deleted(self, participant_id):
        """Marks a participant's frames as deleted from disk."""
        if participant_id in self.state['on_disk']:
            self.state['on_disk'].remove(participant_id)
        self._save_state()

    def is_done(self):
        """Returns True if all participants have been trained."""
        return len(self.state['trained']) == len(self.all_participants)

    def get_num_participants(self):
        """Returns total number of participants."""
        return len(self.all_participants)

    def summary(self):
        """Prints current pipeline status."""
        remaining = self.get_remaining_participants()
        print(f"\n--- Pipeline Status ---")
        print(f"  Total:       {len(self.all_participants)}")
        print(f"  Trained:     {len(self.state['trained'])}")
        print(f"  On disk:     {self.state['on_disk']}")
        print(f"  Current:     {self.state['current']}")
        print(f"  Remaining:   {len(remaining)}")
        print(f"-----------------------\n")