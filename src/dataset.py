"""
EK-100 Action Anticipation Dataset.

Loads pre-extracted JPEG frames for a given set of participants.
Designed to work with the sequential pipeline — each call to
build_dataloader() creates a loader for specific participants only.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.vocabulary import get_action_id


class EK100Dataset(Dataset):
    """
    EPIC-KITCHENS-100 Action Anticipation Dataset.

    For each action annotation:
      - Load 64 frames ending 1 second before action starts
      - Return frames + verb/noun/action labels

    Args:
        csv_path:       path to EPIC_100_train.csv or validation.csv
        frames_dir:     root folder with P01/P01_01/frame_000...jpg
        action_to_id:   dict from vocabulary.py
        participants:   list of participant IDs to include, e.g. ['P01']
                        None = use all participants in CSV
        processor:      AutoVideoProcessor from V-JEPA 2
        fps:            frame rate (60 for EK-100)
        anticipation_s: seconds before action start (1.0)
        num_frames:     frames per clip (64)
        split:          'train' or 'validation' (for logging)
    """

    def __init__(self, csv_path, frames_dir, action_to_id,
                 participants=None, processor=None,
                 fps=60, anticipation_s=1.0, num_frames=64,
                 split='train'):

        self.frames_dir = frames_dir
        self.processor = processor
        self.action_to_id = action_to_id
        self.num_frames = num_frames
        self.split = split

        self.anticipation_frames = int(fps * anticipation_s)

        df = pd.read_csv(csv_path)

        # Filter by participants if specified
        if participants is not None:
            df = df[df['participant_id'].isin(participants)]

        # Filter clips where anticipation window goes negative
        min_start = self.anticipation_frames + num_frames
        df = df[df['start_frame'] > min_start]
        df = df.reset_index(drop=True)

        # Filter clips with missing frame folders
        valid_mask = df.apply(self._video_exists, axis=1)
        n_before = len(df)
        df = df[valid_mask].reset_index(drop=True)
        n_skipped = n_before - len(df)

        if n_skipped > 0:
            print(f"[{split}] Skipped {n_skipped} clips "
                  f"with missing frame folders")

        if len(df) == 0:
            self.annotations = df.assign(action_class=pd.Series(dtype='int64'))
            p_str = participants if participants else 'all'
            print(f"[{split}] Loaded 0 clips | participants: {p_str}")
            return

        # Map (verb, noun) -> action_class
        df['action_class'] = [
            get_action_id(row.verb_class, row.noun_class, action_to_id)
            for row in df.itertuples(index=False)
        ]

        n_unknown = (df['action_class'] == -1).sum()
        if n_unknown > 0:
            print(f"[{split}] {n_unknown}/{len(df)} clips have unseen "
                  f"(verb,noun) pairs (action_class=-1)")

        self.annotations = df
        p_str = participants if participants else 'all'
        print(f"[{split}] Loaded {len(df)} clips "
              f"| participants: {p_str}")

    def _video_exists(self, row):
        folder = os.path.join(
            self.frames_dir,
            row['participant_id'],
            row['video_id']
        )
        return os.path.exists(folder)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        start_frame = int(row['start_frame'])
        verb_class = int(row['verb_class'])
        noun_class = int(row['noun_class'])
        action_class = int(row['action_class'])
        participant_id = row['participant_id']
        video_id = row['video_id']

        # Compute anticipation window
        last_visible = start_frame - self.anticipation_frames
        first_frame = max(0, last_visible - self.num_frames)

        frame_indices = np.linspace(
            first_frame,
            last_visible - 1,
            num=self.num_frames,
            dtype=int
        )

        frames_folder = os.path.join(
            self.frames_dir, participant_id, video_id
        )

        try:
            frames = self._load_frames(frames_folder, frame_indices)
        except Exception as e:
            print(f"Warning: failed to load {frames_folder}: {e}")
            frames = torch.zeros(self.num_frames, 3, 256, 256)

        if self.processor is not None:
            inputs = self.processor(frames, return_tensors="pt")
            frames = inputs['pixel_values_videos'].squeeze(0)

        return {
            'frames': frames,
            'verb_label': torch.tensor(verb_class, dtype=torch.long),
            'noun_label': torch.tensor(noun_class, dtype=torch.long),
            'action_label': torch.tensor(action_class, dtype=torch.long),
            'video_id': video_id,
        }

    def _load_frames(self, frames_folder, frame_indices):
        """
        Load specific frames from JPEG folder.
        Frame files: frame_0000000001.jpg (1-indexed).
        Returns [T, C, H, W] uint8 tensor.
        """
        all_frames = sorted([
            f for f in os.listdir(frames_folder)
            if f.endswith('.jpg')
        ])
        total = len(all_frames)

        frames = []
        for idx in frame_indices:
            idx = max(0, min(idx, total - 1))
            path = os.path.join(frames_folder, all_frames[idx])
            img = Image.open(path).convert('RGB')
            tensor = torch.from_numpy(
                np.array(img)
            ).permute(2, 0, 1)  # [3, H, W]
            frames.append(tensor)

        return torch.stack(frames, dim=0)  # [T, 3, H, W]


def build_dataloader(csv_path, frames_dir, action_to_id,
                     participants=None, processor=None,
                     batch_size=8, num_workers=4,
                     fps=60, anticipation_s=1.0, num_frames=64,
                     split='train', shuffle=None):
    """
    Builds a DataLoader for specific participants.

    Args:
        csv_path:      annotation CSV path
        frames_dir:    root frames directory
        action_to_id:  vocabulary dict
        participants:  list of participant IDs or None for all
        processor:     V-JEPA 2 video processor
        batch_size:    samples per batch
        num_workers:   dataloader workers
        split:         'train' or 'validation'
        shuffle:       if None, auto (True for train, False for val)

    Returns:
        DataLoader
    """
    dataset = EK100Dataset(
        csv_path=csv_path,
        frames_dir=frames_dir,
        action_to_id=action_to_id,
        participants=participants,
        processor=processor,
        fps=int(fps),
        anticipation_s=float(anticipation_s),
        num_frames=int(num_frames),
        split=split
    )

    if shuffle is None:
        shuffle = (split == 'train')

    if len(dataset) == 0:
        raise ValueError(
            f"[{split}] No valid clips found for participants={participants}. "
            f"Check that frames_dir contains participant/video folders: "
            f"{frames_dir}"
        )

    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=True
    )

    print(f"[{split}] {len(loader)} batches "
          f"(batch_size={batch_size})")

    return loader
