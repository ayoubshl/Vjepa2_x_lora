"""
EK-100 Action Anticipation dataset (Decord backend, paper-matched sampling).

# Loads raw MP4 videos, not pre-extracted JPEG frames.
# 32 frames @ 8 FPS, ending 1 second before the action start.
# Resulting input shape matches what V-JEPA 2 expects.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# HINT: import Decord lazily so the module loads even if Decord isn't
# installed yet (e.g., when reading config). Decord is the standard for
# ML video pipelines because it seeks by frame index without decoding
# the rest of the video.
try:
    import decord
    decord.bridge.set_bridge("torch")  # get PyTorch tensors directly
except ImportError:
    decord = None

from src.vocabulary import get_action_id


# ---------------------------------------------------------------------
# Frame index computation — the single source of truth for sampling
# ---------------------------------------------------------------------

def compute_frame_indices(
    start_frame: int,
    fps_source: int = 60,
    fps_target: int = 8,
    num_frames: int = 32,
    anticipation_seconds: float = 1.0,
) -> Optional[np.ndarray]:
    """
    Compute the 32 frame indices for an anticipation clip.

    Paper protocol (Section 6 / Appendix 13.1):
      - context ends `anticipation_seconds` (default 1.0) before action start
      - context window spans `num_frames / fps_target` seconds
        (default 32/8 = 4 seconds)
      - sample `num_frames` indices linearly inside that window

    Args:
        start_frame:           action's start_frame in fps_source units
        fps_source:            source video fps (EK-100 = 60)
        fps_target:            target sampling fps (paper = 8)
        num_frames:            number of frames per clip (paper = 32)
        anticipation_seconds:  gap between context end and action start

    Returns:
        np.ndarray of `num_frames` ints, or None if window underflows frame 0.
    """
    # End of context = start_frame - 1 second of source frames
    anticipation_offset = int(fps_source * anticipation_seconds)
    context_end = start_frame - anticipation_offset

    # Span of context window in source frames
    span_seconds = num_frames / fps_target
    context_start = context_end - int(fps_source * span_seconds)

    # HINT: if the window goes before frame 0, the clip is unusable.
    # Don't pad — that biases the model. Just drop it.
    if context_start < 0:
        return None

    # Linearly spaced indices inside [context_start, context_end-1].
    # context_end is exclusive (the frame 1 second BEFORE the action,
    # not the action's first frame).
    indices = np.linspace(
        context_start, context_end - 1, num=num_frames, dtype=np.int64
    )
    return indices


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class EK100AnticipationDataset(Dataset):
    """
    EK-100 anticipation dataset using Decord video loading.

    Args:
        csv_path:       official train or validation CSV
        videos_dir:     root folder with PXX/PXX_YY.MP4
        action_to_id:   vocab from src.vocabulary
        participants:   list of participant IDs to include (None = all in CSV)
        processor:      HF AutoVideoProcessor instance (handles resize/normalize)
        fps_source:     EK-100 native fps
        fps_target:     paper sampling fps
        num_frames:     paper context length
        anticipation_s: paper anticipation gap (seconds)
        split:          'train' or 'validation' (logging only)
        cache_dir:      if given, cache parsed annotations here for fast reload
    """

    def __init__(
        self,
        csv_path: str,
        videos_dir: str,
        action_to_id: Dict[str, int],
        participants: Optional[List[str]] = None,
        processor=None,
        fps_source: int = 60,
        fps_target: int = 8,
        num_frames: int = 32,
        anticipation_s: float = 1.0,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        if decord is None:
            raise ImportError(
                "decord is required. `pip install decord` "
                "(or `pip install eva-decord` on macOS)."
            )

        self.videos_dir = videos_dir
        self.processor = processor
        self.action_to_id = action_to_id
        self.fps_source = int(fps_source)
        self.fps_target = int(fps_target)
        self.num_frames = int(num_frames)
        self.anticipation_s = float(anticipation_s)
        self.split = split

        # HINT: cache parsed/filtered annotations to a pickle. First load is
        # slow (CSV parse + per-row checks); subsequent loads are instant.
        cache_path = None
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            tag = (
                f"{split}_p{'-'.join(participants) if participants else 'all'}"
                f"_f{num_frames}_fps{fps_target}_a{anticipation_s}"
            )
            # Hash long participant lists to keep filename sane
            if participants is not None and len(participants) > 5:
                import hashlib
                h = hashlib.md5(",".join(sorted(participants)).encode()).hexdigest()[:8]
                tag = f"{split}_p{len(participants)}-{h}_f{num_frames}_fps{fps_target}_a{anticipation_s}"
            cache_path = os.path.join(cache_dir, f"{tag}.pkl")

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.annotations = pickle.load(f)
            print(f"[{split}] Loaded {len(self.annotations)} clips from cache: {cache_path}")
            return

        df = self._parse_csv(csv_path, participants)
        self.annotations = df

        if cache_path:
            with open(cache_path, "wb") as f:
                pickle.dump(self.annotations, f)
            print(f"[{split}] Cached annotations to {cache_path}")

    def _parse_csv(self, csv_path: str, participants: Optional[List[str]]) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        if participants is not None:
            df = df[df["participant_id"].isin(participants)]

        # Drop clips whose 4-second context window goes before frame 0.
        min_start = int(self.fps_source * self.anticipation_s) + int(
            self.fps_source * self.num_frames / self.fps_target
        )
        n_before = len(df)
        df = df[df["start_frame"] > min_start].reset_index(drop=True)
        n_dropped_underflow = n_before - len(df)

        # Drop clips whose video file doesn't exist on disk.
        valid_mask = df.apply(self._video_exists, axis=1)
        n_before = len(df)
        df = df[valid_mask].reset_index(drop=True)
        n_dropped_missing = n_before - len(df)

        # Map (verb,noun) → action_class. Unseen pairs → -1.
        df["action_class"] = [
            get_action_id(row.verb_class, row.noun_class, self.action_to_id)
            for row in df.itertuples(index=False)
        ]

        n_unseen = int((df["action_class"] == -1).sum())
        p_str = participants if participants else "ALL"
        print(
            f"[{self.split}] participants={p_str} | "
            f"clips={len(df)} | "
            f"dropped_underflow={n_dropped_underflow} | "
            f"dropped_missing_video={n_dropped_missing} | "
            f"unseen_action_pairs={n_unseen}"
        )
        return df

    def _video_exists(self, row) -> bool:
        path = self._video_path(row["participant_id"], row["video_id"])
        return os.path.exists(path)

    def _video_path(self, participant_id: str, video_id: str) -> str:
        # EK-100 video naming: PXX_YY.MP4 inside PXX/ folder
        # HINT: if you downloaded with a different layout, change here.
        return os.path.join(self.videos_dir, participant_id, f"{video_id}.MP4")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        row = self.annotations.iloc[idx]

        start_frame = int(row["start_frame"])
        verb_class = int(row["verb_class"])
        noun_class = int(row["noun_class"])
        action_class = int(row["action_class"])

        frame_indices = compute_frame_indices(
            start_frame=start_frame,
            fps_source=self.fps_source,
            fps_target=self.fps_target,
            num_frames=self.num_frames,
            anticipation_seconds=self.anticipation_s,
        )

        # Shouldn't happen — we filtered underflow in _parse_csv — but
        # guard anyway.
        if frame_indices is None:
            raise RuntimeError(f"Underflow at idx={idx} despite filtering")

        video_path = self._video_path(row["participant_id"], row["video_id"])

        try:
            frames = self._load_video_frames(video_path, frame_indices)
        except Exception as e:
            # HINT: never let one bad clip kill training. Log and return a
            # zero-filled fallback. If you see this happen often, debug
            # the offending file rather than training on garbage.
            print(f"[{self.split}] WARN failed to load {video_path}: {e}")
            frames = torch.zeros(self.num_frames, 3, 256, 256, dtype=torch.uint8)

        # HF AutoVideoProcessor expects (T, C, H, W) uint8 or (T, H, W, C).
        # Decord returns (T, H, W, C) by default; we permute to (T, C, H, W)
        # to match what the processor's docs use.
        if frames.ndim == 4 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2).contiguous()

        if self.processor is not None:
            # processor handles resize, center-crop, normalize. Returns
            # 'pixel_values_videos' shape [1, T, C, H, W].
            inputs = self.processor(frames, return_tensors="pt")
            frames = inputs["pixel_values_videos"].squeeze(0)

        return {
            "frames": frames,
            "verb_label":   torch.tensor(verb_class, dtype=torch.long),
            "noun_label":   torch.tensor(noun_class, dtype=torch.long),
            "action_label": torch.tensor(action_class, dtype=torch.long),
            "video_id":     row["video_id"],
            "start_frame":  start_frame,
        }

    def _load_video_frames(
        self,
        video_path: str,
        frame_indices: np.ndarray,
    ) -> torch.Tensor:
        """
        Load specific frames from a video using Decord.

        # HINT: Open VideoReader PER __getitem__ call, not once in __init__.
        # Sharing a VideoReader across workers via fork is unsafe and
        # causes silent data corruption.

        Returns:
            tensor [T, H, W, C] uint8 (Decord's native format)
        """
        # num_threads=1 because PyTorch DataLoader workers handle parallelism.
        vr = decord.VideoReader(video_path, num_threads=1)
        total = len(vr)

        # Clamp indices to be safe — videos can be slightly shorter than
        # the annotations expect because of trimming/re-encoding.
        clamped = np.clip(frame_indices, 0, total - 1)

        # get_batch returns [T, H, W, C] uint8 tensor (with torch bridge).
        frames = vr.get_batch(clamped.tolist())
        # Some Decord versions return a different type — normalize:
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames.asnumpy())
        return frames


# ---------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------

def build_dataloader(
    csv_path: str,
    videos_dir: str,
    action_to_id: Dict[str, int],
    participants: Optional[List[str]],
    processor,
    batch_size: int,
    num_workers: int,
    fps_source: int,
    fps_target: int,
    num_frames: int,
    anticipation_s: float,
    split: str,
    cache_dir: Optional[str] = None,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Build a DataLoader for train or validation.

    # HINT: shuffle is auto-determined unless overridden. Train shuffles,
    # validation doesn't.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = EK100AnticipationDataset(
        csv_path=csv_path,
        videos_dir=videos_dir,
        action_to_id=action_to_id,
        participants=participants,
        processor=processor,
        fps_source=fps_source,
        fps_target=fps_target,
        num_frames=num_frames,
        anticipation_s=anticipation_s,
        split=split,
        cache_dir=cache_dir,
    )

    if len(dataset) == 0:
        raise ValueError(
            f"[{split}] Empty dataset for participants={participants}. "
            f"Check videos_dir={videos_dir} and CSV path."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),  # avoid uneven last batch in training
        persistent_workers=(num_workers > 0),
    )

    print(f"[{split}] {len(loader)} batches | batch_size={batch_size}")
    return loader
