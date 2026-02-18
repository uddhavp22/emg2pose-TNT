"""WebDataset-based streaming dataset for EMG2Pose tar shards.

Streams large sequential tar shards from GCS (or local disk) without random
per-sample file access. Shard-level shuffle + in-memory ring-buffer shuffle
give good statistical properties while preserving sequential I/O within each shard.

Shard tar layout (produced by create_shards.py):
    {key}.emg.npy      float32 (T, 16)   EMG signals
    {key}.joints.npy   float32 (T, 20)   Joint angles
    {key}.mask.npy     bool    (T,)       IK failure mask (True = no failure)
    {key}.meta.json    {"session", "wstart", "wend"}

Usage:
    ds = ShardedEmgDataset(
        shard_pattern="gs://emg2posedata/shards/train/shard-*.tar.gz",
        shuffle_buffer=1000,
        training=True,
    )
    for sample in DataLoader(ds, batch_size=64, num_workers=4):
        emg = sample["emg"]          # (B, C, T)
        joints = sample["joint_angles"]  # (B, J, T)
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch
import webdataset as wds

from emg2pose.transforms import Transform

log = logging.getLogger(__name__)


class ShardedEmgDataset(torch.utils.data.IterableDataset):
    """Streaming IterableDataset over WebDataset tar shards.

    Supports local paths (``file:///path/...``) and GCS (``gs://bucket/...``).
    GCS access requires ``gcsfs`` to be installed (``pip install gcsfs``).

    Args:
        shard_pattern (str | list[str]): Glob pattern or list of shard URLs.
            Examples:
                - ``"file:///tmp/shards/train/shard-*.tar.gz"``
                - ``"gs://emg2posedata/shards/train/shard-*.tar.gz"``
                - ``["gs://bucket/train/shard-000000.tar", ...]``
        shuffle_buffer (int): Size of in-memory ring-buffer for within-shard
            sample shuffle. Set 0 to disable. (default: 1000)
        training (bool): If True, shard order is shuffled each epoch.
            Set False for val/test to ensure deterministic ordering.
        emg_transform (Transform | None): Transform applied to EMG tensor of
            shape (T, C) before transposing to (C, T). Matches existing
            ChannelDownsampling / RotationAugmentation conventions.
    """

    def __init__(
        self,
        shard_pattern: str | list[str],
        shuffle_buffer: int = 1000,
        training: bool = True,
        emg_transform: Transform[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.shard_pattern = shard_pattern
        self.shuffle_buffer = shuffle_buffer
        self.training = training
        self.emg_transform = emg_transform

    def _build_pipeline(self) -> wds.DataPipeline:
        pipeline = wds.WebDataset(
            self.shard_pattern,
            shardshuffle=self.training,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            resampled=False,
            handler=wds.warn_and_continue,
        )
        if self.shuffle_buffer > 0:
            pipeline = pipeline.shuffle(self.shuffle_buffer)
        pipeline = pipeline.decode().map(self._decode_sample)
        return pipeline

    def _decode_sample(self, sample: dict) -> dict[str, torch.Tensor]:
        """Convert a raw webdataset dict to model-ready tensors."""
        emg = torch.from_numpy(sample["emg.npy"].copy()).float()      # (T, C)
        joints = torch.from_numpy(sample["joints.npy"].copy()).float()  # (T, J)
        mask = torch.from_numpy(sample["mask.npy"].copy())              # (T,)

        if self.emg_transform is not None:
            emg = self.emg_transform(emg)
            if not torch.is_tensor(emg):
                emg = torch.as_tensor(emg)

        meta = sample.get("meta.json", {})
        if isinstance(meta, (bytes, str)):
            meta = json.loads(meta)

        return {
            "emg": emg.T,                           # (C, T)
            "joint_angles": joints.T,               # (J, T)
            "no_ik_failure": mask,                  # (T,)
            "window_start_idx": meta.get("wstart", 0),
            "window_end_idx": meta.get("wend", emg.shape[0]),
        }

    def __iter__(self):
        return iter(self._build_pipeline())
