"""LitData-based dataset for streaming EMG2Pose data from cloud storage."""

from pathlib import Path
from typing import Any

import torch
from litdata import StreamingDataset
from litdata.streaming.cache import Dir

from emg2pose.transforms import Transform


class LitDataEmgDataset(StreamingDataset):
    """Streaming dataset for EMG2Pose data using litdata.

    This dataset streams pre-optimized data from cloud storage (e.g., GCS)
    with automatic caching and prefetching for efficient training.

    Args:
        data_dir (str): Path to litdata optimized dataset (can be GCS path like
            gs://bucket/optimized_emg2pose/train)
        cache_dir (str): Local directory for caching chunks. Defaults to /cache/chunks
        emg_transform (Transform): Transform to apply to EMG data
        shuffle (bool): Whether to shuffle the data
        drop_last (bool): Whether to drop the last incomplete batch
        max_cache_size (int): Maximum cache size in bytes. None = unlimited
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str = "/cache/chunks",
        emg_transform: Transform[torch.Tensor, torch.Tensor] | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        max_cache_size: int | None = None,
    ):
        # Configure cache directory
        if data_dir.startswith("gs://") or data_dir.startswith("s3://"):
            # Remote data - use cache
            input_dir = Dir(path=cache_dir, url=data_dir, max_cache_size=max_cache_size)
        else:
            # Local data - no cache needed
            input_dir = data_dir

        super().__init__(
            input_dir=input_dir,
            shuffle=shuffle,
            drop_last=drop_last,
        )

        # NOTE: Do NOT use self.transform - litdata uses that internally
        self.emg_transform = emg_transform
        self.data_dir = data_dir
        self.cache_dir = cache_dir

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single window of data.

        Returns:
            dict with keys:
                - emg: Tensor of shape (C, T) - EMG signals
                - joint_angles: Tensor of shape (num_joints, T) - Joint angles
                - no_ik_failure: Tensor of shape (T,) - IK failure mask
        """
        # Load item from litdata
        item = super().__getitem__(idx)

        # Convert to tensors - EMG shape is (T, 16), joint_angles is (T, num_joints)
        emg = torch.as_tensor(item["emg"])
        joint_angles = torch.as_tensor(item["joint_angles"])
        no_ik_failure = torch.as_tensor(item["no_ik_failure"])

        # Apply transform to EMG (e.g. ChannelDownsampling)
        # Transform expects tensor of shape (T, C), same as original pipeline
        if self.emg_transform is not None:
            emg = self.emg_transform(emg)
            if not torch.is_tensor(emg):
                emg = torch.as_tensor(emg)

        # Return in the same format as WindowedEmgDataset
        return {
            "emg": emg.T,  # (T, C) → (C, T)
            "joint_angles": joint_angles.T,  # (T, J) → (J, T)
            "no_ik_failure": no_ik_failure,  # (T,)
            "window_start_idx": item.get("window_start_idx", 0),
            "window_end_idx": item.get("window_end_idx", len(emg)),
        }
