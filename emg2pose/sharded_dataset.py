"""Sharded (WebDataset-style .tar.gz) IterableDataset for EMG2Pose data."""

import io
import logging
import random
import tarfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset

from emg2pose.transforms import Transform

log = logging.getLogger(__name__)


def _find_gcs_local_mount(data_location: str) -> str:
    """Resolve a gs:// path to a local GCS mount path in Lightning Studios.

    Scans /teamspace/gcs_connections/ for a mounted connection that contains a
    'shards' directory, which is the expected layout for sharded EMG2Pose data.
    """
    if not data_location.startswith("gs://"):
        return data_location

    gcs_root = Path("/teamspace/gcs_connections")
    if not gcs_root.exists():
        raise FileNotFoundError(
            f"GCS connections directory not found: {gcs_root}. "
            f"Cannot resolve gs:// path: {data_location}"
        )

    for conn_dir in sorted(gcs_root.iterdir()):
        if conn_dir.is_dir() and (conn_dir / "shards").exists():
            return str(conn_dir)

    connections = [d for d in gcs_root.iterdir() if d.is_dir()]
    if connections:
        return str(connections[0])

    raise FileNotFoundError(
        f"No GCS connection with a 'shards' directory found under {gcs_root}."
    )


class ShardedEmgDataset(IterableDataset):
    """IterableDataset that streams EMG2Pose data from .tar.gz shard files.

    Each shard is a tar.gz archive containing per-sample files:
        {id}.emg.npy      - (T, C) float32 EMG signals
        {id}.joints.npy   - (T, J) float32 joint angles
        {id}.mask.npy     - (T,) bool IK-failure mask

    Shards are distributed across DataLoader workers using strided indexing.
    Within each worker a thread pool pre-fetches ``prefetch_shards`` shards
    ahead to hide I/O latency.

    Local shard cache
    -----------------
    GCS FUSE reads are ~10–100x slower than local SSD.  When ``cache_dir`` is
    set, each shard is extracted and saved as a compact ``.npz`` file on first
    access.  Subsequent reads skip the tar.gz entirely and load the arrays
    directly — typically 9x faster than re-decompressing from the local copy
    of the tar.gz.

    Args:
        shard_dir: Directory containing shard-*.tar.gz files.
        emg_transform: Optional transform applied to the EMG tensor (T, C).
        shuffle: If True, randomise shard order each iteration (for training).
        prefetch_shards: Number of shards to load ahead in background threads.
        cache_dir: If set, shards are cached as .npz files here on first
            access. Strongly recommended when shard_dir is a GCS mount.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        emg_transform: Transform[torch.Tensor, torch.Tensor] | None = None,
        shuffle: bool = False,
        prefetch_shards: int = 4,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.shard_dir = Path(shard_dir)
        self.shard_files = sorted(self.shard_dir.glob("shard-*.tar.gz"))
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard-*.tar.gz files found in {self.shard_dir}"
            )
        self.emg_transform = emg_transform
        self.shuffle = shuffle
        self.prefetch_shards = max(1, prefetch_shards)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Shard loading: tar.gz → numpy arrays, with npz local cache.
    # ------------------------------------------------------------------
    def _read_tar_shard(self, shard_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read a .tar.gz shard and return stacked (emg, joints, mask) arrays."""
        raw = shard_path.read_bytes()
        emg_list, joints_list, mask_list = [], [], []
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
            members = tar.getmembers()
            emg_members = sorted(
                (m for m in members if m.name.endswith(".emg.npy")),
                key=lambda m: m.name,
            )
            for m in emg_members:
                sid = m.name[: -len(".emg.npy")]
                emg_list.append(np.load(io.BytesIO(tar.extractfile(f"{sid}.emg.npy").read())))
                joints_list.append(np.load(io.BytesIO(tar.extractfile(f"{sid}.joints.npy").read())))
                mask_list.append(np.load(io.BytesIO(tar.extractfile(f"{sid}.mask.npy").read())))
        return np.stack(emg_list), np.stack(joints_list), np.stack(mask_list)

    def _load_shard(self, shard_path: Path) -> list[tuple]:
        """Return a list of (emg, joints, mask) numpy tuples for one shard."""
        if self.cache_dir is not None:
            npz_path = self.cache_dir / (shard_path.stem.replace(".tar", "") + ".npz")
            if not npz_path.exists():
                # Build cache: read tar.gz → save as npz (atomic write)
                emg, joints, mask = self._read_tar_shard(shard_path)
                # np.savez appends .npz automatically, so use a .tmp.npz temp name.
                tmp = npz_path.with_name(npz_path.stem + ".tmp.npz")
                np.savez(tmp, emg=emg, joints=joints, mask=mask)
                tmp.rename(npz_path)  # atomic on POSIX; safe under concurrent workers
                return list(zip(emg, joints, mask))
            # Fast path: load directly from npz
            d = np.load(npz_path)
            return list(zip(d["emg"], d["joints"], d["mask"]))

        # No cache: read tar.gz every time
        emg, joints, mask = self._read_tar_shard(shard_path)
        return list(zip(emg, joints, mask))

    # ------------------------------------------------------------------
    # Tensor conversion + optional transform.
    # ------------------------------------------------------------------
    def _samples_to_dicts(self, samples: list[tuple]) -> list[dict]:
        out = []
        for emg, joints, mask in samples:
            emg_tensor = torch.from_numpy(np.ascontiguousarray(emg))  # (T, C)
            if self.emg_transform is not None:
                emg_tensor = self.emg_transform(emg_tensor)
                if not torch.is_tensor(emg_tensor):
                    emg_tensor = torch.as_tensor(emg_tensor)
            out.append({
                "emg": emg_tensor.T,                                    # (C, T)
                "joint_angles": torch.from_numpy(np.ascontiguousarray(joints)).T,  # (J, T)
                "no_ik_failure": torch.from_numpy(np.ascontiguousarray(mask)),     # (T,)
                "window_start_idx": 0,
                "window_end_idx": emg.shape[0],
            })
        return out

    # ------------------------------------------------------------------
    # Main iterator: thread pool prefetches next N shards concurrently.
    # ------------------------------------------------------------------
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            shard_subset = list(self.shard_files)
        else:
            shard_subset = self.shard_files[worker_info.id :: worker_info.num_workers]

        if self.shuffle:
            shard_subset = list(shard_subset)
            random.shuffle(shard_subset)

        shard_iter = iter(shard_subset)

        with ThreadPoolExecutor(max_workers=self.prefetch_shards) as pool:
            futures: deque = deque()

            # Seed the prefetch queue
            for _ in range(self.prefetch_shards):
                try:
                    futures.append(pool.submit(self._load_shard, next(shard_iter)))
                except StopIteration:
                    break

            while futures:
                raw_samples = futures.popleft().result()
                try:
                    futures.append(pool.submit(self._load_shard, next(shard_iter)))
                except StopIteration:
                    pass
                yield from self._samples_to_dicts(raw_samples)
