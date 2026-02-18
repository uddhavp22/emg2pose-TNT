#!/usr/bin/env python3
"""Convert EMG2Pose HDF5 dataset to WebDataset tar shards for sequential GCS streaming.

Each shard is a .tar (or .tar.gz) file containing pre-windowed EMG + joint angle samples.
Shards are large (target 500 MB–2 GB) to enable high-throughput sequential reads from GCS.

Usage:
    python create_shards.py \
        --input_dir /teamspace/gcs_connections/emg2posedata/emg2pose_data \
        --output_dir gs://emg2posedata/shards \
        --split_config config/data_split/full_split.yaml \
        --window_length 10000 \
        --stride 10000 \
        --shard_size_mb 1024 \
        --num_workers 4

Output layout:
    {output_dir}/{split}/shard-000000.tar
    {output_dir}/{split}/shard-000001.tar
    ...

Each tar sample (key = zero-padded 12-digit index):
    {key}.emg.npy      float32 (T, 16)
    {key}.joints.npy   float32 (T, 20)
    {key}.mask.npy     bool    (T,)
    {key}.meta.json    {"session": str, "wstart": int, "wend": int}
"""

import argparse
import io
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HDF5 window extraction
# ---------------------------------------------------------------------------

def _iter_windows(
    hdf5_path: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
) -> Iterator[dict]:
    """Yield windows from a single HDF5 session file.

    Each yielded dict has keys: emg, joints, mask, session, wstart, wend.
    """
    with h5py.File(hdf5_path, "r") as f:
        group = f["emg2pose"]
        ts = group["timeseries"]
        session_length = len(ts)
        session_name = group.attrs.get("session", hdf5_path.stem)

        if skip_ik_failures:
            joint_angles_all = ts["joint_angles"][:]
            no_ik_all = get_ik_failures_mask(joint_angles_all)
            blocks = get_contiguous_ones(no_ik_all)
            blocks = [(t0, t1 - 1) for (t0, t1) in blocks if (t1 - t0) >= window_length]
        else:
            blocks = [(0, session_length)]

        for start_idx, end_idx in blocks:
            block_len = end_idx - start_idx
            num_windows = (block_len - window_length) // stride + 1
            for i in range(num_windows):
                wstart = start_idx + i * stride
                wend = wstart + window_length
                window = ts[wstart:wend]
                emg = window["emg"].astype(np.float32)          # (T, 16)
                joints = window["joint_angles"].astype(np.float32)  # (T, 20)
                if skip_ik_failures:
                    mask = no_ik_all[wstart:wend]
                else:
                    mask = get_ik_failures_mask(window["joint_angles"])
                yield {
                    "emg": emg,
                    "joints": joints,
                    "mask": mask.astype(np.bool_),
                    "session": session_name,
                    "wstart": int(wstart),
                    "wend": int(wend),
                }


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

def _estimate_bytes(window: dict) -> int:
    return window["emg"].nbytes + window["joints"].nbytes + window["mask"].nbytes


def _write_tar_shard(windows: list[dict], out_path: Path, compress: str) -> Path:
    """Write a list of window dicts to a tar shard file.

    Args:
        windows: list of window dicts (emg, joints, mask, session, wstart, wend)
        out_path: destination .tar or .tar.gz path (local)
        compress: "" for plain tar, "gz" for .tar.gz
    """
    mode = f"w:{compress}" if compress else "w:"
    with tarfile.open(out_path, mode) as tf:
        for idx, w in enumerate(windows):
            key = f"{idx:012d}"
            # EMG
            buf = io.BytesIO()
            np.save(buf, w["emg"])
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.emg.npy")
            info.size = buf.getbuffer().nbytes
            tf.addfile(info, buf)
            # Joints
            buf = io.BytesIO()
            np.save(buf, w["joints"])
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.joints.npy")
            info.size = buf.getbuffer().nbytes
            tf.addfile(info, buf)
            # Mask
            buf = io.BytesIO()
            np.save(buf, w["mask"])
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.mask.npy")
            info.size = buf.getbuffer().nbytes
            tf.addfile(info, buf)
            # Metadata JSON
            meta = json.dumps(
                {"session": w["session"], "wstart": w["wstart"], "wend": w["wend"]}
            ).encode()
            buf = io.BytesIO(meta)
            info = tarfile.TarInfo(name=f"{key}.meta.json")
            info.size = len(meta)
            tf.addfile(info, buf)
    return out_path


def _upload_batch_to_gcs(local_paths: list[Path], gcs_dest_dir: str) -> None:
    """Upload a batch of local files to a GCS directory in parallel.

    Uses ``gsutil -m cp`` so all files in the batch are uploaded concurrently
    by gsutil's internal thread pool. Far faster than one subprocess per file.
    """
    cmd = ["gsutil", "-m", "-q", "cp"] + [str(p) for p in local_paths] + [gcs_dest_dir]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Per-session worker (runs in subprocess)
# ---------------------------------------------------------------------------

def _process_session(
    hdf5_path_str: str,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    shard_size_bytes: int,
    compress: str,
    tmp_dir: str,
    worker_id: int,
) -> tuple[list[str], int]:
    """Process one HDF5 session and produce local shard files.

    Returns (list of local shard paths, total windows processed).
    The main process re-indexes and uploads the shard paths.
    """
    hdf5_path = Path(hdf5_path_str)
    shard_paths = []
    buffer: list[dict] = []
    buffer_bytes = 0
    local_shard_idx = 0
    windows_processed = 0

    for window in _iter_windows(hdf5_path, window_length, stride, skip_ik_failures):
        buffer.append(window)
        buffer_bytes += _estimate_bytes(window)
        windows_processed += 1
        if buffer_bytes >= shard_size_bytes:
            ext = f".tar.{compress}" if compress else ".tar"
            local_path = Path(tmp_dir) / f"worker{worker_id:04d}_shard{local_shard_idx:06d}{ext}"
            _write_tar_shard(buffer, local_path, compress)
            shard_paths.append(str(local_path))
            buffer = []
            buffer_bytes = 0
            local_shard_idx += 1

    if buffer:
        ext = f".tar.{compress}" if compress else ".tar"
        local_path = Path(tmp_dir) / f"worker{worker_id:04d}_shard{local_shard_idx:06d}{ext}"
        _write_tar_shard(buffer, local_path, compress)
        shard_paths.append(str(local_path))

    return shard_paths, windows_processed


# ---------------------------------------------------------------------------
# Split-level conversion
# ---------------------------------------------------------------------------

def _load_session_paths(
    input_dir: Path,
    split_config_path: Path,
    split_name: str,
) -> list[Path]:
    config = OmegaConf.load(split_config_path)

    if split_name in config:
        sessions = config[split_name]
    else:
        from emg2pose.utils import load_splits

        metadata_file_raw = OmegaConf.to_container(config, resolve=False).get("metadata_file")
        if not metadata_file_raw:
            raise ValueError(f"Cannot determine sessions for split '{split_name}'")
        metadata_file = str(metadata_file_raw).replace("${data_location}", str(input_dir))
        subsample = OmegaConf.to_container(config, resolve=False).get("subsample", 1)
        splits = load_splits(metadata_file=metadata_file, subsample=subsample)
        sessions = splits[split_name]

    paths = [input_dir / f"{s}.hdf5" for s in sessions]
    existing = [p for p in paths if p.exists()]
    missing = len(paths) - len(existing)
    if missing:
        log.warning(f"  {missing}/{len(paths)} files missing for {split_name}")
    log.info(f"  Found {len(existing)} sessions for {split_name}")
    return existing


def convert_split(
    input_dir: Path,
    output_dir: str,
    split_config_path: Path,
    split_name: str,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    shard_size_mb: int,
    compress: str,
    num_workers: int,
) -> None:
    log.info(f"Converting split: {split_name}")
    session_paths = _load_session_paths(input_dir, split_config_path, split_name)
    if not session_paths:
        log.warning(f"No sessions for {split_name}, skipping.")
        return

    shard_size_bytes = shard_size_mb * 1024 * 1024
    is_gcs = output_dir.startswith("gs://")

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Dispatch sessions to workers
        futures = {}
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for wid, sp in enumerate(session_paths):
                fut = pool.submit(
                    _process_session,
                    str(sp),
                    window_length,
                    stride,
                    skip_ik_failures,
                    shard_size_bytes,
                    compress,
                    tmp_dir,
                    wid,
                )
                futures[fut] = wid

            all_local_shards: list[str] = []
            total_windows = 0
            total_shards = 0
            with tqdm(
                total=len(futures),
                desc=f"[{split_name}] sessions",
                unit="session",
                dynamic_ncols=True,
            ) as pbar:
                for fut in as_completed(futures):
                    try:
                        shard_paths, win_count = fut.result()
                        all_local_shards.extend(shard_paths)
                        total_windows += win_count
                        total_shards += len(shard_paths)
                        pbar.set_postfix(
                            windows=f"{total_windows:,}",
                            shards=total_shards,
                            refresh=False,
                        )
                    except Exception as e:
                        log.error(f"Worker failed: {e}")
                    pbar.update(1)

        if not all_local_shards:
            log.warning(f"No shards produced for {split_name}")
            return

        # Step 1: Rename worker shards to final sequential names in-place
        ext = ".tar.gz" if compress == "gz" else (".tar.bz2" if compress == "bz2" else ".tar")
        split_output = f"{output_dir}/{split_name}"
        renamed: list[Path] = []
        for global_idx, local_path in enumerate(sorted(all_local_shards)):
            new_path = Path(tmp_dir) / f"shard-{global_idx:06d}{ext}"
            os.rename(local_path, new_path)
            renamed.append(new_path)

        # Step 2: Upload (or move) — one gsutil -m call per batch of BATCH_SIZE files
        BATCH_SIZE = 500  # stays well under ARG_MAX; gsutil -m parallelises within each batch
        log.info(
            f"  Uploading {len(renamed)} shards → {split_output} "
            f"(batches of {BATCH_SIZE}, gsutil -m parallel)"
        )
        if not is_gcs:
            Path(split_output).mkdir(parents=True, exist_ok=True)

        batches = [renamed[i : i + BATCH_SIZE] for i in range(0, len(renamed), BATCH_SIZE)]
        for batch in tqdm(batches, desc=f"[{split_name}] upload", unit="batch", dynamic_ncols=True):
            if is_gcs:
                _upload_batch_to_gcs(batch, f"{split_output}/")
            else:
                for p in batch:
                    os.rename(p, Path(split_output) / p.name)

    log.info(f"  Done: {split_name} — {len(all_local_shards)} shards")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert EMG2Pose HDF5 files to WebDataset tar shards"
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root (local path or gs://bucket/path)",
    )
    parser.add_argument(
        "--split_config",
        type=Path,
        default=Path("config/data_split/full_split.yaml"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--window_length", type=int, default=10_000)
    parser.add_argument("--stride", type=int, default=10_000)
    parser.add_argument("--skip_ik_failures", action="store_true")
    parser.add_argument(
        "--shard_size_mb",
        type=int,
        default=1024,
        help="Target shard size in MB (default: 1024 = 1 GB)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gz",
        choices=["gz", "bz2", "none"],
        help="Tar compression: gz (.tar.gz), bz2 (.tar.bz2), none (.tar). "
             "webdataset reads all formats natively. (default: gz)",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    compress = "" if args.compression == "none" else args.compression

    log.info("=" * 70)
    log.info("EMG2Pose HDF5 → WebDataset Shard Conversion")
    log.info("=" * 70)
    log.info(f"Input:       {args.input_dir}")
    log.info(f"Output:      {args.output_dir}")
    log.info(f"Splits:      {', '.join(args.splits)}")
    log.info(f"Window:      {args.window_length} samples")
    log.info(f"Stride:      {args.stride} samples")
    log.info(f"Shard size:  {args.shard_size_mb} MB")
    log.info(f"Compression: {args.compression}")
    log.info(f"Workers:     {args.num_workers}")
    log.info("=" * 70)

    for split_name in args.splits:
        convert_split(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split_config_path=args.split_config,
            split_name=split_name,
            window_length=args.window_length,
            stride=args.stride,
            skip_ik_failures=args.skip_ik_failures,
            shard_size_mb=args.shard_size_mb,
            compress=compress,
            num_workers=args.num_workers,
        )

    log.info("=" * 70)
    log.info("Conversion complete.")
    log.info("")
    log.info("Next steps:")
    log.info("  1. Train with:  python -m emg2pose.train +datamodule=sharded")
    log.info(f"  2. Set:         data_location={args.output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
