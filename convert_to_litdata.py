#!/usr/bin/env python3
"""Convert EMG2Pose HDF5 dataset to litdata optimized format for cloud streaming.

This script converts the HDF5 session files to litdata's optimized binary format,
which enables fast streaming from GCS during training.

Usage:
    python convert_to_litdata.py \
        --input_dir /teamspace/gcs_connections/emg2posedata/emg2pose_data \
        --output_dir gs://emg2posedata/optimized_emg2pose \
        --split_config config/data_split/full_split.yaml \
        --window_length 10000 \
        --stride 10000 \
        --num_workers 8
"""

import argparse
import logging
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
from litdata import optimize
from omegaconf import OmegaConf
from tqdm import tqdm

from emg2pose.data import Emg2PoseSessionData
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_windows_from_session(
    hdf5_path: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool = False,
) -> Iterator[dict]:
    """Generate windows from a single HDF5 session file.

    This follows the same logic as WindowedEmgDataset but yields individual windows.
    """
    log.info(f"Processing {hdf5_path.name}")

    with h5py.File(hdf5_path, "r") as f:
        emg2pose_group = f["emg2pose"]
        timeseries = emg2pose_group["timeseries"]
        session_length = len(timeseries)

        # Load metadata
        metadata = {key: val for key, val in emg2pose_group.attrs.items()}
        session_name = metadata.get("session", hdf5_path.stem)

        # Determine valid blocks (handling IK failures if needed)
        if skip_ik_failures:
            # Load joint angles to compute IK failure mask
            joint_angles = timeseries["joint_angles"][:]
            no_ik_failure = get_ik_failures_mask(joint_angles)

            # Get contiguous blocks without IK failures
            blocks = get_contiguous_ones(no_ik_failure)
            blocks = [
                (t0, t1 - 1)
                for (t0, t1) in blocks
                if (t1 - t0) >= window_length
            ]
        else:
            blocks = [(0, session_length)]

        # Generate windows from each block
        window_count = 0
        for start_idx, end_idx in blocks:
            block_length = end_idx - start_idx
            num_windows = (block_length - window_length) // stride + 1

            for i in range(num_windows):
                offset = start_idx + i * stride
                window_start = offset
                window_end = offset + window_length

                # Read window data
                window = timeseries[window_start:window_end]

                # Extract fields
                emg = window["emg"]  # Shape: (T, 16)
                joint_angles = window["joint_angles"]  # Shape: (T, num_joints)
                timestamps = window["time"]  # Shape: (T,)

                # Compute IK failure mask for this window
                if skip_ik_failures:
                    no_ik_failure_window = no_ik_failure[window_start:window_end]
                else:
                    no_ik_failure_window = get_ik_failures_mask(joint_angles)

                yield {
                    "emg": emg.astype(np.float32),
                    "joint_angles": joint_angles.astype(np.float32),
                    "no_ik_failure": no_ik_failure_window.astype(np.bool_),
                    "timestamps": timestamps.astype(np.float64),
                    "session_name": session_name,
                    "window_start_idx": window_start,
                    "window_end_idx": window_end,
                }

                window_count += 1

        log.info(f"  Generated {window_count} windows from {hdf5_path.name}")


def process_session_for_optimize(
    session_path: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
):
    """Process a single session and return all windows.

    This function is called by litdata.optimize() for each input session.
    """
    try:
        windows = list(generate_windows_from_session(
            session_path,
            window_length=window_length,
            stride=stride,
            skip_ik_failures=skip_ik_failures,
        ))
        return windows
    except Exception as e:
        log.error(f"Error processing {session_path}: {e}")
        return []


def load_session_paths_from_split(
    input_dir: Path,
    split_config_path: Path,
    split_name: str,
) -> list[Path]:
    """Load session paths for a specific split (train/val/test)."""

    config = OmegaConf.load(split_config_path)

    # Handle two types of split configs
    if split_name in config:
        # Direct split specification (like mini_split.yaml)
        sessions = config[split_name]
    else:
        # Need to load from metadata.csv (like full_split.yaml)
        from emg2pose.utils import load_splits

        # Get metadata_file path and manually replace ${data_location}
        metadata_file_raw = OmegaConf.to_container(config, resolve=False).get("metadata_file")
        if metadata_file_raw:
            # Manual string replacement for ${data_location}
            metadata_file = str(metadata_file_raw).replace("${data_location}", str(input_dir))
            subsample = OmegaConf.to_container(config, resolve=False).get("subsample", 1)

            splits = load_splits(
                metadata_file=metadata_file,
                subsample=subsample,
            )
            sessions = splits[split_name]
        else:
            raise ValueError(f"Cannot determine sessions for split '{split_name}'")

    # Convert session names to file paths
    session_paths = [input_dir / f"{session}.hdf5" for session in sessions]

    # Verify files exist
    existing_paths = [p for p in session_paths if p.exists()]
    missing_count = len(session_paths) - len(existing_paths)

    if missing_count > 0:
        log.warning(f"Missing {missing_count}/{len(session_paths)} files for {split_name} split")

    log.info(f"Found {len(existing_paths)} sessions for {split_name} split")
    return existing_paths


def convert_split(
    input_dir: Path,
    output_dir: str,
    split_config_path: Path,
    split_name: str,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    chunk_bytes: str,
    compression: str,
    num_workers: int,
):
    """Convert a single split (train/val/test) to litdata format."""

    log.info(f"Converting {split_name} split...")

    # Load session paths for this split
    session_paths = load_session_paths_from_split(
        input_dir=input_dir,
        split_config_path=split_config_path,
        split_name=split_name,
    )

    if not session_paths:
        log.warning(f"No sessions found for {split_name} split, skipping")
        return

    # Create partial function with fixed parameters
    from functools import partial
    process_fn = partial(
        process_session_for_optimize,
        window_length=window_length,
        stride=stride,
        skip_ik_failures=skip_ik_failures,
    )

    # Output directory for this split
    split_output_dir = f"{output_dir}/{split_name}"

    log.info(f"Optimizing to {split_output_dir}")
    log.info(f"  Window length: {window_length}")
    log.info(f"  Stride: {stride}")
    log.info(f"  Skip IK failures: {skip_ik_failures}")
    log.info(f"  Chunk size: {chunk_bytes}")
    log.info(f"  Compression: {compression}")
    log.info(f"  Processing {len(session_paths)} sessions")

    # Run optimization - inputs are the session paths
    optimize(
        fn=process_fn,
        inputs=session_paths,
        output_dir=split_output_dir,
        chunk_bytes=chunk_bytes,
        compression=compression,
        num_workers=num_workers,
    )

    log.info(f"✓ Completed {split_name} split")


def main():
    parser = argparse.ArgumentParser(
        description="Convert EMG2Pose HDF5 dataset to litdata format"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Input directory containing HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for litdata format (can be GCS path like gs://bucket/path)",
    )
    parser.add_argument(
        "--split_config",
        type=Path,
        default=Path("config/data_split/full_split.yaml"),
        help="Path to split configuration file",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to convert (default: train val test)",
    )
    parser.add_argument(
        "--window_length",
        type=int,
        default=10_000,
        help="Window length in samples (default: 10000)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=10_000,
        help="Stride between windows (default: 10000)",
    )
    parser.add_argument(
        "--skip_ik_failures",
        action="store_true",
        help="Skip windows with IK failures",
    )
    parser.add_argument(
        "--chunk_bytes",
        type=str,
        default="64MB",
        help="Chunk size for litdata optimization (default: 64MB)",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "lz4", "gzip", None],
        help="Compression algorithm (default: zstd)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for optimization (default: 8)",
    )

    args = parser.parse_args()

    log.info("=" * 80)
    log.info("EMG2Pose HDF5 → litdata Conversion")
    log.info("=" * 80)
    log.info(f"Input:  {args.input_dir}")
    log.info(f"Output: {args.output_dir}")
    log.info(f"Splits: {', '.join(args.splits)}")
    log.info("=" * 80)

    # Convert each split
    for split_name in args.splits:
        try:
            convert_split(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                split_config_path=args.split_config,
                split_name=split_name,
                window_length=args.window_length,
                stride=args.stride,
                skip_ik_failures=args.skip_ik_failures,
                chunk_bytes=args.chunk_bytes,
                compression=args.compression,
                num_workers=args.num_workers,
            )
        except Exception as e:
            log.error(f"Failed to convert {split_name} split: {e}")
            raise

    log.info("=" * 80)
    log.info("✓ Conversion complete!")
    log.info("=" * 80)
    log.info("\nNext steps:")
    log.info("1. Update your datamodule to use LitDataEmgDataset")
    log.info(f"2. Point data_location to: {args.output_dir}")
    log.info("3. Train with streaming from GCS!")


if __name__ == "__main__":
    main()
