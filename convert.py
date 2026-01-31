#!/usr/bin/env python3
"""
Find best contiguous 4-channel block (out of 16) that matches a given 4-channel recording.

Assumptions:
- 4-channel CSV has columns for 4 channels (no header or with header; configurable).
- 16-channel generator yields numpy arrays shaped (T, 16), one recording at a time.
- Channels are ordered 0..15 around the band. Candidate blocks are contiguous on a ring.

Scoring strategy (default):
- Compute per-window EMG envelope features (RMS, waveform length, log-variance) for:
    a) each candidate 4-of-16 block
    b) the 4-channel CSV
- Fit ridge regression mapping block-features -> 4ch-features and score by R^2
- Average scores across recordings from the generator.

If you have true time-aligned paired data (same timestamps), you can switch to a
direct time-series correlation scoring (see score_time_aligned()).
"""

from __future__ import annotations
import os

import h5py
import numpy as np
import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Utilities / feature extraction
# -----------------------------

def _sliding_window_view_2d(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    Produce a 3D view: (num_windows, win, channels) from 2D x: (T, C)
    without copying large amounts of data.

    Works best on contiguous arrays. If x isn't contiguous, we copy once.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (T, C), got shape {x.shape}")

    x = np.ascontiguousarray(x)
    T, C = x.shape
    if T < win:
        raise ValueError(f"Input length T={T} shorter than window win={win}")

    n = 1 + (T - win) // step
    # Create strided view
    s0, s1 = x.strides
    out_shape = (n, win, C)
    out_strides = (step * s0, s0, s1)
    return np.lib.stride_tricks.as_strided(x, shape=out_shape, strides=out_strides)


def emg_window_features(
    x: np.ndarray,
    win: int,
    step: int,
    eps: float = 1e-8,
    use_rectified: bool = True,
) -> np.ndarray:
    """
    Compute robust EMG features per channel per window.

    Input: x shape (T, C)
    Output: features shape (num_windows, C * F) where F=3:
      - RMS
      - Waveform Length (WL)
      - log-variance

    Notes:
    - Rectified helps when polarity may flip between montages.
    - These are simple, fast features that often transfer well across hardware.
    """
    if use_rectified:
        x = np.abs(x)

    w = _sliding_window_view_2d(x, win=win, step=step)  # (W, win, C)
    # RMS
    rms = np.sqrt(np.mean(w * w, axis=1) + eps)  # (W, C)
    # WL: sum of absolute differences within window
    wl = np.sum(np.abs(np.diff(w, axis=1)), axis=1) + eps  # (W, C)
    # log-variance
    var = np.var(w, axis=1) + eps  # (W, C)
    logvar = np.log(var)  # (W, C)

    # Concatenate features along channel axis: [rms | wl | logvar]
    feats = np.concatenate([rms, wl, logvar], axis=1)  # (W, 3C)
    return feats


def standardize_fit_transform(
    X: np.ndarray,
    mean_: Optional[np.ndarray] = None,
    std_: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize columns: (X - mean) / std.
    If mean_/std_ not provided, compute from X.
    Returns standardized X and the mean/std used.
    """
    if mean_ is None:
        mean_ = X.mean(axis=0)
    if std_ is None:
        std_ = X.std(axis=0)
    std_ = np.maximum(std_, eps)
    return (X - mean_) / std_, mean_, std_


def ridge_regression_r2(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """
    Fit ridge regression Y ≈ X B (with intercept handled by standardization)
    and return multi-output R^2 averaged across outputs.

    X: (N, Dx)
    Y: (N, Dy)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X and Y must have same rows, got {X.shape[0]} vs {Y.shape[0]}")

    # Standardize X and Y to reduce gain/scale mismatch
    Xs, mx, sx = standardize_fit_transform(X)
    Ys, my, sy = standardize_fit_transform(Y)

    # Closed-form ridge: B = (X^T X + alpha I)^(-1) X^T Y
    XtX = Xs.T @ Xs
    Dx = XtX.shape[0]
    B = np.linalg.solve(XtX + alpha * np.eye(Dx), Xs.T @ Ys)  # (Dx, Dy)

    Yhat = Xs @ B  # (N, Dy)

    # R^2 per output in standardized space (equivalent for comparison)
    ss_res = np.sum((Ys - Yhat) ** 2, axis=0)
    ss_tot = np.sum((Ys - Ys.mean(axis=0)) ** 2, axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    return float(np.mean(r2))


# -----------------------------
# Candidate block enumeration
# -----------------------------

def contiguous_blocks_16_to_4() -> List[Tuple[int, int, int, int]]:
    """All 16 contiguous 4-channel blocks on a ring of size 16."""
    blocks = []
    for k in range(16):
        blocks.append(((k + 0) % 16, (k + 1) % 16, (k + 2) % 16, (k + 3) % 16))
    return blocks


# -----------------------------
# Core scoring loop
# -----------------------------

@dataclass
class BlockScore:
    block: Tuple[int, int, int, int]
    mean_r2: float
    std_r2: float
    n_records: int


def score_blocks_feature_map(
    gen16: Iterable[np.ndarray],
    y4: np.ndarray,
    win: int,
    step: int,
    alpha: float,
    max_records: Optional[int] = None,
) -> List[BlockScore]:
    """
    Score each contiguous 4-of-16 block by feature mapping performance to y4.

    gen16 yields arrays (T, 16)
    y4 is array (Ty, 4)

    Strategy:
    - Convert y4 to window-features Fy (Wy, 12)
    - For each 16-ch recording:
        - compute window-features for each candidate block: Fx_k (Wx, 12)
        - align #windows by truncation to min(Wx, Wy)
        - fit ridge Fx_k -> Fy and record R^2
    - Aggregate mean/std R^2 over recordings.

    Note: This does NOT require time alignment across recordings; it assumes the
    4-ch CSV reflects the same overall task distribution. If you *do* have paired,
    time-aligned data, you can improve this (see notes).
    """
    if y4.ndim != 2 or y4.shape[1] != 4:
        raise ValueError(f"Expected y4 shape (T, 4), got {y4.shape}")

    blocks = contiguous_blocks_16_to_4()

    Fy = emg_window_features(y4, win=win, step=step, use_rectified=True)  # (Wy, 12)

    # Accumulate per-block scores across generator items
    per_block_scores: Dict[Tuple[int, int, int, int], List[float]] = {b: [] for b in blocks}

    for idx, x16 in enumerate(gen16):
        if max_records is not None and idx >= max_records:
            break

        if x16.ndim != 2 or x16.shape[1] != 16:
            raise ValueError(f"Generator item {idx} must have shape (T, 16), got {x16.shape}")

        # For this recording, compute features for all 16 channels once (Wx, 48)
        Fx_all = emg_window_features(x16, win=win, step=step, use_rectified=True)
        # Fx_all layout is [rms(16) | wl(16) | logvar(16)] -> total 48 cols

        Wy = Fy.shape[0]

        for b in blocks:
            # Extract the 4 channels' features from Fx_all.
            # Need to take those indices from each of the 3 feature bands.
            ch = np.array(b, dtype=int)
            # Columns for rms are [0..15], wl are [16..31], logvar are [32..47]
            cols = np.concatenate([ch, ch + 16, ch + 32])
            Fx = Fx_all[:, cols]  # (Wx, 12)

            W = min(Fx.shape[0], Wy)
            if W < 10:
                continue  # not enough windows for a stable regression

            r2 = ridge_regression_r2(Fx[:W], Fy[:W], alpha=alpha)
            per_block_scores[b].append(r2)

    results: List[BlockScore] = []
    for b, scores in per_block_scores.items():
        if len(scores) == 0:
            results.append(BlockScore(block=b, mean_r2=float("-inf"), std_r2=float("nan"), n_records=0))
        else:
            s = np.array(scores, dtype=float)
            results.append(BlockScore(block=b, mean_r2=float(s.mean()), std_r2=float(s.std(ddof=1) if len(s) > 1 else 0.0),
                                      n_records=len(scores)))

    results.sort(key=lambda r: r.mean_r2, reverse=True)
    return results


# -----------------------------
# I/O: 4-channel CSV
# -----------------------------

def load_4ch_csv(
    path: str,
    delimiter: str = ",",
    has_header: bool = True,
    column_names: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Load CSV file(s) into numpy array with selected columns.
    
    Args:
        path: Path to either a single CSV file or a directory containing CSV files.
        delimiter: CSV delimiter (default: ",").
        has_header: Whether CSV files have a header row (default: True).
        column_names: List of column names or indices to select.
                     If None and has_header=True, uses column names from header.
                     If None and has_header=False, selects first 4 columns by index.
                     Can be a mix of column names (str) and indices (int).
    
    Returns:
        Concatenated numpy array of shape (T, num_selected_cols) from all CSV files found.
        If multiple files, they are concatenated along the time axis.
    """
    
    data_list = []
    
    if os.path.isfile(path):
        csv_files = [path]
    elif os.path.isdir(path):
        csv_files = [
            os.path.join(path, f) for f in os.listdir(path) 
            if f.endswith('.csv')
        ]
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {path}")
        csv_files.sort()  # Sort for consistent ordering
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    for csv_file in csv_files:
        if has_header:
            # Load with header to access column names
            data = np.genfromtxt(csv_file, delimiter=delimiter, skip_header=1)
            header_raw = np.genfromtxt(csv_file, delimiter=delimiter, max_rows=1, dtype=str)
            # Flatten header in case it's a 0-d or 1-d array
            header = np.atleast_1d(header_raw).flatten().tolist()
        else:
            data = np.genfromtxt(csv_file, delimiter=delimiter)
            header = None
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Determine which columns to select
        if column_names is None:
            if has_header:
                # If no columns specified but header exists, select all
                col_indices = list(range(data.shape[1]))
            else:
                # If no header and no columns specified, default to first 4
                if data.shape[1] < 4:
                    raise ValueError(f"CSV must have at least 4 columns, got {data.shape[1]} in {csv_file}")
                col_indices = [0, 1, 2, 3]
        else:
            # Convert column names/indices to indices
            col_indices = []
            for col in column_names:
                if isinstance(col, int):
                    col_indices.append(col)
                elif isinstance(col, str):
                    if header is None:
                        raise ValueError(f"Cannot use column name '{col}' when has_header=False")
                    try:
                        idx = header.index(col)
                        col_indices.append(idx)
                    except ValueError:
                        raise ValueError(f"Column '{col}' not found in {csv_file}. Available: {header}")
                else:
                    raise TypeError(f"Column must be str or int, got {type(col)}")
        
        # Validate column indices
        for idx in col_indices:
            if idx < 0 or idx >= data.shape[1]:
                raise ValueError(f"Column index {idx} out of bounds for {data.shape[1]} columns in {csv_file}")
        
        # Select columns
        y_selected = data[:, col_indices].astype(np.float32)
        data_list.append(y_selected)
    
    if not data_list:
        raise ValueError("No data loaded from CSV file(s)")
    
    return np.concatenate(data_list, axis=0)


# -----------------------------
# User-supplied generator stub
# -----------------------------

def your_16ch_generator_stub() -> Generator[np.ndarray, None, None]:
    TARGET = "/Users/zanderbaker/emg2pose_dataset_mini/"
    
    for file in os.listdir(TARGET):
        if not file.endswith(".hdf5"):
            continue
        filepath = os.path.join(TARGET, file)
        f = h5py.File(filepath, "r")
        data = f.get("emg2pose")
        emg_np = data.get("timeseries")["emg"]
        f.close()
        
        yield emg_np


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Find best contiguous 4-channel block from 16-channel EMG data.")
    p.add_argument("--csv4", required=True, help="Path to 4-channel CSV file/directory.")
    p.add_argument("--delimiter", default="\t", help="CSV delimiter (default: ,).")
    p.add_argument("--no-header", action="store_true", help="Set if CSV has no header row.")
    p.add_argument(
        "--columns",
        type=str,
        default=None,
        help="Comma-separated list of column names or indices to select (default: 0,1,2,3 or all if header exists)."
    )
    p.add_argument("--win", type=int, default=400, help="Window length in samples (default: 400).")
    p.add_argument("--step", type=int, default=200, help="Step size in samples (default: 200).")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha (default: 1.0).")
    p.add_argument("--max-records", type=int, default=None, help="Max number of 16-ch recordings to score.")
    p.add_argument("--topk", type=int, default=8, help="How many top candidates to print (default: 8).")
    args = p.parse_args()

    # Parse column selection
    column_names = None
    if args.columns:
        column_names = []
        for col in args.columns.split(','):
            col = col.strip()
            # Try to parse as integer, otherwise treat as column name
            try:
                column_names.append(int(col))
            except ValueError:
                column_names.append(col)

    y4 = load_4ch_csv(
        args.csv4,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        column_names=column_names,
    )

    # TODO: swap stub with your real generator
    gen16 = your_16ch_generator_stub()

    scores = score_blocks_feature_map(
        gen16=gen16,
        y4=y4,
        win=args.win,
        step=args.step,
        alpha=args.alpha,
        max_records=args.max_records,
    )

    print("\nTop candidates (contiguous blocks on ring):")
    for r in scores[: args.topk]:
        print(
            f"block={r.block}  mean_R2={r.mean_r2: .4f}  std={r.std_r2: .4f}  n_records={r.n_records}"
        )

    print("\nAll candidates (ranked):")
    for r in scores:
        print(
            f"block={r.block}  mean_R2={r.mean_r2: .4f}  std={r.std_r2: .4f}  n_records={r.n_records}"
        )


if __name__ == "__main__":
    main()
