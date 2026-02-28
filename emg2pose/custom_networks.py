"""
Custom network architectures for emg2pose.

Add your own encoder classes here. Each class is then referenced by its
full dotted path in `experiments.py` via the `network_params` field:

    EXPERIMENTS["my_run"] = ExperimentConfig(
        network_params={
            "_target_": "emg2pose.custom_networks.MyNetwork",
            "in_channels": 16,
            ...
        },
    )

──────────────────────────────────────────────────────────────────────────────
INTERFACE CONTRACT
──────────────────────────────────────────────────────────────────────────────

Every custom network must satisfy this interface:

    class MyNetwork(nn.Module):
        def __init__(self, ...):
            self.left_context: int = 0   # frames consumed from left (causal lag)
            self.right_context: int = 0  # frames consumed from right (usually 0)

        def forward(self, emg: Tensor) -> Tensor:
            # Input:  (batch, in_channels, time)  — e.g. (B, 16, T)
            # Output: (batch, feature_dim, time)  — e.g. (B, 64, T')

            # T' = T - left_context - right_context
            # If left_context=0, T' == T (no temporal lag).

left_context / right_context
    These tell the pose module how many frames your network "eats" from each
    side of the window. The module uses them to align predictions with targets.

    - left_context > 0: causal lag (e.g. TDS has ~1790 frames ≈ 0.9 sec)
    - left_context = 0: fully online, no lag (Transformer, NeuroPose)

Input/output tensor format: BCT (batch, channels, time).
    If your model internally uses BTD (batch, time, dim), swap with .swapaxes().

feature_dim / decoder in_channels
    The decoder (MLP or LSTM) concatenates your features with 20 joint angles,
    so decoder.in_channels = feature_dim + 20.

    - Default YAML decoders assume feature_dim = 64 (same as TDS output).
    - If you use a different feature_dim, set decoder_params in ExperimentConfig.

──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
from typing import runtime_checkable, Protocol

import torch
import torch.nn as nn
from torch import Tensor


# ── Interface ──────────────────────────────────────────────────────────────────

@runtime_checkable
class NetworkInterface(Protocol):
    """
    Structural protocol describing what every encoder must provide.
    Not required to inherit from — just documents the expected interface.
    """
    left_context: int
    right_context: int

    def forward(self, emg: Tensor) -> Tensor:
        """(B, C, T) → (B, feature_dim, T')"""
        ...


# ── Example: Transformer EMG Encoder ──────────────────────────────────────────

class TransformerEMGEncoder(nn.Module):
    """
    Causal transformer encoder for EMG signals.

    Projects each EMG frame to `feature_dim`, runs a causal multi-head
    self-attention stack, and returns per-frame feature embeddings.

    No temporal lag (left_context = 0), so it works with any window length.

    Args:
        in_channels:    Number of EMG channels (default 16 for full band)
        feature_dim:    Output feature dimension. Must match decoder in_channels
                        minus 20 (the joint state). Default 64 matches TDS.
        num_layers:     Number of transformer encoder layers
        num_heads:      Number of attention heads (must divide feature_dim)
        ffn_dim:        Feed-forward hidden dim. Defaults to 4 * feature_dim
        dropout:        Dropout probability
        max_len:        Max sequence length for positional encoding

    Input / output:
        emg:     (B, in_channels, T)
        return:  (B, feature_dim, T)
    """

    def __init__(
        self,
        in_channels: int = 16,
        feature_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        max_len: int = 20_000,
    ):
        super().__init__()

        assert feature_dim % num_heads == 0, (
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.left_context = 0
        self.right_context = 0
        self.feature_dim = feature_dim

        ffn_dim = ffn_dim or feature_dim * 4

        # Project raw EMG channels → feature_dim
        self.input_proj = nn.Linear(in_channels, feature_dim)

        # Sinusoidal positional encoding (fixed, not learned)
        self.register_buffer("pos_enc", _sinusoidal_encoding(max_len, feature_dim))

        # Causal transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,   # (B, T, D) convention inside
            norm_first=True,    # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, emg: Tensor) -> Tensor:
        """
        Args:
            emg: (B, C, T)
        Returns:
            features: (B, feature_dim, T)
        """
        B, C, T = emg.shape

        # (B, C, T) → (B, T, C)
        x = emg.permute(0, 2, 1)

        # Project to feature_dim: (B, T, feature_dim)
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_enc[:T]  # (T, feature_dim) broadcasts over batch

        # Causal mask: each position can only attend to itself and the past
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=emg.device)

        # (B, T, feature_dim)
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # (B, T, feature_dim) → (B, feature_dim, T)
        return x.permute(0, 2, 1)


def _sinusoidal_encoding(max_len: int, d_model: int) -> Tensor:
    """Returns (max_len, d_model) sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe  # (max_len, d_model)


# ── Add your own networks below ────────────────────────────────────────────────
#
# class MyCNNEncoder(nn.Module):
#     def __init__(self, in_channels=16, feature_dim=64, ...):
#         self.left_context = 0
#         self.right_context = 0
#         ...
#
#     def forward(self, emg: Tensor) -> Tensor:
#         # emg: (B, in_channels, T) → return: (B, feature_dim, T)
#         ...
