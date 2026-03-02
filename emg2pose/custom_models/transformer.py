"""
Causal Transformer encoder for EMG signals.

Reference in an experiment config:
    network_params={
        "_target_": "emg2pose.custom_models.transformer.TransformerEMGEncoder",
        "in_channels": 16,
        "feature_dim": 64,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
    }

Or via the short re-export path:
    "_target_": "emg2pose.custom_models.TransformerEMGEncoder"
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


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
