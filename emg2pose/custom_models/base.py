"""
Shared interface contract for all custom EMG encoders.

Every encoder must satisfy the NetworkInterface protocol — either by inheriting
from nn.Module and setting the required attributes, or structurally (duck-typing).

──────────────────────────────────────────────────────────────────────────────
INTERFACE CONTRACT
──────────────────────────────────────────────────────────────────────────────

    class MyNetwork(nn.Module):
        def __init__(self, ...):
            self.left_context: int = 0   # frames consumed from left (causal lag)
            self.right_context: int = 0  # frames consumed from right (usually 0)

        def forward(self, emg: Tensor) -> Tensor:
            # Input:  (batch, in_channels, time)  — e.g. (B, 16, T)
            # Output: (batch, feature_dim, time)  — e.g. (B, 64, T')
            # T' = T - left_context - right_context

left_context / right_context
    Tell the pose module how many frames your network "eats" from each side of
    the window. Used to align predictions with targets.

    - left_context > 0: causal lag (e.g. TDS has ~1790 frames ≈ 0.9 s)
    - left_context = 0: fully online, no lag (Transformer, NeuroPose)

Input/output tensor format: BCT (batch, channels, time).
    If your model internally uses BTD (batch, time, dim), swap with .swapaxes().

feature_dim / decoder in_channels
    The decoder concatenates your features with 20 joint angles, so:
        decoder.in_channels = feature_dim + 20

    - Default YAML decoders assume feature_dim = 64 (matches TDS output).
    - If you use a different feature_dim, set decoder_params in ExperimentConfig.
"""

from __future__ import annotations

from typing import runtime_checkable, Protocol

from torch import Tensor


@runtime_checkable
class NetworkInterface(Protocol):
    """
    Structural protocol every encoder must satisfy.
    Inheritance is not required — just document and duck-type check.
    """
    left_context: int
    right_context: int

    def forward(self, emg: Tensor) -> Tensor:
        """(B, C, T) → (B, feature_dim, T')"""
        ...
