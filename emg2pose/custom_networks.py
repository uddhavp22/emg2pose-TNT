"""
Backward-compatibility shim.

Models have moved to emg2pose/custom_models/.
Each model now lives in its own file:

    emg2pose/custom_models/transformer.py  → TransformerEMGEncoder
    emg2pose/custom_models/base.py         → NetworkInterface

New models should be added there, not here.
Prefer pointing _target_ at the full module path:
    "_target_": "emg2pose.custom_models.transformer.TransformerEMGEncoder"

The re-exports below keep old configs (using emg2pose.custom_networks.*) working.
"""

from emg2pose.custom_models.base import NetworkInterface  # noqa: F401
from emg2pose.custom_models.transformer import TransformerEMGEncoder  # noqa: F401
