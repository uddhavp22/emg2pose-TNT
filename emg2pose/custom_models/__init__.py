"""
Custom model sub-package for emg2pose.

Each model lives in its own module. Add a new model by:
  1. Creating emg2pose/custom_models/<your_model>.py
  2. Importing your class here so it's accessible as
       emg2pose.custom_models.<YourClass>
  3. Pointing an experiment config at it:
       "_target_": "emg2pose.custom_models.<your_model>.<YourClass>"
     or via the short re-export path:
       "_target_": "emg2pose.custom_models.<YourClass>"

The NetworkInterface protocol (in base.py) documents the contract every
encoder must satisfy.
"""

from emg2pose.custom_models.base import NetworkInterface
from emg2pose.custom_models.transformer import TransformerEMGEncoder

__all__ = [
    "NetworkInterface",
    "TransformerEMGEncoder",
]
