from collections.abc import KeysView

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch

TARGET = "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left.hdf5"

f = h5py.File(f"/Users/zanderbaker/emg2pose_dataset_mini/{TARGET}", "r")

data = f.get("emg2pose")
emg_np = data.get("timeseries")["emg"]
print(emg_np.shape)