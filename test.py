from collections.abc import KeysView

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch

f = h5py.File("/Users/zanderbaker/emg2pose_dataset_mini/2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left.hdf5", "r")

a = f.get("emg2pose")
print(a.get("timeseries").dtype.fields)
print(a.get("timeseries")["emg"].shape)