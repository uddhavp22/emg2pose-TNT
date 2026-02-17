# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch

from emg2pose import utils
from emg2pose.data import WindowedEmgDataset
from emg2pose.metrics import get_default_metrics
from emg2pose.pose_modules import BasePoseModule
from hydra.utils import instantiate

from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader


log = logging.getLogger(__name__)


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        val_test_window_length: int | None = None,
        skip_ik_failures: bool = False,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.train_transforms,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.val_transforms,
                    window_length=self.val_test_window_length,
                    padding=self.padding,
                    jitter=False,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEmgDataset(
                    hdf5_path,
                    transform=self.test_transforms,
                    window_length=self.val_test_window_length,
                    padding=(0, 0),
                    jitter=False,
                    skip_ik_failures=self.skip_ik_failures,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class LitDataEmgDataModule(pl.LightningDataModule):
    """DataModule for streaming EMG2Pose data from cloud storage using litdata.

    This module is designed to stream pre-optimized litdata datasets from cloud
    storage (e.g., GCS) with automatic caching and prefetching.

    Args:
        data_location (str): Base path to litdata optimized datasets
            (e.g., gs://bucket/optimized_emg2pose)
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        cache_dir (str): Local directory for caching chunks (default: /cache/chunks)
        max_cache_size (int | None): Maximum cache size in bytes. None = unlimited
    """

    def __init__(
        self,
        data_location: str,
        batch_size: int,
        num_workers: int,
        cache_dir: str = "/cache/chunks",
        max_cache_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.data_location = data_location
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size

        # Transforms will be set by train.py
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

    def setup(self, stage: str | None = None) -> None:
        from emg2pose.litdata_dataset import LitDataEmgDataset

        # Train dataset
        train_dir = f"{self.data_location}/train"
        self.train_dataset = LitDataEmgDataset(
            data_dir=train_dir,
            cache_dir=self.cache_dir,
            emg_transform=self.train_transforms,
            shuffle=True,
            drop_last=True,
            max_cache_size=self.max_cache_size,
        )

        # Validation dataset
        val_dir = f"{self.data_location}/val"
        self.val_dataset = LitDataEmgDataset(
            data_dir=val_dir,
            cache_dir=self.cache_dir,
            emg_transform=self.val_transforms,
            shuffle=False,
            drop_last=False,
            max_cache_size=self.max_cache_size,
        )

        # Test dataset
        test_dir = f"{self.data_location}/test"
        self.test_dataset = LitDataEmgDataset(
            data_dir=test_dir,
            cache_dir=self.cache_dir,
            emg_transform=self.test_transforms,
            shuffle=False,
            drop_last=False,
            max_cache_size=self.max_cache_size,
        )

        log.info(f"Train dataset: {len(self.train_dataset)} samples")
        log.info(f"Val dataset: {len(self.val_dataset)} samples")
        log.info(f"Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            # Note: shuffle is handled by StreamingDataset
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


class Emg2PoseModule(pl.LightningModule):
    def __init__(
        self,
        network_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        provide_initial_pos: bool = False,
        loss_weights: dict[str, float] | None = None,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BasePoseModule = instantiate(network_conf, _convert_="all")
        self.provide_initial_pos = provide_initial_pos
        self.loss_weights = loss_weights or {"mae": 1}

        # TODO: add metrics to Hydra config instead
        self.metrics_list = get_default_metrics()

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward(batch, self.provide_initial_pos)

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:

        # Generate predictions
        batch["no_ik_failure"] = self.update_ik_failure_mask(batch["no_ik_failure"])
        preds, targets, no_ik_failure = self.forward(batch)

        # Compute metrics
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(preds, targets, no_ik_failure, stage))
        self.log_dict(metrics, sync_dist=True)

        # Compute loss
        loss = 0.0
        for loss_name, weight in self.loss_weights.items():
            loss += metrics[f"{stage}_{loss_name}"] * weight
        self.log(f"{stage}_loss", loss, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        return self._step(batch, stage="test")

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer_conf,
            lr_scheduler_config=self.hparams.lr_scheduler_conf,
        )

    def on_train_epoch_end(self) -> None:
        """Log additional metrics at the end of each training epoch."""
        # Log current learning rate
        opt = self.optimizers()
        if opt is not None:
            if isinstance(opt, list):
                for idx, optimizer in enumerate(opt):
                    lr = optimizer.param_groups[0]["lr"]
                    self.log(f"lr_group_{idx}", lr, sync_dist=True)
            else:
                lr = opt.param_groups[0]["lr"]
                self.log("learning_rate", lr, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Log additional metrics at the end of each validation epoch."""
        # Current epoch number
        self.log("epoch", float(self.current_epoch), sync_dist=True)

    def update_ik_failure_mask(self, no_ik_failure: torch.Tensor) -> torch.Tensor:
        """Update the mask to only include samples where there are no ik failures."""

        # Mask out samples where the initial position is zero, because state
        # initialization doesn't work under these conditions. Note that the initial
        # position is the left_context'th sample, not the 0th sample.
        mask = no_ik_failure.clone()

        if self.provide_initial_pos:
            mask[~mask[:, self.model.left_context]] = False

        if mask.sum() == 0:
            log.warning("All samples masked out due to missing initial state!")

        return mask
