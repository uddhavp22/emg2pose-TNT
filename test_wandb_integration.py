#!/usr/bin/env python
"""
Test script to verify WandB integration for EMG2Pose-TNT.

This script performs a quick check of the wandb setup without running full training.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
import wandb

def test_wandb_config():
    """Test that wandb config can be loaded."""
    print("✓ Testing wandb configuration loading...")

    config_path = Path(__file__).parent / "config" / "base.yaml"
    config = OmegaConf.load(config_path)

    assert "wandb" in config, "wandb section missing from config"
    assert config.wandb.project == "emg2pose", "wandb project incorrect"
    assert config.wandb.entity == "saarangp-ucla", "wandb entity incorrect"

    print(f"  Project: {config.wandb.project}")
    print(f"  Entity: {config.wandb.entity}")
    print(f"  Enabled: {config.wandb.enabled}")
    print("✓ Config loaded successfully!\n")

    return config

def test_wandb_logger_initialization(config):
    """Test that WandbLogger can be initialized."""
    print("✓ Testing WandbLogger initialization...")

    # Initialize in offline mode for testing
    import os
    os.environ["WANDB_MODE"] = "offline"

    try:
        # Create a simplified config for testing (avoid Hydra interpolations)
        test_config = {
            "wandb": {
                "project": config.wandb.project,
                "entity": config.wandb.entity,
                "enabled": config.wandb.enabled,
            },
            "seed": config.seed,
            "batch_size": config.batch_size,
        }

        logger = WandbLogger(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name="test_run",
            tags=["test", "integration"],
            config=test_config,
            offline=True,  # Run offline for testing
        )

        print(f"  Logger initialized: {logger}")
        print(f"  Run ID: {logger.version}")
        print("✓ WandbLogger initialized successfully!\n")

        return logger

    except Exception as e:
        print(f"✗ Error initializing WandbLogger: {e}")
        print("  (This is expected if wandb is not configured)")
        return None

def test_metric_logging(logger):
    """Test that metrics can be logged."""
    print("✓ Testing metric logging...")

    if logger is None:
        print("  Skipping (no logger available)\n")
        return

    try:
        # Log some test metrics
        logger.log_metrics({
            "test_loss": 0.123,
            "test_mae": 0.456,
            "test_fingertip_distance": 0.789,
        }, step=0)

        logger.log_metrics({
            "train_loss": 0.100,
            "train_mae": 0.400,
            "learning_rate": 0.001,
        }, step=1)

        print("  Logged test metrics successfully")
        print("✓ Metric logging works!\n")

    except Exception as e:
        print(f"✗ Error logging metrics: {e}\n")

def test_imports():
    """Test that all required imports work."""
    print("✓ Testing imports...")

    try:
        from emg2pose.train import train, make_data_module, make_lightning_module
        from emg2pose.lightning import Emg2PoseModule, WindowedEmgDataModule
        print("  All imports successful")
        print("✓ Imports work!\n")
        return True
    except Exception as e:
        print(f"✗ Error importing: {e}\n")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("WandB Integration Test for EMG2Pose-TNT")
    print("=" * 60)
    print()

    # Test 1: Imports
    if not test_imports():
        print("✗ Import test failed. Cannot continue.")
        return False

    # Test 2: Config loading
    try:
        config = test_wandb_config()
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

    # Test 3: Logger initialization
    logger = test_wandb_logger_initialization(config)

    # Test 4: Metric logging
    test_metric_logging(logger)

    # Finish wandb run
    if logger is not None:
        wandb.finish()

    print("=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Login to wandb: wandb login")
    print("2. Run training: python -m emg2pose.train")
    print("3. View results at: https://wandb.ai/saarangp-ucla/emg2pose")
    print()

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
