"""
Centralized experiment registry for emg2pose.

Define your experiments here as typed Python objects, then run them with:

    python experiments.py <name>
    python experiments.py --list
    python experiments.py <name> --dry-run

Or import and run programmatically:

    from experiments import EXPERIMENTS
    EXPERIMENTS["my_run"].run()
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ── Types ──────────────────────────────────────────────────────────────────────

ExperimentPreset = Literal[
    "tracking_vemg2pose",   # TDS + LSTM + velocity + initial pos  (best model)
    "tracking_emg2pose",    # TDS + MLP  + absolute pos + initial pos
    "regression_emg2pose",  # TDS + MLP  + absolute pos, no initial pos
    "regression_vemg2pose", # TDS + velocity, no initial pos
    "regression_neuropose", # NeuroPose U-Net baseline
]

DatamodulePreset = Literal[
    "default",   # HDF5-based, reads .hdf5 files from data_location
    "litdata",   # Streaming via LitData
    "sharded",   # Streaming via pre-sharded files (fastest for large datasets)
]

DataSplitPreset = Literal[
    "full_split",  # Full dataset across many subjects
    "mini_split",  # 1 subject, 3 recordings — great for fast debugging
]

TransformsPreset = Literal[
    "basic",                          # Just EMG → tensor
    "rotation_augmentation",          # + random wrist rotation aug (train only)
    "channel_downsampling",           # + 2x channel downsampling
    "litdata_channel_downsampling",   # channel downsampling for litdata
    "sharded_rotation_augmentation",  # rotation aug for sharded datamodule
]

# ── Config Model ───────────────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    All knobs you'd ever want to turn for an emg2pose experiment.
    Unset fields inherit from the experiment preset's defaults.
    """

    # ── Algorithm ──────────────────────────────────────────────────────────────
    experiment: ExperimentPreset = "tracking_vemg2pose"

    # ── Custom network (bypasses YAML) ─────────────────────────────────────────
    # Leave both as None to use the experiment preset's default network.
    #
    # network_params: dict that fully describes the encoder. Must include
    #   "_target_" pointing to your class (e.g. "emg2pose.custom_networks.MyNet").
    #   All other keys are passed as __init__ kwargs.
    #   Write your class in emg2pose/custom_networks.py.
    #
    # decoder_params: same format for the decoder head. Only needed when your
    #   network's feature_dim differs from the default 64, because the default
    #   decoder YAML has in_channels=84 (64 features + 20 joint state) hardcoded.
    #   New in_channels = your feature_dim + 20.
    #
    # Example — swap in a Transformer encoder with feature_dim=128:
    #
    #   network_params={
    #       "_target_": "emg2pose.custom_networks.TransformerEMGEncoder",
    #       "in_channels": 16,
    #       "feature_dim": 128,
    #       "num_layers": 4,
    #       "num_heads": 4,
    #   },
    #   decoder_params={
    #       "_target_": "emg2pose.networks.SequentialLSTM",
    #       "in_channels": 148,   # 128 + 20
    #       "out_channels": 20,
    #       "hidden_size": 512,
    #       "num_layers": 2,
    #       "scale": 0.01,
    #   },
    network_params: Optional[dict[str, Any]] = None
    decoder_params: Optional[dict[str, Any]] = None

    # ── Data ───────────────────────────────────────────────────────────────────
    datamodule: DatamodulePreset = "sharded"
    data_split: DataSplitPreset = "full_split"
    transforms: TransformsPreset = "basic"
    data_location: str = "~/emg2pose_data"

    # ── Training ───────────────────────────────────────────────────────────────
    batch_size: int = 64
    num_workers: int = 4
    seed: int = 42
    max_epochs: int = 500
    # Set to override the experiment preset's lr
    lr: Optional[float] = None
    # Set to override the experiment preset's gradient clipping
    gradient_clip_val: Optional[float] = None
    # Early stopping patience (epochs)
    early_stopping_patience: int = 50

    # ── Loss weights ───────────────────────────────────────────────────────────
    mae_weight: float = 1.0
    fingertip_distance_weight: float = 0.01

    # ── Flags ──────────────────────────────────────────────────────────────────
    train: bool = True
    eval: bool = True
    # Path to a checkpoint to resume from (or eval-only)
    checkpoint: Optional[str] = None

    # ── WandB ──────────────────────────────────────────────────────────────────
    wandb_enabled: bool = True
    wandb_name: Optional[str] = None
    wandb_tags: list[str] = Field(default_factory=list)
    wandb_notes: Optional[str] = None
    # "best", "all", "last", or False
    wandb_log_model: str = "best"

    # ── Derived helpers ────────────────────────────────────────────────────────

    def to_hydra_overrides(self) -> list[str]:
        """Convert to a flat list of Hydra CLI override strings."""
        ov: list[str] = [
            f"experiment={self.experiment}",
            f"datamodule={self.datamodule}",
            f"data_split={self.data_split}",
            f"transforms={self.transforms}",
            f"data_location={self.data_location}",
            f"batch_size={self.batch_size}",
            f"num_workers={self.num_workers}",
            f"seed={self.seed}",
            f"trainer.max_epochs={self.max_epochs}",
            f"callbacks[2].patience={self.early_stopping_patience}",
            f"loss_weights.mae={self.mae_weight}",
            f"loss_weights.fingertip_distance={self.fingertip_distance_weight}",
            f"train={str(self.train).lower()}",
            f"eval={str(self.eval).lower()}",
            # WandB
            f"wandb.enabled={str(self.wandb_enabled).lower()}",
            f"wandb.log_model={self.wandb_log_model}",
        ]

        if self.lr is not None:
            ov.append(f"optimizer.lr={self.lr}")
        if self.gradient_clip_val is not None:
            ov.append(f"trainer.gradient_clip_val={self.gradient_clip_val}")
        if self.checkpoint is not None:
            ov.append(f"checkpoint={self.checkpoint}")
        if self.wandb_name is not None:
            ov.append(f"wandb.name={self.wandb_name}")
        if self.wandb_tags:
            tag_str = ",".join(self.wandb_tags)
            ov.append(f"wandb.tags=[{tag_str}]")
        if self.wandb_notes is not None:
            ov.append(f'wandb.notes="{self.wandb_notes}"')

        return ov

    def run(self) -> None:
        """Build a Hydra config and call train() directly (no subprocess)."""
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf, open_dict
        from emg2pose.train import train

        GlobalHydra.instance().clear()
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
        overrides = self.to_hydra_overrides()

        label = self.wandb_name or self.experiment
        print(f"\n=== Running experiment: {label} ===")
        print("Hydra overrides:")
        for o in overrides:
            print(f"  {o}")

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(config_name="base", overrides=overrides)

            # Patch network and/or decoder with inline Python dicts.
            # This replaces whatever the experiment YAML loaded from config/network/.
            if self.network_params is not None:
                if "_target_" not in self.network_params:
                    raise ValueError(
                        "network_params must include '_target_' — "
                        "e.g. 'emg2pose.custom_networks.TransformerEMGEncoder'"
                    )
                print(f"\n  [custom network] {self.network_params['_target_']}")
                with open_dict(cfg):
                    cfg.network = OmegaConf.create(self.network_params)

            if self.decoder_params is not None:
                if "_target_" not in self.decoder_params:
                    raise ValueError(
                        "decoder_params must include '_target_' — "
                        "e.g. 'emg2pose.networks.SequentialLSTM'"
                    )
                print(f"  [custom decoder] {self.decoder_params['_target_']}")
                with open_dict(cfg):
                    cfg.pose_module.decoder = OmegaConf.create(self.decoder_params)

            print()
            train(cfg)


# ── Experiment Registry ────────────────────────────────────────────────────────
#
# Add your experiments here. Each key is the name you pass on the CLI.
# Inherit from a base config or start fresh — all fields have sensible defaults.
#
# Example:
#   EXPERIMENTS["my_run"].run()
#   $ python experiments.py my_run

EXPERIMENTS: dict[str, ExperimentConfig] = {

    # ── Paper baselines ────────────────────────────────────────────────────────

    "baseline_tracking": ExperimentConfig(
        experiment="tracking_vemg2pose",
        wandb_name="baseline-tracking",
        wandb_tags=["baseline", "lstm", "velocity"],
        wandb_notes="Reproduce paper: TDS encoder + LSTM decoder + velocity prediction",
    ),

    "baseline_regression": ExperimentConfig(
        experiment="regression_emg2pose",
        wandb_name="baseline-regression",
        wandb_tags=["baseline", "mlp"],
        wandb_notes="TDS + MLP decoder, absolute pose prediction",
    ),

    "baseline_neuropose": ExperimentConfig(
        experiment="regression_neuropose",
        wandb_name="baseline-neuropose",
        wandb_tags=["baseline", "neuropose", "unet"],
        wandb_notes="NeuroPose U-Net architecture baseline",
    ),

    # ── Debug / fast iteration ─────────────────────────────────────────────────

    "debug": ExperimentConfig(
        experiment="tracking_vemg2pose",
        data_split="mini_split",
        max_epochs=3,
        early_stopping_patience=999,  # disable early stopping
        wandb_enabled=False,
        wandb_name="debug",
        wandb_tags=["debug"],
    ),

    "debug_neuropose": ExperimentConfig(
        experiment="regression_neuropose",
        data_split="mini_split",
        max_epochs=3,
        early_stopping_patience=999,
        wandb_enabled=False,
    ),

    # ── Custom architecture examples ───────────────────────────────────────────
    # These show how to use network_params / decoder_params to define your own
    # architectures without touching any YAML files.

    # Transformer encoder — debug run (mini split, no wandb)
    "transformer_debug": ExperimentConfig(
        experiment="tracking_vemg2pose",
        data_split="mini_split",
        max_epochs=3,
        early_stopping_patience=999,
        wandb_enabled=False,
        network_params={
            "_target_": "emg2pose.custom_networks.TransformerEMGEncoder",
            "in_channels": 16,
            "feature_dim": 64,   # matches default decoder (no decoder_params needed)
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
        },
        # feature_dim=64 → in_channels for decoder = 64+20 = 84, which is the
        # YAML default, so no decoder_params needed here.
    ),

    # Transformer encoder — full training run
    "transformer_tracking": ExperimentConfig(
        experiment="tracking_vemg2pose",
        lr=5e-4,
        wandb_name="transformer-tracking",
        wandb_tags=["custom", "transformer", "lstm-decoder"],
        network_params={
            "_target_": "emg2pose.custom_networks.TransformerEMGEncoder",
            "in_channels": 16,
            "feature_dim": 128,     # larger → must set decoder_params
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
        },
        decoder_params={
            "_target_": "emg2pose.networks.SequentialLSTM",
            "in_channels": 148,     # 128 features + 20 joint state
            "out_channels": 20,
            "hidden_size": 512,
            "num_layers": 2,
            "scale": 0.01,
        },
    ),

    # ── Your custom experiments ────────────────────────────────────────────────
    # Add new ones below. Copy an existing entry and tweak what you want.

    "example_custom": ExperimentConfig(
        experiment="tracking_vemg2pose",
        lr=5e-4,                    # override default lr
        batch_size=128,
        max_epochs=200,
        wandb_name="tracking-highbatch-tuned",
        wandb_tags=["custom", "tuned"],
        wandb_notes="Bigger batch + tuned lr to see if faster convergence",
    ),

    "example_no_aug": ExperimentConfig(
        experiment="tracking_vemg2pose",
        transforms="basic",         # no rotation augmentation
        wandb_name="tracking-no-aug",
        wandb_tags=["ablation", "no-aug"],
        wandb_notes="Ablation: disable rotation augmentation",
    ),

    "example_eval_only": ExperimentConfig(
        experiment="tracking_vemg2pose",
        train=False,
        eval=True,
        checkpoint="/path/to/your/checkpoint.ckpt",  # <- set this
        wandb_enabled=False,
    ),
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run emg2pose experiments from the Python registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments.py --list
  python experiments.py debug
  python experiments.py baseline_tracking
  python experiments.py baseline_tracking --dry-run
        """,
    )
    parser.add_argument("name", nargs="?", help="Experiment name to run")
    parser.add_argument("--list", "-l", action="store_true", help="List all registered experiments")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print config without running")
    args = parser.parse_args()

    if args.list or args.name is None:
        col = 28
        print(f"\n{'Name':<{col}}  Experiment preset")
        print("-" * (col + 30))
        for name, cfg in EXPERIMENTS.items():
            tag = f"[{', '.join(cfg.wandb_tags)}]" if cfg.wandb_tags else ""
            print(f"  {name:<{col - 2}}  {cfg.experiment}  {tag}")
        print()
        if args.name is None and not args.list:
            parser.print_help()
        sys.exit(0)

    if args.name not in EXPERIMENTS:
        print(f"Error: unknown experiment '{args.name}'")
        print(f"Run `python experiments.py --list` to see available options.")
        sys.exit(1)

    cfg = EXPERIMENTS[args.name]

    if args.dry_run:
        print(f"\nConfig for '{args.name}':")
        for k, v in cfg.model_dump().items():
            if v is not None and v != [] and v is not False or k in ("train", "eval", "wandb_enabled"):
                print(f"  {k}: {v}")
        print("\nHydra overrides:")
        for o in cfg.to_hydra_overrides():
            print(f"  {o}")
        if cfg.network_params:
            print(f"\n  [post-compose] cfg.network  ← {cfg.network_params}")
        if cfg.decoder_params:
            print(f"  [post-compose] cfg.pose_module.decoder ← {cfg.decoder_params}")
        sys.exit(0)

    cfg.run()


if __name__ == "__main__":
    main()
