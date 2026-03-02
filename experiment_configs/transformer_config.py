"""Transformer encoder experiments for emg2pose.

Run all experiments in this file:
    python experiment_configs/transformer.py --list
    python experiment_configs/transformer.py transformer_debug
    python experiment_configs/transformer.py transformer_tracking --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import ExperimentConfig  # noqa: E402


EXPERIMENTS: dict[str, ExperimentConfig] = {

    # ── Debug run — small model, mini split, no WandB ─────────────────────────
    # feature_dim=64 matches the default decoder (in_channels=84 = 64+20),
    # so no decoder_params override is needed.

    "transformer_debug": ExperimentConfig(
        experiment="tracking_vemg2pose",
        data_split="mini_split",
        max_epochs=3,
        early_stopping_patience=999,
        wandb_enabled=False,
        network_params={
            "_target_": "emg2pose.custom_models.transformer.TransformerEMGEncoder",
            "in_channels": 16,
            "feature_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
        },
    ),

    # ── Full training run — larger model, custom decoder to match feature_dim ──
    # feature_dim=128 → decoder in_channels must be 128+20=148.

    "transformer_tracking": ExperimentConfig(
        experiment="tracking_vemg2pose",
        lr=5e-4,
        wandb_name="transformer-tracking",
        wandb_tags=["transformer", "lstm-decoder"],
        network_params={
            "_target_": "emg2pose.custom_models.transformer.TransformerEMGEncoder",
            "in_channels": 16,
            "feature_dim": 128,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
        },
        decoder_params={
            "_target_": "emg2pose.networks.SequentialLSTM",
            "in_channels": 148,   # 128 features + 20 joint state
            "out_channels": 20,
            "hidden_size": 512,
            "num_layers": 2,
            "scale": 0.01,
        },
    ),

}


# ── Standalone CLI ─────────────────────────────────────────────────────────────

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", nargs="?", help="Experiment name to run")
    parser.add_argument("--list", "-l", action="store_true", help="List experiments")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Print config only")
    args = parser.parse_args()

    if args.list or args.name is None:
        col = 28
        print(f"\n{'Name':<{col}}  Preset")
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
        print(f"Available: {', '.join(EXPERIMENTS)}")
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
    _main()
