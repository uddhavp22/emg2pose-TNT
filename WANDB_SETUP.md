# Weights & Biases Integration for EMG2Pose-TNT

This document describes the complete Weights & Biases (wandb) integration for the EMG2Pose-TNT project.

## Overview

The project now includes comprehensive wandb logging that tracks:
- **All hyperparameters** from Hydra configs
- **Training, validation, and test metrics** automatically
- **Model gradients and parameters** during training
- **Learning rate schedules**
- **Model checkpoints** with automatic versioning
- **Final evaluation metrics** as run summaries

## Quick Start

### 1. Login to WandB

```bash
wandb login
```

### 2. Configure Your Run

Edit `config/base.yaml` or override via command line:

```yaml
wandb:
  enabled: True
  project: emg2pose
  entity: saarangp-ucla  # Your wandb username or team
  name: null  # Auto-generated or specify custom name
  tags: []  # Add tags for organization
```

### 3. Run Training

```bash
python -m emg2pose.train
```

## Configuration Options

All wandb settings are in `config/base.yaml` under the `wandb` key:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable wandb logging |
| `project` | `emg2pose` | WandB project name |
| `entity` | `saarangp-ucla` | WandB username or team name |
| `name` | `null` | Custom run name (auto-generated if null) |
| `tags` | `[]` | List of tags for run organization |
| `notes` | `null` | Notes about the experiment |
| `log_model` | `all` | Checkpoint logging: `all`, `best`, `last`, or `False` |
| `save_dir` | `null` | Directory to save wandb files |
| `watch_model` | `True` | Track gradients and parameters |
| `watch_log` | `all` | What to track: `gradients`, `parameters`, `all`, or `None` |
| `watch_log_freq` | `100` | How often to log gradients/parameters (steps) |
| `watch_log_graph` | `True` | Log model computational graph |

## Metrics Logged

### Training Metrics (per step)

- `train_loss` - Weighted combination of losses
- `train_mae` - Mean Absolute Error on joint angles
- `train_vel` - Angular velocity (rad/s)
- `train_acc` - Angular acceleration (rad/s²)
- `train_jerk` - Angular jerk (rad/s³)
- `train_fingertip_distance` - 3D fingertip position error (mm)
- `train_landmark_distance` - 3D landmark position error (mm)
- Per-finger metrics: `train_mae_{finger}` for each finger
- Proximal-distal metrics: `train_mae_{group}` for joint groups

### Validation Metrics (per epoch)

All the same metrics as training, prefixed with `val_` instead of `train_`

### Test Metrics (final evaluation)

All the same metrics as training, prefixed with `test_`

### Additional Metrics

- `learning_rate` - Current learning rate
- `epoch` - Current epoch number

## Command Line Overrides

You can override wandb settings from the command line:

```bash
# Custom run name and tags
python -m emg2pose.train \
  wandb.name="experiment_lstm_baseline" \
  wandb.tags="[baseline,lstm,fulldata]"

# Disable wandb
python -m emg2pose.train wandb.enabled=False

# Change project
python -m emg2pose.train \
  wandb.project="emg2pose-experiments" \
  wandb.entity="your-username"

# Disable model watching (faster training)
python -m emg2pose.train wandb.watch_model=False
```

## Example Experiments

### Baseline LSTM Model

```bash
python -m emg2pose.train \
  experiment=tracking_vemg2pose \
  wandb.name="lstm_baseline" \
  wandb.tags="[baseline,lstm,tracking]" \
  wandb.notes="Baseline LSTM model for EMG-to-pose tracking"
```

### MLP Decoder Experiment

```bash
python -m emg2pose.train \
  experiment=regression_emg2pose \
  pose_module/decoder=mlp \
  wandb.name="mlp_decoder" \
  wandb.tags="[mlp,regression,ablation]" \
  wandb.notes="Ablation study with MLP decoder"
```

### Hyperparameter Sweep

```bash
# Run with different learning rates
python -m emg2pose.train \
  optimizer.lr=0.001 \
  wandb.name="lr_0.001" \
  wandb.tags="[sweep,lr_sweep]"

python -m emg2pose.train \
  optimizer.lr=0.0001 \
  wandb.name="lr_0.0001" \
  wandb.tags="[sweep,lr_sweep]"
```

## Advanced Features

### Model Checkpoints

By default, all model checkpoints are logged to wandb as artifacts:

- Best checkpoint (based on `val_loss`)
- Last checkpoint
- All intermediate checkpoints (if `log_model: all`)

Access them in the WandB UI under the "Artifacts" tab.

### Gradient and Parameter Tracking

The integration tracks:
- Gradient histograms for all model parameters
- Parameter histograms
- Gradient norms
- Model computational graph

Disable with `wandb.watch_model=False` for faster training.

### Config Tracking

The entire Hydra configuration is automatically logged, including:
- Network architecture
- Optimizer settings
- Data augmentation transforms
- Loss weights
- All hyperparameters

View in WandB UI under "Config" tab.

## Organizing Runs

### Using Tags

Tags help organize related experiments:

```bash
# Tag by experiment type
wandb.tags="[baseline]"
wandb.tags="[ablation,no_augmentation]"
wandb.tags="[final_model,production]"

# Tag by architecture
wandb.tags="[lstm,stateful]"
wandb.tags="[mlp,stateless]"

# Tag by dataset
wandb.tags="[mini_split]"
wandb.tags="[full_split]"
```

### Using Notes

Add detailed notes for context:

```bash
wandb.notes="Testing effect of window length on tracking performance"
wandb.notes="Reproducing paper results with updated PyTorch version"
```

## Comparison and Analysis

### In WandB UI

1. **Compare Runs**: Select multiple runs and click "Compare"
2. **Create Reports**: Document findings with embedded charts
3. **Parallel Coordinates**: Find optimal hyperparameters
4. **Custom Charts**: Create visualizations for specific metrics

### Common Comparisons

- Compare different architectures (LSTM vs MLP)
- Compare loss weight configurations
- Compare data augmentation strategies
- Compare window lengths and padding settings

## Troubleshooting

### WandB Not Logging

Check that:
1. You're logged in: `wandb login`
2. `wandb.enabled=True` in config
3. Entity name is correct: `wandb.entity=your-username`

### Slow Training

If training is slow:
1. Disable model watching: `wandb.watch_model=False`
2. Reduce watch frequency: `wandb.watch_log_freq=500`
3. Log fewer checkpoints: `wandb.log_model=best`

### Authentication Issues

```bash
# Re-login
wandb login --relogin

# Use API key directly
export WANDB_API_KEY=your_key_here
```

## Integration Details

### Files Modified

1. **`emg2pose/train.py`**
   - Added WandbLogger initialization
   - Added model watching with `wandb.watch()`
   - Added final metrics logging
   - Added wandb.finish() call

2. **`emg2pose/lightning.py`**
   - Added epoch-level callbacks
   - Added learning rate logging
   - Enhanced metric logging

3. **`config/base.yaml`**
   - Added complete wandb configuration section

### Automatic Features

Thanks to PyTorch Lightning integration:
- All `self.log()` calls automatically sync to wandb
- System metrics (GPU, CPU, memory) are tracked
- Training time and epoch duration are logged
- Checkpoint callbacks work seamlessly with wandb

## Best Practices

1. **Always use tags** to organize experiments
2. **Use descriptive run names** for easy identification
3. **Add notes** for context on important runs
4. **Compare runs** regularly to track progress
5. **Create reports** to document findings
6. **Archive old runs** to keep workspace clean

## Example Workflow

```bash
# 1. Start with baseline
python -m emg2pose.train \
  experiment=tracking_vemg2pose \
  wandb.name="baseline_v1" \
  wandb.tags="[baseline,v1]"

# 2. Try different architecture
python -m emg2pose.train \
  experiment=tracking_vemg2pose \
  pose_module/decoder=mlp \
  wandb.name="mlp_decoder_v1" \
  wandb.tags="[ablation,mlp,v1]"

# 3. Hyperparameter tuning
python -m emg2pose.train \
  experiment=tracking_vemg2pose \
  optimizer.lr=0.0001 \
  loss_weights.mae=1 \
  loss_weights.fingertip_distance=0.1 \
  wandb.name="tuned_losses_v1" \
  wandb.tags="[tuning,losses,v1]"

# 4. Final model
python -m emg2pose.train \
  experiment=tracking_vemg2pose \
  # ... best hyperparameters ... \
  wandb.name="final_model_v1" \
  wandb.tags="[final,production,v1]" \
  wandb.notes="Final model for deployment"
```

## Additional Resources

- [WandB Documentation](https://docs.wandb.ai/)
- [PyTorch Lightning + WandB](https://docs.wandb.ai/guides/integrations/lightning)
- [Hydra Configuration](https://hydra.cc/docs/intro/)
