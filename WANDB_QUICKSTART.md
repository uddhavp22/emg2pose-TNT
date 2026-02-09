# WandB Quick Start Guide

## Setup Complete! ✓

Your EMG2Pose-TNT project now has **complete Weights & Biases integration**. All metrics, configs, and model checkpoints will be automatically logged.

## Quick Start (3 steps)

### 1. Login to WandB

```bash
wandb login
```

Enter your API key from: https://wandb.ai/authorize

### 2. Run Training

```bash
python -m emg2pose.train
```

That's it! Everything is logged automatically.

### 3. View Results

Visit: https://wandb.ai/saarangp-ucla/emg2pose

## What Gets Logged Automatically

### ✓ All Hyperparameters
- Network architecture (LSTM/MLP, layers, etc.)
- Learning rate, optimizer settings
- Batch size, window length
- Loss weights
- Data augmentation settings
- Everything from your Hydra config

### ✓ Training Metrics (every step)
- `train_loss` - Overall weighted loss
- `train_mae` - Mean absolute error on joint angles
- `train_vel`, `train_acc`, `train_jerk` - Motion smoothness
- `train_fingertip_distance` - 3D fingertip accuracy
- `train_landmark_distance` - 3D landmark accuracy
- Per-finger errors for each finger
- `learning_rate` - Current LR

### ✓ Validation Metrics (every epoch)
- All the same metrics as training, prefixed with `val_`

### ✓ Test Metrics (final evaluation)
- All the same metrics, prefixed with `test_`

### ✓ Model Artifacts
- Best checkpoint
- Last checkpoint
- All checkpoints (optional)

### ✓ Model Internals
- Gradient histograms
- Parameter distributions
- Gradient norms
- Model computational graph

## Example Commands

### Basic Training
```bash
python -m emg2pose.train
```

### With Custom Name and Tags
```bash
python -m emg2pose.train \
  wandb.name="lstm_baseline_v1" \
  wandb.tags="[baseline,lstm,v1]"
```

### Different Architecture
```bash
python -m emg2pose.train \
  experiment=regression_emg2pose \
  pose_module/decoder=mlp \
  wandb.name="mlp_decoder_test" \
  wandb.tags="[ablation,mlp]"
```

### Hyperparameter Tuning
```bash
python -m emg2pose.train \
  optimizer.lr=0.0001 \
  loss_weights.fingertip_distance=0.1 \
  wandb.name="tuned_lr_0.0001" \
  wandb.tags="[tuning,lr]"
```

### Disable WandB
```bash
python -m emg2pose.train wandb.enabled=False
```

## Configuration

All settings in `config/base.yaml`:

```yaml
wandb:
  enabled: True           # Turn on/off
  project: emg2pose       # Project name
  entity: saarangp-ucla   # Your username
  name: null              # Auto-generated or custom
  tags: []                # For organization
  log_model: all          # Checkpoint logging
  watch_model: True       # Track gradients
```

Override any setting:
```bash
python -m emg2pose.train wandb.project="my-project"
```

## Common Tasks

### Compare Multiple Runs
1. Go to https://wandb.ai/saarangp-ucla/emg2pose
2. Select multiple runs
3. Click "Compare"
4. View side-by-side metrics and configs

### Find Best Hyperparameters
1. Run multiple experiments with different hyperparameters
2. Use "Parallel Coordinates" plot
3. Identify optimal configurations

### Create Report
1. Select runs
2. Click "Create Report"
3. Add charts, notes, and findings
4. Share with team

## Troubleshooting

### Not Logged In
```bash
wandb login --relogin
```

### Wrong Project/Entity
Edit `config/base.yaml` or override:
```bash
python -m emg2pose.train \
  wandb.entity="your-username" \
  wandb.project="your-project"
```

### Slow Training
Disable model watching:
```bash
python -m emg2pose.train wandb.watch_model=False
```

## Test Your Setup

Run the test script:
```bash
python test_wandb_integration.py
```

Should see:
```
✓ Testing imports...
✓ Testing wandb configuration loading...
✓ Testing WandbLogger initialization...
✓ Testing metric logging...
✓ All tests completed!
```

## Next Steps

1. **Run your first experiment:**
   ```bash
   wandb login
   python -m emg2pose.train wandb.name="baseline_v1" wandb.tags="[baseline]"
   ```

2. **Check the results:**
   Visit https://wandb.ai/saarangp-ucla/emg2pose

3. **Try different configurations:**
   ```bash
   python -m emg2pose.train experiment=tracking_vemg2pose wandb.name="lstm_tracking"
   python -m emg2pose.train experiment=regression_emg2pose wandb.name="regression"
   ```

4. **Compare and iterate!**

## Full Documentation

See [WANDB_SETUP.md](WANDB_SETUP.md) for complete documentation including:
- Detailed configuration options
- All metrics explained
- Advanced features
- Best practices
- Example workflows

## Support

- WandB Docs: https://docs.wandb.ai/
- Issues: https://github.com/wandb/wandb/issues
- EMG2Pose Project: https://wandb.ai/saarangp-ucla/emg2pose
