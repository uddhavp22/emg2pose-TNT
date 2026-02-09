# WandB Integration Summary for EMG2Pose-TNT

## ✓ Integration Complete!

Your project now has **comprehensive Weights & Biases logging** for all experiments.

## Files Modified

### 1. [emg2pose/train.py](emg2pose/train.py)
**Added:**
- WandbLogger initialization with full config logging
- Model watching for gradients and parameters
- Final metrics logging to wandb summary
- Automatic wandb.finish() call
- Support for enabling/disabling via config

**Key Features:**
```python
# Automatic config logging
wandb_logger = WandbLogger(
    project="emg2pose",
    entity="saarangp-ucla",
    config=config_dict,  # All Hydra config logged
    log_model="all",     # Save all checkpoints
)

# Watch model for gradients
wandb_logger.watch(module, log="all", log_freq=100)
```

### 2. [emg2pose/lightning.py](emg2pose/lightning.py)
**Added:**
- `on_train_epoch_end()` - Logs learning rate after each epoch
- `on_validation_epoch_end()` - Logs epoch number
- Enhanced metric tracking

**Automatic Logging:**
- All metrics from `_step()` are automatically synced to wandb
- Learning rate scheduling tracked
- Epoch timing and progress

### 3. [config/base.yaml](config/base.yaml)
**Added complete wandb configuration section:**
```yaml
wandb:
  enabled: True
  project: emg2pose
  entity: saarangp-ucla
  name: null
  tags: []
  notes: null
  log_model: all
  save_dir: null
  watch_model: True
  watch_log: all
  watch_log_freq: 100
  watch_log_graph: True
```

### 4. [environment.yml](environment.yml)
**Added:**
- `wandb>=0.15.0` to pip dependencies

### 5. New Documentation Files

#### [WANDB_QUICKSTART.md](WANDB_QUICKSTART.md)
- 3-step quick start guide
- Common commands and examples
- Quick reference for configuration
- Troubleshooting tips

#### [WANDB_SETUP.md](WANDB_SETUP.md)
- Complete documentation (2000+ words)
- All features explained in detail
- Configuration options reference
- Best practices and workflows
- Advanced features guide

#### [test_wandb_integration.py](test_wandb_integration.py)
- Automated test script
- Verifies all components work
- Runs in offline mode for safety
- Easy to run: `python test_wandb_integration.py`

## Metrics Automatically Logged

### Training (every step)
- `train_loss` - Weighted combination loss
- `train_mae` - Mean absolute error (radians)
- `train_vel` - Angular velocity (rad/s)
- `train_acc` - Angular acceleration (rad/s²)
- `train_jerk` - Angular jerk (rad/s³)
- `train_fingertip_distance` - 3D fingertip error (mm)
- `train_landmark_distance` - 3D landmark error (mm)
- `train_mae_{finger}` - Per-finger errors (thumb, index, middle, ring, pinky)
- `train_mae_{group}` - Proximal/distal grouped errors
- `learning_rate` - Current learning rate

### Validation (every epoch)
- All training metrics with `val_` prefix

### Test (final evaluation)
- All training metrics with `test_` prefix
- Final metrics saved to run summary

### Model Internals (optional, configurable)
- Gradient histograms for all layers
- Parameter distributions
- Gradient norms
- Model computational graph

## Configuration Logged

**Everything from Hydra config is automatically logged:**
- Network architecture (pose_module, decoder, network)
- Optimizer settings (type, lr, weight decay, etc.)
- Data settings (window_length, batch_size, padding)
- Loss weights (mae, fingertip_distance, etc.)
- Trainer settings (max_epochs, gradient_clip_val)
- Transform/augmentation settings
- Data split configuration
- Random seed for reproducibility

## Usage Examples

### Basic Training
```bash
# Login once
wandb login

# Run training - everything logged automatically
python -m emg2pose.train
```

### Named Experiment with Tags
```bash
python -m emg2pose.train \
  wandb.name="lstm_baseline_v1" \
  wandb.tags="[baseline,lstm,full_data]" \
  wandb.notes="Baseline LSTM with full dataset"
```

### Different Architecture
```bash
python -m emg2pose.train \
  experiment=regression_emg2pose \
  pose_module/decoder=mlp \
  wandb.name="mlp_decoder_ablation" \
  wandb.tags="[ablation,mlp]"
```

### Hyperparameter Sweep
```bash
# Try different learning rates
for lr in 0.001 0.0005 0.0001; do
  python -m emg2pose.train \
    optimizer.lr=$lr \
    wandb.name="lr_sweep_${lr}" \
    wandb.tags="[sweep,lr]"
done
```

### Disable WandB (for debugging)
```bash
python -m emg2pose.train wandb.enabled=False
```

## Testing

Run the integration test:
```bash
python test_wandb_integration.py
```

Expected output:
```
✓ Testing imports...
✓ Testing wandb configuration loading...
✓ Testing WandbLogger initialization...
✓ Testing metric logging...
✓ All tests completed!
```

## Features Included

### ✓ Automatic Logging
- No code changes needed in model code
- PyTorch Lightning handles everything
- All `self.log()` calls sync to wandb

### ✓ Checkpoint Management
- Best model saved automatically
- Last checkpoint saved
- All checkpoints uploaded (configurable)
- Easy model recovery

### ✓ Hyperparameter Tracking
- Full config logged as nested dict
- Easy comparison between runs
- Parallel coordinates visualization

### ✓ Model Watching
- Gradient tracking (configurable frequency)
- Parameter distributions
- Helps debug training issues

### ✓ Flexible Configuration
- Enable/disable via config
- Override from command line
- Per-experiment customization

### ✓ Organization
- Tags for grouping experiments
- Custom run names
- Notes for context
- Project-level organization

## View Your Results

After running training, view at:
**https://wandb.ai/saarangp-ucla/emg2pose**

## Comparison with Old Setup

| Feature | Before | After |
|---------|--------|-------|
| Logging | Local files only | Cloud + Local |
| Visualization | Manual plotting | Automatic dashboards |
| Hyperparameters | In filename/logs | Structured in wandb |
| Model checkpoints | Local only | Versioned in cloud |
| Comparison | Manual | Built-in tools |
| Sharing | Send files | Share link |
| Metrics | Limited | 20+ metrics |
| Gradients | Not tracked | Tracked automatically |

## Next Steps

1. **Login:**
   ```bash
   wandb login
   ```

2. **Run first experiment:**
   ```bash
   python -m emg2pose.train \
     wandb.name="baseline_v1" \
     wandb.tags="[baseline]"
   ```

3. **Check results:**
   Visit https://wandb.ai/saarangp-ucla/emg2pose

4. **Read docs:**
   - Quick start: [WANDB_QUICKSTART.md](WANDB_QUICKSTART.md)
   - Full docs: [WANDB_SETUP.md](WANDB_SETUP.md)

5. **Start experimenting!**

## Support

- **Test Script:** `python test_wandb_integration.py`
- **WandB Docs:** https://docs.wandb.ai/
- **PyTorch Lightning + WandB:** https://docs.wandb.ai/guides/integrations/lightning

## Summary

✅ Complete wandb integration
✅ All metrics logged automatically
✅ All configs logged automatically
✅ Model checkpoints versioned
✅ Gradients and parameters tracked
✅ Easy to use - just run `python -m emg2pose.train`
✅ Easy to customize via config overrides
✅ Comprehensive documentation provided
✅ Test script for verification

**Everything is ready to go! Just login and start training.** 🚀
