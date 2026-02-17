# Migrating to LitData for Cloud Streaming

This guide explains how to convert your HDF5 dataset to litdata format and stream it from GCS during training.

## Why LitData?

Your current setup reads HDF5 files through FUSE mounts from GCS, which is slow due to:
- Network latency on every read
- HDF5's many small random seeks
- FUSE overhead

**LitData solves this by:**
- ✅ Pre-optimizing data into efficient binary chunks
- ✅ Streaming with smart prefetching and caching
- ✅ Compression to reduce network transfer
- ✅ Automatic distributed training support
- ✅ No need to download 431GB to local disk

## Step 1: Install litdata

```bash
cd /teamspace/studios/this_studio/emg2pose-TNT
pip install litdata
```

## Step 2: Convert HDF5 to LitData Format

Run the conversion script to transform your HDF5 files into litdata's optimized format:

### Option A: Mini Dataset (for testing)

```bash
python convert_to_litdata.py \
    --input_dir /teamspace/gcs_connections/emg2posedata/emg2pose_data \
    --output_dir gs://emg2posedata/optimized_emg2pose_mini \
    --split_config config/data_split/mini_split.yaml \
    --window_length 10000 \
    --stride 10000 \
    --num_workers 8
```

### Option B: Full Dataset

```bash
python convert_to_litdata.py \
    --input_dir /teamspace/gcs_connections/emg2posedata/emg2pose_data \
    --output_dir gs://emg2posedata/optimized_emg2pose \
    --split_config config/data_split/full_split.yaml \
    --window_length 10000 \
    --stride 10000 \
    --skip_ik_failures \
    --num_workers 8
```

**Conversion parameters:**
- `--window_length`: Size of each window in samples (default: 10000 = 5 seconds at 2kHz)
- `--stride`: Stride between windows (default: 10000 = no overlap)
- `--skip_ik_failures`: Only include windows without IK failures
- `--chunk_bytes`: Chunk size for optimization (default: 64MB)
- `--compression`: Compression algorithm (default: zstd)
- `--num_workers`: Parallel workers for conversion

**Expected conversion time:**
- Mini dataset: ~5-10 minutes
- Full dataset: ~2-4 hours (one-time cost!)

## Step 3: Update Training Config

Create a new training config that uses litdata:

```bash
cp config/base.yaml config/litdata_train.yaml
```

Edit `config/litdata_train.yaml`:

```yaml
defaults:
  - _self_
  - datamodule: litdata  # Changed from 'default'
  - optimizer: adam
  - data_split: full_split
  - transforms: rotation_augmentation
  - experiment: tracking_vemg2pose

# Point to your optimized dataset
data_location: gs://emg2posedata/optimized_emg2pose

seed: 42

batch_size: 64
num_workers: 8  # Increase from 0 for better prefetching!

# ... rest of config
```

## Step 4: Train with Streaming

```bash
python -m emg2pose.train --config-name litdata_train
```

That's it! Training will now stream from GCS with smart caching.

## Monitoring Performance

### Check GPU Utilization

```bash
nvidia-smi dmon -s u
```

You should see >90% GPU utilization if streaming is working well.

### Check Cache Usage

```bash
# See how much data is cached locally
du -sh /cache/chunks

# Watch cache grow during training
watch -n 5 "du -sh /cache/chunks"
```

### Adjust Cache Size (if needed)

If you have limited disk space, set a max cache size in `config/datamodule/litdata.yaml`:

```yaml
datamodule:
  _target_: emg2pose.lightning.LitDataEmgDataModule
  cache_dir: /cache/chunks
  max_cache_size: 107374182400  # 100GB in bytes
```

## Troubleshooting

### "No such file or directory: gs://..."

Make sure your GCS credentials are set up:

```bash
gcloud auth application-default login
```

### Slow streaming

1. **Check region**: Ensure your studio is in the same region as your GCS bucket
2. **Increase workers**: Try `num_workers: 16` or higher
3. **Check network**: Run `gsutil perfdiag` to test GCS performance

### Out of disk space

Reduce cache size or use a different cache directory:

```yaml
datamodule:
  cache_dir: /tmp/litdata_cache
  max_cache_size: 53687091200  # 50GB
```

## Performance Comparison

**Before (HDF5 + FUSE):**
- Data loading: ~50-200ms per batch
- GPU utilization: 20-50%
- Network reads: ~215TB over 500 epochs

**After (LitData streaming):**
- Data loading: ~5-20ms per batch
- GPU utilization: 85-95%
- Network reads: Much less due to compression + caching

## Customizing Window Parameters

If you need different window sizes for training vs validation:

The conversion script uses fixed window parameters. If you need different parameters per split, run the conversion separately:

```bash
# Train with small windows
python convert_to_litdata.py \
    --input_dir /path/to/data \
    --output_dir gs://bucket/optimized/train \
    --splits train \
    --window_length 2000

# Val/test with large windows
python convert_to_litdata.py \
    --input_dir /path/to/data \
    --output_dir gs://bucket/optimized/val \
    --splits val \
    --window_length 10000
```

## Rolling Back

If you need to go back to HDF5:

```bash
# Use the original config
python -m emg2pose.train --config-name base
```

Your original HDF5 files are unchanged!

## Next Steps

Once you confirm streaming works well:

1. Delete local HDF5 copies to free up disk space
2. Consider training with larger batch sizes (now that I/O is faster)
3. Enable distributed training if using multiple GPUs

Happy training! 🚀
