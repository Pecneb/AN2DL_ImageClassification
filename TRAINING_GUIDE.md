# Updated Training Workflow

This guide explains the new preprocessing + training workflow for the breast cancer classification project.

## Overview

The workflow is now split into two stages:
1. **Preprocessing** (run once) - Apply masks, remove artifacts, crop to ROI
2. **Training** (run many times) - Train on preprocessed images

## Step 1: Preprocess Dataset

Run the preprocessing script to clean and prepare your images:

```bash
python preprocess_dataset.py \
    --input-dir data/train_data \
    --output-dir data/train_data_preprocessed \
    --csv-path data/train_labels.csv \
    --padding 20
```

**What this does:**
- ✅ Detects artifacts (green/orange/brown/white markers)
- ✅ Applies tissue masks
- ✅ Removes detected artifacts from masks
- ✅ Crops images to ROI bounding box (+ padding)
- ✅ Saves cleaned RGB images
- ✅ Creates updated CSV with failed images removed

**Output:**
- `data/train_data_preprocessed/` - Preprocessed images
- `data/train_data_preprocessed/train_labels.csv` - Updated labels

## Step 2: Train Model

Train on the preprocessed images:

### Basic Training (Single Split)
```bash
python train_effnet_mask.py \
    --train-csv data/train_data_preprocessed/train_labels.csv \
    --img-dir data/train_data_preprocessed \
    --batch-size 16 \
    --epochs 50 \
    --lr 1e-3 \
    --optimizer adamw
```

### Recommended Settings for 600 Images
```bash
python train_effnet_mask.py \
    --train-csv data/train_data_preprocessed/train_labels.csv \
    --img-dir data/train_data_preprocessed \
    --batch-size 16 \
    --epochs 50 \
    --patience 10 \
    --lr 1e-3 \
    --optimizer lion \
    --freeze-backbone-epochs 5 \
    --use-weighted-sampler \
    --warmup-epochs 3
```

### Cross-Validation
```bash
python train_effnet_mask.py \
    --train-csv data/train_data_preprocessed/train_labels.csv \
    --img-dir data/train_data_preprocessed \
    --use-cv \
    --n-splits 5 \
    --batch-size 16 \
    --epochs 30 \
    --optimizer ranger
```

## Key Changes from Previous Version

### What Changed:

1. **Architecture**: 4-channel (RGB + mask) → **3-channel (RGB only)**
   - Simpler, standard architecture
   - Mask information preserved through preprocessing (crop to ROI)

2. **Preprocessing**: Runtime → **Offline (run once)**
   - Faster training iterations
   - Consistent data across experiments
   - No mask loading during training

3. **DataLoader**: Complex mask application → **Simple image loading**
   - Standard PyTorch transforms (Resize, ToTensor, Normalize)
   - No manual tensor conversion
   - Faster data loading

4. **Images**: Variable size with background → **Cropped to ROI**
   - No wasted computation on background pixels
   - Better information density
   - Scale normalization

### Why These Changes?

**For 600 images:**
- ✅ More efficient use of limited data (every pixel is tissue)
- ✅ Faster training (smaller effective images)
- ✅ Better generalization (no positional bias)
- ✅ Simpler architecture (3-channel standard)
- ✅ Faster iteration (preprocess once, train many times)

## Optimizer Choices

### AdamW (Default)
```bash
--optimizer adamw --lr 1e-3
```
- General purpose, reliable
- Good baseline

### Ranger (RAdam + Lookahead)
```bash
--optimizer ranger --lr 1e-3
```
- Better convergence, flatter minima
- More stable training
- Good for noisy gradients

### Lion (EvoLved Sign Momentum) ⭐ RECOMMENDED
```bash
--optimizer lion --lr 1e-3
```
- **Best for small datasets** (your 600 images!)
- Memory efficient (50% less than Adam)
- Stronger regularization
- Auto-adjusts LR (divides by 3 internally)

## TensorBoard

View training progress and validation predictions:

```bash
tensorboard --logdir tb_logs
```

Navigate to http://localhost:6006 and check:
- **SCALARS**: Loss, accuracy, F1 score curves
- **IMAGES**: Validation predictions with labels (green=correct, red=wrong)

## Example Complete Workflow

```bash
# 1. Preprocess (once)
python preprocess_dataset.py \
    --input-dir data/train_data \
    --output-dir data/train_data_preprocessed \
    --csv-path data/train_labels.csv

# 2. Train with Lion optimizer (recommended)
python train_effnet_mask.py \
    --train-csv data/train_data_preprocessed/train_labels.csv \
    --img-dir data/train_data_preprocessed \
    --batch-size 16 \
    --epochs 50 \
    --patience 10 \
    --lr 1e-3 \
    --optimizer lion \
    --freeze-backbone-epochs 5 \
    --use-weighted-sampler

# 3. View results
tensorboard --logdir tb_logs
```

## Preprocessing Options

```bash
python preprocess_dataset.py --help
```

**Key options:**
- `--padding`: Pixels to add around ROI (default: 20)
- `--min-tissue-ratio`: Minimum tissue fraction to keep image (default: 0.001)

## Training Tips

1. **Start with Lion optimizer** - best for small datasets
2. **Use batch size 16** - 37-38 gradient updates/epoch for 600 images
3. **Enable weighted sampler** - handles class imbalance
4. **Freeze backbone initially** - prevents overfitting (5-10 epochs)
5. **Patience 10** - give model time to converge
6. **Monitor TensorBoard** - watch validation images for insights

## File Structure

```
AN2DL_ImageClassification/
├── preprocess_dataset.py          # New: Offline preprocessing
├── train_effnet_mask.py            # Updated: 3-channel training
├── data/
│   ├── train_data/                 # Original images + masks
│   │   ├── img_0001.png
│   │   ├── mask_0001.png
│   │   └── ...
│   ├── train_labels.csv
│   └── train_data_preprocessed/    # New: Preprocessed images
│       ├── img_0001.png            # Masked + cropped RGB
│       ├── train_labels.csv        # Updated labels
│       └── ...
└── tb_logs/                        # TensorBoard logs
```

## Quick Start

```bash
# One-time preprocessing
python preprocess_dataset.py \
    --input-dir data/train_data \
    --output-dir data/train_data_preprocessed \
    --csv-path data/train_labels.csv

# Training experiments (fast iteration)
python train_effnet_mask.py \
    --train-csv data/train_data_preprocessed/train_labels.csv \
    --img-dir data/train_data_preprocessed \
    --optimizer lion \
    --batch-size 16
```

Now you can iterate on training hyperparameters quickly without reprocessing!
