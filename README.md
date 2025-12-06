# AN2DL Image Classification

This repository contains training and inference code for an image classification task where RGB images are augmented with an additional mask channel (4-channel input).

The main entrypoints are:
- `train_effnet_mask.py`: EfficientNet-B0 with mask as 4th channel.
- `train_convnext_mask.py`: ConvNeXt-Base with mask as 4th channel.
- `infer_effnet_mask.py`: Inference script for EfficientNet-B0 4-channel models.

## 1. Environment Setup

Recommended minimal environment (PyTorch + PyTorch Lightning + torchvision, etc.):

```bash
# From repository root
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
# or: source .venv/bin/activate.fish / .venv\Scripts\activate for Windows

pip install --upgrade pip
pip install -r requirements.txt  # if you have one
```

If `requirements.txt` is not available, install at least:

```bash
pip install torch torchvision pytorch-lightning pandas scikit-learn pillow
```

## 2. Data Layout

By default, scripts assume the following structure under the repo root:

```text
AN2DL_ImageClassification/
├── data/
│   ├── train_labels.csv
│   ├── train_data/
│   │   ├── img_XXXX.png / .jpg ...   # training RGB images
│   │   ├── mask_XXXX.png             # corresponding binary/gray masks
│   └── test_data/
│       ├── img_YYYY.png / .jpg ...   # test RGB images
│       ├── mask_YYYY.png             # corresponding masks
```

- `train_labels.csv` must contain at least columns:
  - `sample_index`: filename of the image (e.g. `img_0001.png`).
  - `label`: class label string.
- Mask filenames are inferred from the image name: for `img_0001.png` the mask is `mask_0001.png` in the same (or specified) directory.

You can override paths via CLI arguments (see below).

## 3. Training: EfficientNet-B0 (`train_effnet_mask.py`)

### Basic single-split training

```bash
python train_effnet_mask.py \
  --train-csv data/train_labels.csv \
  --img-dir data/train_data \
  --mask-dir data/train_data \
  --batch-size 32 \
  --epochs 20 \
  --lr 1e-3 \
  --img-size 224
```

Key arguments:
- `--train-csv`: path to training labels CSV.
- `--img-dir`: directory with RGB training images.
- `--mask-dir`: directory with mask images (same naming convention).
- `--batch-size`: batch size (default: 32).
- `--epochs`: number of epochs (default: 10).
- `--lr`: learning rate (default: 1e-3).
- `--warmup-epochs`: number of warmup epochs for the LR scheduler.
- `--num-workers`: dataloader workers (default: 4).
- `--img-size`: resize size (square) (default: 224).
- `--hard-example-k`: top-K hardest training samples to log per epoch (saved under `hard_examples/`).
- `--patience`: early stopping patience in epochs.

The script logs to TensorBoard logs under `tb_logs/effnet_single_split` and saves checkpoints (best/last) using `ModelCheckpoint` (monitored on `val_acc`).

### Cross-validation training

To run Stratified K-Fold cross-validation instead of a single train/validation split:

```bash
python train_effnet_mask.py \
  --use-cv \
  --n-splits 5 \
  --train-csv data/train_labels.csv \
  --img-dir data/train_data \
  --mask-dir data/train_data
```

Additional CV-related arguments:
- `--use-cv`: if set, run K-fold cross-validation.
- `--n-splits`: number of folds (default: 5).
- `--val-size`: only used in single-split mode (default: 0.2).
- `--seed`: random seed for splitting.

Checkpoints are saved under `tb_logs/effnet_foldX` for each fold.

## 4. Training: ConvNeXt-Base (`train_convnext_mask.py`)

Usage is analogous to the EfficientNet script, but with slightly different defaults.

### Basic single-split training

```bash
python train_convnext_mask.py \
  --train-csv data/train_labels.csv \
  --img-dir data/train_data \
  --mask-dir data/train_data \
  --batch-size 16 \
  --epochs 20 \
  --lr 1e-4 \
  --img-size 224
```

Important arguments (see the script for full list):
- `--batch-size` (default: 16).
- `--epochs` (default: 10).
- `--lr` (default: 1e-4).
- `--warmup-epochs`, `--num-workers`, `--img-size`, `--hard-example-k`, `--patience` similar to EfficientNet script.
- `--val-size`, `--seed` as above.

Hard examples are saved under `hard_examples_convnext/`.

### Cross-validation training

```bash
python train_convnext_mask.py \
  --use-cv \
  --n-splits 5 \
  --train-csv data/train_labels.csv \
  --img-dir data/train_data \
  --mask-dir data/train_data
```

Logs go to `tb_logs/convnext_foldX` or `tb_logs/convnext_single_split` depending on mode, and checkpoints are saved via `ModelCheckpoint` monitoring `val_acc`.

## 5. Inference: EfficientNet-B0 (`infer_effnet_mask.py`)

After training an EfficientNet model, use `infer_effnet_mask.py` to generate predictions on test images.

### Example

```bash
python infer_effnet_mask.py \
  --checkpoint path/to/checkpoint.ckpt \
  --train-csv data/train_labels.csv \
  --img-dir data/test_data \
  --mask-dir data/test_data \
  --output-csv submission_effnet_baseline.csv \
  --batch-size 32 \
  --img-size 224
```

Arguments:
- `--checkpoint` (required): path to a trained model checkpoint (`.ckpt` from PyTorch Lightning or `.pth` state dict).
- `--train-csv`: used only to reconstruct the label mapping (string label ↔ integer index). Must be the same label set as during training.
- `--img-dir`: directory containing test RGB images (`img_XXXX.*`).
- `--mask-dir`: directory containing corresponding masks (`mask_XXXX.png`).
- `--output-csv`: where to save predictions CSV (default: `submission.csv`).
- `--batch-size`, `--img-size`: inference batch size and resize size.

The output CSV has the format:

```text
sample_index,label
img_0001.png,<predicted_label>
img_0002.png,<predicted_label>
...
```

## 6. Hardware and Device Selection

Both training scripts automatically select the device:
- GPU if `torch.cuda.is_available()`.
- Apple Silicon MPS if available (`torch.mps.is_available()`).
- Otherwise CPU.

For best performance, run on a CUDA-capable GPU or Apple Silicon with MPS support.

## 7. TensorBoard Monitoring

To monitor training metrics and F1 score:

```bash
tensorboard --logdir tb_logs
```

Then open the printed URL in your browser to inspect loss, accuracy, and macro-F1 curves for different runs/folds.
