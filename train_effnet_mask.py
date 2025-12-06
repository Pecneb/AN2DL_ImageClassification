import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

if torch.cuda.is_available():
    DEVICE = "gpu"
elif hasattr(torch, "mps") and torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# --- Dataset ---
class MaskedImageDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        # String label names to numbers
        self.class_numbers = {k: i for i, k in enumerate(df["label"].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["sample_index"])
        mask_path = os.path.join(
            self.mask_dir,
            f"mask_{os.path.splitext(row['sample_index'])[0].split('_')[1]}.png",
        )
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        img = np.array(img)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)
        x = np.concatenate([img, mask], axis=-1)  # shape: H x W x 4
        if self.transform:
            x = self.transform(x)
        y = self.class_numbers[row["label"]]
        y = torch.tensor(y, dtype=torch.long)
        return x, y, row["sample_index"]


# --- Model ---
class EfficientNetB0_4ch(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load torchvision EfficientNet-B0 with ImageNet weights
        self.base = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Adapt first conv layer to 4 input channels (RGB + mask)
        old_conv = self.base.features[0][0]
        new_conv = torch.nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            # Initialize mask channel as mean of RGB weights
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.base.features[0][0] = new_conv

        # Replace classifier head for correct num_classes
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)


# --- Lightning Module ---
class LitEffNet(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float = 1e-3,
        warmup_epochs: int = 0,
        hard_example_k: int = 16,
    ):
        super().__init__()
        self.model = EfficientNetB0_4ch(num_classes)
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.criterion = torch.nn.CrossEntropyLoss()
        self.hard_example_k = hard_example_k
        self.training_hard_examples = []  # list of dicts with loss and sample_index
        self.val_preds = []
        self.val_targets = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, sample_indices = batch
        logits = self(x)
        per_sample_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = per_sample_loss.mean()
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        # Track hardest examples in this epoch
        for si, l in zip(sample_indices, per_sample_loss.detach().cpu().tolist()):
            self.training_hard_examples.append(
                {
                    "sample_index": si,
                    "loss": float(l),
                    "epoch": int(self.current_epoch),
                }
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        # store for F1 computation at epoch end
        self.val_preds.append(logits.argmax(dim=1).detach().cpu())
        self.val_targets.append(y.detach().cpu())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.warmup_epochs > 0:

            def lr_lambda(current_epoch):
                if current_epoch < self.warmup_epochs:
                    return float(current_epoch + 1) / float(max(1, self.warmup_epochs))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer

    def on_train_epoch_end(self):
        # Keep only top-k hardest examples for this epoch
        if not self.training_hard_examples:
            return
        k = min(self.hard_example_k, len(self.training_hard_examples))
        sorted_examples = sorted(
            self.training_hard_examples, key=lambda d: d["loss"], reverse=True
        )[:k]
        df = pd.DataFrame(sorted_examples)
        os.makedirs("hard_examples", exist_ok=True)
        df.to_csv(
            os.path.join(
                "hard_examples", f"hard_examples_epoch_{self.current_epoch}.csv"
            ),
            index=False,
        )
        # reset buffer for next epoch
        self.training_hard_examples = []

    def on_validation_epoch_end(self):
        # compute macro F1 over the entire validation epoch
        if not self.val_preds:
            return
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
        f1 = f1_score(targets, preds, average="macro")
        self.log('val_f1_macro', f1, prog_bar=True)
        self.val_preds = []
        self.val_targets = []


# --- Training with CV ---
def run_cv(
    df,
    img_dir,
    mask_dir,
    num_classes,
    batch_size: int = 32,
    epochs: int = 10,
    warmup_epochs: int = 0,
    n_splits: int = 5,
    num_workers: int = 4,
    lr: float = 1e-3,
    img_size: int = 224,
    patience: int = 3,
    hard_example_k: int = 16,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"Fold {fold+1}/{n_splits}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((img_size, img_size)),
            ]
        )
        train_ds = MaskedImageDataset(train_df, img_dir, mask_dir, transform)
        val_ds = MaskedImageDataset(val_df, img_dir, mask_dir, transform)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        model = LitEffNet(
            num_classes,
            lr=lr,
            warmup_epochs=warmup_epochs,
            hard_example_k=hard_example_k,
        )
        logger = TensorBoardLogger("tb_logs", name=f"effnet_fold{fold+1}")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=3,  # keep best 3 models
            save_last=True,  # also keep last epoch
            filename="epoch{epoch:02d}-valloss{val_acc:.4f}",  # detailed filename
            auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=patience),
                # ModelCheckpoint(monitor="val_loss"),
                checkpoint_cb
            ],
            accelerator=DEVICE,
            devices=1,
            logger=logger,
            log_every_n_steps=1,
        )
        trainer.fit(model, train_loader, val_loader)


def run_single_split(
    df,
    img_dir,
    mask_dir,
    num_classes,
    batch_size: int = 32,
    epochs: int = 10,
    warmup_epochs: int = 0,
    val_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 4,
    lr: float = 1e-3,
    img_size: int = 224,
    patience: int = 3,
    hard_example_k: int = 16,
):
    """Train on a single train/validation split instead of cross-validation."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize((img_size, img_size)),
        ]
    )

    train_ds = MaskedImageDataset(train_df, img_dir, mask_dir, transform)
    val_ds = MaskedImageDataset(val_df, img_dir, mask_dir, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    model = LitEffNet(
        num_classes, lr=lr, warmup_epochs=warmup_epochs, hard_example_k=hard_example_k
    )
    logger = TensorBoardLogger("tb_logs", name="effnet_single_split")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=3,  # keep best 3 models
        save_last=True,  # also keep last epoch
        filename="epoch{epoch:02d}-valacc{val_acc:.4f}",  # detailed filename
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience),
            #ModelCheckpoint(monitor="val_loss"),
            checkpoint_cb
        ],
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNet-B0 training with masks as 4th channel"
    )

    # Data
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/train_labels.csv",
        help="Path to train labels CSV",
    )
    parser.add_argument(
        "--img-dir",
        type=str,
        default="data/train_data",
        help="Directory with training images",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="data/train_data",
        help="Directory with mask images",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="Number of warmup epochs for LR"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Image resize size (square)"
    )
    parser.add_argument(
        "--hard-example-k",
        type=int,
        default=16,
        help="Top-K hardest training samples to save per epoch",
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early stopping patience in epochs"
    )

    # Validation / CV
    parser.add_argument(
        "--use-cv",
        action="store_true",
        help="Use StratifiedKFold cross-validation instead of single split",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument(
        "--val-size", type=float, default=0.2, help="Validation size for single split"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    df = pd.read_csv(args.train_csv)
    img_dir = args.img_dir
    mask_dir = args.mask_dir
    num_classes = df["label"].nunique()

    if args.use_cv:
        run_cv(
            df,
            img_dir,
            mask_dir,
            num_classes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            n_splits=args.n_splits,
            num_workers=args.num_workers,
            lr=args.lr,
            img_size=args.img_size,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
        )
    else:
        run_single_split(
            df,
            img_dir,
            mask_dir,
            num_classes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            val_size=args.val_size,
            random_state=args.seed,
            num_workers=args.num_workers,
            lr=args.lr,
            img_size=args.img_size,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
        )
