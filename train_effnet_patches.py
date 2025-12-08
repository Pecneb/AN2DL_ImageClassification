import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


# --- Ranger Optimizer (RAdam + Lookahead) ---
class Ranger(optim.Optimizer):
    """Ranger optimizer combining RAdam and Lookahead.
    
    Ranger = RAdam + Lookahead
    - RAdam: Rectified Adam with variance warmup
    - Lookahead: Maintains slow and fast weights for better convergence
    
    Args:
        params: model parameters
        lr: learning rate (default: 1e-3)
        alpha: Lookahead interpolation coefficient (default: 0.5)
        k: Lookahead step interval (default: 6)
        betas: RAdam coefficients (default: (0.95, 0.999))
        eps: term for numerical stability (default: 1e-5)
        weight_decay: weight decay (L2 penalty) (default: 0)
    """
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(0.95, 0.999), eps=1e-5, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha parameter: {alpha}")
        if not 0 < k:
            raise ValueError(f"Invalid k parameter: {k}")
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, k=k)
        super(Ranger, self).__init__(params, defaults)
        
        # Lookahead setup
        for group in self.param_groups:
            group['step_counter'] = 0
            
        # Initialize slow weights
        self.slow_weights = [[p.clone().detach() for p in group['params']] for group in self.param_groups]

    def __setstate__(self, state):
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # RAdam step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Ranger does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Rectification term (RAdam)
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

                if rho_t > 5:
                    # Adaptive learning rate with variance rectification
                    rect = ((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                    step_size = group['lr'] * rect / bias_correction1
                    
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # Use simple SGD-style update when variance is not reliable
                    step_size = group['lr'] / bias_correction1
                    p.data.add_(exp_avg, alpha=-step_size)

        # Lookahead step
        for group, slow_group in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % group['k'] == 0:
                for p, slow_p in zip(group['params'], slow_group):
                    if p.grad is None:
                        continue
                    # Interpolate between slow and fast weights
                    slow_p.data.add_(p.data - slow_p.data, alpha=group['alpha'])
                    p.data.copy_(slow_p.data)

        return loss


# --- Lion Optimizer (EvoLved Sign Momentum) ---
class Lion(optim.Optimizer):
    """Lion optimizer - EvoLved Sign Momentum.
    
    Paper: https://arxiv.org/abs/2302.06675
    Lion uses sign-based updates with momentum, requiring less memory and
    often achieving better results than Adam/AdamW with 3-10x smaller learning rates.
    
    Args:
        params: model parameters
        lr: learning rate (default: 1e-4, use smaller LR than Adam/AdamW)
        betas: coefficients for momentum (default: (0.9, 0.99))
        weight_decay: weight decay coefficient (default: 0)
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super(Lion, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight update using sign of interpolated gradient
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update momentum (exponential moving average)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


if torch.cuda.is_available():
    DEVICE = "gpu"
elif hasattr(torch, "mps") and torch.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# --- Patch-Based Dataset ---
class PatchBasedDataset(Dataset):
    """Dataset that extracts random patches at full resolution from preprocessed images.
    
    During training: extracts random patches per epoch (data augmentation)
    During validation: uses deterministic center crop or grid-based patches
    """
    def __init__(
        self,
        df,
        img_dir,
        patch_size: int = 384,
        patches_per_image: int = 2,
        transform=None,
        is_training: bool = True,
    ):
        self.df = df
        self.img_dir = img_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.is_training = is_training

        # String label names to numbers
        self.class_numbers = {k: i for i, k in enumerate(df["label"].unique())}

    def __len__(self):
        return len(self.df) * self.patches_per_image

    def __getitem__(self, idx):
        # Map flat index to (image_idx, patch_idx)
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image
        
        row = self.df.iloc[img_idx]
        img_path = os.path.join(self.img_dir, row["sample_index"])
        label = row["label"]
        
        # Load full-resolution preprocessed image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # Extract patch at full resolution
        if self.is_training:
            # Random crop during training
            if w > self.patch_size and h > self.patch_size:
                left = np.random.randint(0, w - self.patch_size)
                top = np.random.randint(0, h - self.patch_size)
                right = left + self.patch_size
                bottom = top + self.patch_size
                patch = img.crop((left, top, right, bottom))
            else:
                # Image smaller than patch size, use full image and resize
                patch = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        else:
            # Deterministic grid-based patches during validation
            if w < self.patch_size or h < self.patch_size:
                # Image smaller than patch size, resize entire image
                patch = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
            else:
                # Divide image into grid and extract specific patch
                n_cols = max(1, w // self.patch_size)
                n_rows = max(1, h // self.patch_size)
                
                col = patch_idx % n_cols
                row_idx = patch_idx // n_cols
                
                left = col * self.patch_size
                top = row_idx * self.patch_size
                right = min(left + self.patch_size, w)
                bottom = min(top + self.patch_size, h)
                
                # Ensure valid coordinates
                if right <= left or bottom <= top:
                    # Fallback: use entire image
                    patch = img.resize((self.patch_size, self.patch_size), Image.BILINEAR)
                else:
                    patch = img.crop((left, top, right, bottom))
                    
                    # Resize to patch_size if needed
                    if patch.size != (self.patch_size, self.patch_size):
                        patch = patch.resize((self.patch_size, self.patch_size), Image.BILINEAR)
        
        if self.transform:
            patch = self.transform(patch)
        
        y = self.class_numbers[label]
        y = torch.tensor(y, dtype=torch.long)
        
        # Return patch, label, and original image index for aggregation
        return patch, y, img_idx

    def get_image_labels(self):
        """Return labels for each unique image (not patch)."""
        labels = []
        for _, row in self.df.iterrows():
            labels.append(self.class_numbers[row["label"]])
        return torch.tensor(labels, dtype=torch.long)


# --- Model ---
class EfficientNetV2S(torch.nn.Module):
    """Standard EfficientNetV2-S for 3-channel RGB input."""
    def __init__(self, num_classes):
        super().__init__()
        # Load torchvision EfficientNetV2-S with ImageNet weights
        self.base = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Replace classifier head with higher dropout for regularization
        in_features = self.base.classifier[1].in_features
        self.base.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)


# --- Lightning Module with Patch Aggregation ---
class LitEffNetPatches(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
        max_epochs: int = 50,
        label_smoothing: float = 0.1,
        hard_example_k: int = 16,
        freeze_backbone_epochs: int = 0,
        optimizer_name: str = "adamw",
        patches_per_image: int = 2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = EfficientNetV2S(num_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.optimizer_name = optimizer_name.lower()
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.hard_example_k = hard_example_k
        self.patches_per_image = patches_per_image
        self.training_hard_examples = []
        self.val_patch_logits = []  # Store patch logits for aggregation
        self.val_patch_labels = []  # Store patch labels
        self.val_image_indices = []  # Store image indices for aggregation
        self.class_names = ["Luminal A", "Luminal B", "HER2(+)", "Triple Negative"]

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self.freeze_backbone_epochs > 0:
            print(f"Freezing backbone for first {self.freeze_backbone_epochs} epochs")
            for param in self.model.base.features.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            print(f"\nUnfreezing backbone at epoch {self.current_epoch}")
            for param in self.model.base.features.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        x, y, img_indices = batch
        logits = self(x)
        per_sample_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")
        loss = per_sample_loss.mean()
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Track hardest examples (by patch)
        for img_idx, l in zip(img_indices, per_sample_loss.detach().cpu().tolist()):
            self.training_hard_examples.append(
                {
                    "image_index": int(img_idx),
                    "loss": float(l),
                    "epoch": int(self.current_epoch),
                }
            )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, img_indices = batch
        logits = self(x)
        
        # Store patch-level predictions for later aggregation
        self.val_patch_logits.append(logits.detach().cpu())
        self.val_patch_labels.append(y.detach().cpu())
        self.val_image_indices.append(img_indices.detach().cpu())
        
        # Log patch-level metrics (approximate)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_patch_loss", loss, prog_bar=True)
        self.log("val_patch_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        if self.optimizer_name == "ranger":
            optimizer = Ranger(
                self.parameters(),
                lr=self.lr,
                alpha=0.5,
                k=6,
                betas=(0.95, 0.999),
                weight_decay=self.weight_decay
            )
            print(f"Using Ranger optimizer (RAdam + Lookahead)")
        elif self.optimizer_name == "lion":
            lion_lr = self.lr / 3.0
            optimizer = Lion(
                self.parameters(),
                lr=lion_lr,
                betas=(0.9, 0.99),
                weight_decay=self.weight_decay
            )
            print(f"Using Lion optimizer (adjusted LR: {lion_lr:.2e} from {self.lr:.2e})")
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            print(f"Using AdamW optimizer")

        if self.warmup_epochs > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=self.warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs - self.warmup_epochs,
                eta_min=1e-6
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_epochs]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.max_epochs,
                eta_min=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

    def on_train_epoch_end(self):
        if not self.training_hard_examples:
            return
        k = min(self.hard_example_k, len(self.training_hard_examples))
        sorted_examples = sorted(
            self.training_hard_examples, key=lambda d: d["loss"], reverse=True
        )[:k]
        df = pd.DataFrame(sorted_examples)
        os.makedirs("hard_examples_patches", exist_ok=True)
        df.to_csv(
            os.path.join(
                "hard_examples_patches", f"hard_examples_epoch_{self.current_epoch}.csv"
            ),
            index=False,
        )
        self.training_hard_examples = []

    def on_validation_epoch_end(self):
        if not self.val_patch_logits:
            return
        
        # Aggregate patch predictions to image-level predictions
        all_logits = torch.cat(self.val_patch_logits)  # [total_patches, num_classes]
        all_labels = torch.cat(self.val_patch_labels)  # [total_patches]
        all_img_indices = torch.cat(self.val_image_indices)  # [total_patches]
        
        # Group patches by image and aggregate
        unique_img_indices = torch.unique(all_img_indices)
        image_preds = []
        image_targets = []
        
        for img_idx in unique_img_indices:
            mask = all_img_indices == img_idx
            patch_logits = all_logits[mask]  # [num_patches_for_this_image, num_classes]
            patch_labels = all_labels[mask]
            
            # Aggregate patch logits by averaging (soft voting)
            avg_logits = patch_logits.mean(dim=0)  # [num_classes]
            image_pred = avg_logits.argmax().item()
            image_preds.append(image_pred)
            
            # All patches from same image should have same label
            image_targets.append(patch_labels[0].item())
        
        # Compute image-level metrics
        image_preds = np.array(image_preds)
        image_targets = np.array(image_targets)
        
        f1 = f1_score(image_targets, image_preds, average="macro")
        acc = (image_preds == image_targets).mean()
        
        self.log('val_f1_macro', f1, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        print(f"\nEpoch {self.current_epoch} - Image-level metrics:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1 (macro): {f1:.4f}")
        print(f"  Images evaluated: {len(unique_img_indices)}")
        
        # Clear buffers
        self.val_patch_logits = []
        self.val_patch_labels = []
        self.val_image_indices = []


# --- Training with CV ---
def run_cv(
    df,
    img_dir,
    num_classes,
    batch_size: int = 32,
    epochs: int = 10,
    warmup_epochs: int = 0,
    n_splits: int = 5,
    num_workers: int = 4,
    lr: float = 1e-3,
    patch_size: int = 384,
    patches_per_image: int = 2,
    patience: int = 3,
    hard_example_k: int = 16,
    use_weighted_sampler: bool = False,
    freeze_backbone_epochs: int = 0,
    optimizer_name: str = "adamw",
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*60}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Training transforms (geometric augmentations at patch level)
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms (no augmentation)
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_ds = PatchBasedDataset(
            train_df,
            img_dir,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            transform=train_transform,
            is_training=True,
        )
        val_ds = PatchBasedDataset(
            val_df,
            img_dir,
            patch_size=patch_size,
            patches_per_image=patches_per_image,
            transform=val_transform,
            is_training=False,
        )
        
        print(f"Training: {len(train_df)} images → {len(train_ds)} patches")
        print(f"Validation: {len(val_df)} images → {len(val_ds)} patches")
        
        # Setup sampler for class imbalance (if enabled)
        if use_weighted_sampler:
            # Note: This weights patches, not images. For true image-level weighting,
            # we'd need a custom batch sampler. This is a reasonable approximation.
            class_counts = train_df["label"].value_counts()
            weight_per_class = {cls: 1.0/count for cls, count in class_counts.items()}
            sample_weights = []
            for idx in range(len(train_ds)):
                img_idx = idx // patches_per_image
                label = train_df.iloc[img_idx]["label"]
                sample_weights.append(weight_per_class[label])
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
        
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        model = LitEffNetPatches(
            num_classes,
            lr=lr,
            weight_decay=1e-4,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            label_smoothing=0.1,
            hard_example_k=hard_example_k,
            freeze_backbone_epochs=freeze_backbone_epochs,
            optimizer_name=optimizer_name,
            patches_per_image=patches_per_image,
        )
        logger = TensorBoardLogger("tb_logs", name=f"effnet_patches_fold{fold+1}")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3,
            save_last=True,
            filename="epoch{epoch:02d}-val_f1{val_f1_macro:.4f}",
            auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="val_f1_macro", patience=patience, mode="max"),
                checkpoint_cb
            ],
            accelerator=DEVICE,
            devices=1,
            logger=logger,
            log_every_n_steps=1,
            gradient_clip_val=1.0,
            deterministic=True,
        )
        trainer.fit(model, train_loader, val_loader)


def run_single_split(
    df,
    img_dir,
    num_classes,
    batch_size: int = 32,
    epochs: int = 10,
    warmup_epochs: int = 0,
    val_size: float = 0.2,
    random_state: int = 42,
    num_workers: int = 4,
    lr: float = 1e-3,
    patch_size: int = 384,
    patches_per_image: int = 2,
    patience: int = 3,
    hard_example_k: int = 16,
    use_weighted_sampler: bool = False,
    freeze_backbone_epochs: int = 0,
    optimizer_name: str = "adamw",
):
    """Train on a single train/validation split with patch-based approach."""
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Training transforms (geometric augmentations at patch level)
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PatchBasedDataset(
        train_df,
        img_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        transform=train_transform,
        is_training=True,
    )
    val_ds = PatchBasedDataset(
        val_df,
        img_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        transform=val_transform,
        is_training=False,
    )

    print(f"\n{'='*60}")
    print(f"Dataset Summary:")
    print(f"{'='*60}")
    print(f"Training: {len(train_df)} images → {len(train_ds)} patches")
    print(f"Validation: {len(val_df)} images → {len(val_ds)} patches")
    print(f"Patch size: {patch_size}×{patch_size}")
    print(f"Patches per image: {patches_per_image}")
    print(f"{'='*60}\n")

    # Setup sampler for class imbalance (if enabled)
    if use_weighted_sampler:
        class_counts = train_df["label"].value_counts()
        weight_per_class = {cls: 1.0/count for cls, count in class_counts.items()}
        sample_weights = []
        for idx in range(len(train_ds)):
            img_idx = idx // patches_per_image
            label = train_df.iloc[img_idx]["label"]
            sample_weights.append(weight_per_class[label])
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
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

    model = LitEffNetPatches(
        num_classes,
        lr=lr,
        weight_decay=1e-4,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        label_smoothing=0.1,
        hard_example_k=hard_example_k,
        freeze_backbone_epochs=freeze_backbone_epochs,
        optimizer_name=optimizer_name,
        patches_per_image=patches_per_image,
    )
    logger = TensorBoardLogger("tb_logs", name="effnet_patches_single_split")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        save_top_k=3,
        save_last=True,
        filename="epoch{epoch:02d}-val_f1{val_f1_macro:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="val_f1_macro", patience=patience, mode="max"),
            checkpoint_cb
        ],
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        deterministic=True,
    )

    trainer.fit(model, train_loader, val_loader)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EfficientNetV2-S training with patch-based approach at full resolution"
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
        default="data/train_data_preprocessed",
        help="Directory with preprocessed training images (full resolution)",
    )

    # Training
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (for patches)")
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
        "--patch-size", type=int, default=384, help="Patch size (square) - extracted at full resolution"
    )
    parser.add_argument(
        "--patches-per-image", type=int, default=2, 
        help="Number of patches to extract per image per epoch (training augmentation)"
    )
    parser.add_argument(
        "--hard-example-k",
        type=int,
        default=16,
        help="Top-K hardest training samples to save per epoch",
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience in epochs"
    )
    parser.add_argument(
        "--use-weighted-sampler",
        action="store_true",
        help="Use WeightedRandomSampler to handle class imbalance",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Number of initial epochs to freeze backbone (two-stage training)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "ranger", "lion"],
        help="Optimizer to use: adamw (AdamW), ranger (RAdam + Lookahead), or lion (EvoLved Sign Momentum)",
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
    num_classes = df["label"].nunique()

    print("\n" + "="*60)
    print("PATCH-BASED TRAINING AT FULL RESOLUTION")
    print("="*60)
    print(f"Images will be loaded at full resolution")
    print(f"Patches ({args.patch_size}×{args.patch_size}) extracted per image: {args.patches_per_image}")
    print(f"Total training samples per epoch: {len(df) * args.patches_per_image} patches")
    print(f"This preserves fine-grained details for histopathology classification")
    print("="*60 + "\n")

    if args.use_cv:
        run_cv(
            df,
            img_dir,
            num_classes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            n_splits=args.n_splits,
            num_workers=args.num_workers,
            lr=args.lr,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
            use_weighted_sampler=args.use_weighted_sampler,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            optimizer_name=args.optimizer,
        )
    else:
        run_single_split(
            df,
            img_dir,
            num_classes,
            batch_size=args.batch_size,
            epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            val_size=args.val_size,
            random_state=args.seed,
            num_workers=args.num_workers,
            lr=args.lr,
            patch_size=args.patch_size,
            patches_per_image=args.patches_per_image,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
            use_weighted_sampler=args.use_weighted_sampler,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            optimizer_name=args.optimizer,
        )
