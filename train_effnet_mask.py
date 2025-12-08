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


# --- Dataset ---
class PreprocessedImageDataset(Dataset):
    """Dataset for preprocessed images (already masked and cropped)."""
    def __init__(
        self,
        df,
        img_dir,
        transform=None,
    ):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

        # String label names to numbers
        self.class_numbers = {k: i for i, k in enumerate(df["label"].unique())}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["sample_index"])
        
        # Load preprocessed image (already masked and cropped)
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        y = self.class_numbers[row["label"]]
        y = torch.tensor(y, dtype=torch.long)
        return img, y, row["sample_index"]


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


# --- Lightning Module ---
class LitEffNet(pl.LightningModule):
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
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 1.0,
        mixup_prob: float = 0.5,
        cutmix_prob: float = 0.5,
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
        self.training_hard_examples = []  # list of dicts with loss and sample_index
        self.val_preds = []
        self.val_targets = []
        self.val_images = []  # Store images for TensorBoard visualization
        self.class_names = ["Luminal A", "Luminal B", "HER2(+)", "Triple Negative"]
        
        # Mixup / CutMix augmentation
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob

    def forward(self, x):
        return self.model(x)

    def _sample_lambda(self, alpha: float) -> float:
        """Sample mixing coefficient from Beta distribution."""
        if alpha <= 0:
            return 1.0
        return float(np.random.beta(alpha, alpha))

    def _mixup(self, x, y):
        """Apply Mixup augmentation: linear interpolation of images and labels."""
        lam = self._sample_lambda(self.mixup_alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        x_mixed = lam * x + (1.0 - lam) * x[index]
        y_a, y_b = y, y[index]
        return x_mixed, y_a, y_b, lam

    def _cutmix(self, x, y):
        """Apply CutMix augmentation: paste random box from one image to another."""
        lam = self._sample_lambda(self.cutmix_alpha)
        batch_size, _, h, w = x.size()
        index = torch.randperm(batch_size, device=x.device)

        # Generate random bounding box
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        cut_w = int(w * np.sqrt(1.0 - lam))
        cut_h = int(h * np.sqrt(1.0 - lam))

        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))

        x_mixed = x.clone()
        x_mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual box size
        lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
        y_a, y_b = y, y[index]
        return x_mixed, y_a, y_b, lam_adjusted

    def on_train_start(self):
        # Freeze backbone for initial epochs if specified
        if self.freeze_backbone_epochs > 0:
            print(f"Freezing backbone for first {self.freeze_backbone_epochs} epochs")
            for param in self.model.base.features.parameters():
                param.requires_grad = False

    def on_train_epoch_start(self):
        # Unfreeze backbone after warmup period
        if self.current_epoch == self.freeze_backbone_epochs and self.freeze_backbone_epochs > 0:
            print(f"\nUnfreezing backbone at epoch {self.current_epoch}")
            for param in self.model.base.features.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        x, y, sample_indices = batch

        # Choose augmentation (only during training)
        use_mixup = (self.mixup_alpha > 0) and (np.random.rand() < self.mixup_prob)
        use_cutmix = (self.cutmix_alpha > 0) and (not use_mixup) and (np.random.rand() < self.cutmix_prob)

        if use_mixup:
            x, y_a, y_b, lam = self._mixup(x, y)
            logits = self(x)
            loss_a = torch.nn.functional.cross_entropy(logits, y_a, reduction="none")
            loss_b = torch.nn.functional.cross_entropy(logits, y_b, reduction="none")
            per_sample_loss = lam * loss_a + (1.0 - lam) * loss_b

        elif use_cutmix:
            x, y_a, y_b, lam = self._cutmix(x, y)
            logits = self(x)
            loss_a = torch.nn.functional.cross_entropy(logits, y_a, reduction="none")
            loss_b = torch.nn.functional.cross_entropy(logits, y_b, reduction="none")
            per_sample_loss = lam * loss_a + (1.0 - lam) * loss_b

        else:
            logits = self(x)
            per_sample_loss = torch.nn.functional.cross_entropy(logits, y, reduction="none")

        loss = per_sample_loss.mean()

        # Training accuracy is approximate under mixup/cutmix, but useful for monitoring
        # Validation metrics are what truly matter
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        # Track hardest examples (approximate under mixup/cutmix, but OK for heuristic)
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
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        # store for F1 computation at epoch end
        self.val_preds.append(logits.argmax(dim=1).detach().cpu())
        self.val_targets.append(y.detach().cpu())
        
        # Store first batch of images for TensorBoard visualization (only first batch to save memory)
        if batch_idx == 0 and len(self.val_images) == 0:
            self.val_images = [(x.detach().cpu(), y.detach().cpu(), logits.argmax(dim=1).detach().cpu())]

    def configure_optimizers(self):
        # Choose optimizer based on configuration
        if self.optimizer_name == "ranger":
            # Ranger optimizer (RAdam + Lookahead)
            optimizer = Ranger(
                self.parameters(),
                lr=self.lr,
                alpha=0.5,  # Lookahead interpolation coefficient
                k=6,  # Lookahead step interval
                betas=(0.95, 0.999),  # RAdam betas
                weight_decay=self.weight_decay
            )
            print(f"Using Ranger optimizer (RAdam + Lookahead)")
        elif self.optimizer_name == "lion":
            # Lion optimizer (EvoLved Sign Momentum)
            # Note: Lion typically needs 3-10x smaller learning rate than Adam
            lion_lr = self.lr / 3.0  # Auto-adjust LR for Lion
            optimizer = Lion(
                self.parameters(),
                lr=lion_lr,
                betas=(0.9, 0.99),
                weight_decay=self.weight_decay
            )
            print(f"Using Lion optimizer (adjusted LR: {lion_lr:.2e} from {self.lr:.2e})")
        else:
            # AdamW with weight decay for better regularization
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999)
            )
            print(f"Using AdamW optimizer")

        # Cosine annealing with warmup
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
            # Just cosine annealing without warmup
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
        
        # Log images with predictions to TensorBoard
        if self.val_images and self.logger:
            self._log_images_to_tensorboard()
        
        self.val_preds = []
        self.val_targets = []
        self.val_images = []
    
    def _log_images_to_tensorboard(self):
        """Log validation images with predictions to TensorBoard."""
        import torchvision
        from PIL import Image, ImageDraw, ImageFont
        
        x_batch, y_batch, pred_batch = self.val_images[0]
        
        # Take up to 16 images from the batch
        num_images = min(16, x_batch.shape[0])
        images_to_log = []
        
        for i in range(num_images):
            # Get image (3 channels: RGB)
            img = x_batch[i]
            
            # Denormalize (ImageNet normalization)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # Convert to PIL for adding text
            img_pil = torchvision.transforms.ToPILImage()(img)
            draw = ImageDraw.Draw(img_pil)
            
            # Get labels
            true_label = self.class_names[y_batch[i].item()] if y_batch[i].item() < len(self.class_names) else str(y_batch[i].item())
            pred_label = self.class_names[pred_batch[i].item()] if pred_batch[i].item() < len(self.class_names) else str(pred_batch[i].item())
            
            # Determine text color (green if correct, red if wrong)
            color = "green" if y_batch[i] == pred_batch[i] else "red"
            
            # Add text with prediction and ground truth
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()
            
            text = f"True: {true_label}\\nPred: {pred_label}"
            draw.text((10, 10), text, fill=color, font=font)
            
            # Convert back to tensor
            img_tensor = torchvision.transforms.ToTensor()(img_pil)
            images_to_log.append(img_tensor)
        
        # Create grid and log to TensorBoard
        grid = torchvision.utils.make_grid(images_to_log, nrow=4, padding=10, pad_value=1)
        self.logger.experiment.add_image(
            'val/predictions',
            grid,
            global_step=self.current_epoch
        )


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
    img_size: int = 384,
    patience: int = 3,
    hard_example_k: int = 16,
    use_weighted_sampler: bool = False,
    freeze_backbone_epochs: int = 0,
    optimizer_name: str = "adamw",
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
    cutmix_prob: float = 0.5,
    use_randaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 7,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"])):
        print(f"Fold {fold+1}/{n_splits}")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Training transforms with geometric augmentations and optional RandAugment
        train_transform_list = [
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=15),
        ]
        
        # Add RandAugment if enabled
        if use_randaugment:
            train_transform_list.append(T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
        
        train_transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_transform = T.Compose(train_transform_list)
        
        # Validation transforms (no augmentation)
        val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_ds = PreprocessedImageDataset(
            train_df,
            img_dir,
            transform=train_transform,
        )
        val_ds = PreprocessedImageDataset(
            val_df,
            img_dir,
            transform=val_transform,
        )
        
        # Setup sampler for class imbalance
        if use_weighted_sampler:
            class_counts = train_df["label"].value_counts()
            weight_per_class = {cls: 1.0/count for cls, count in class_counts.items()}
            sample_weights = [weight_per_class[row["label"]] for _, row in train_df.iterrows()]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
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
        model = LitEffNet(
            num_classes,
            lr=lr,
            weight_decay=1e-4,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            label_smoothing=0.1,
            hard_example_k=hard_example_k,
            freeze_backbone_epochs=freeze_backbone_epochs,
            optimizer_name=optimizer_name,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mixup_prob=mixup_prob,
            cutmix_prob=cutmix_prob,
        )
        logger = TensorBoardLogger("tb_logs", name=f"effnet_fold{fold+1}")
        checkpoint_cb = ModelCheckpoint(
            monitor="val_f1_macro",
            mode="max",
            save_top_k=3,  # keep best 3 models
            save_last=True,  # also keep last epoch
            filename="epoch{epoch:02d}-val_f1_macro{val_acc:.4f}",  # detailed filename
            auto_insert_metric_name=False,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
                checkpoint_cb
            ],
            accelerator=DEVICE,
            devices=1,
            logger=logger,
            log_every_n_steps=1,
            gradient_clip_val=1.0,  # Gradient clipping for stability
            deterministic=True,  # Reproducibility
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
    img_size: int = 384,
    patience: int = 3,
    hard_example_k: int = 16,
    use_weighted_sampler: bool = False,
    freeze_backbone_epochs: int = 0,
    optimizer_name: str = "adamw",
    mixup_alpha: float = 0.4,
    cutmix_alpha: float = 1.0,
    mixup_prob: float = 0.5,
    cutmix_prob: float = 0.5,
    use_randaugment: bool = False,
    randaugment_n: int = 2,
    randaugment_m: int = 7,
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

    # Training transforms with geometric augmentations and optional RandAugment
    train_transform_list = [
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ]
    
    # Add RandAugment if enabled
    if use_randaugment:
        train_transform_list.append(T.RandAugment(num_ops=randaugment_n, magnitude=randaugment_m))
    
    train_transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = T.Compose(train_transform_list)
    
    # Validation transforms (no augmentation)
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = PreprocessedImageDataset(
        train_df,
        img_dir,
        transform=train_transform,
    )
    val_ds = PreprocessedImageDataset(
        val_df,
        img_dir,
        transform=val_transform,
    )

    # Setup sampler for class imbalance
    if use_weighted_sampler:
        class_counts = train_df["label"].value_counts()
        weight_per_class = {cls: 1.0/count for cls, count in class_counts.items()}
        sample_weights = [weight_per_class[row["label"]] for _, row in train_df.iterrows()]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)
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

    model = LitEffNet(
        num_classes,
        lr=lr,
        weight_decay=1e-4,
        warmup_epochs=warmup_epochs,
        max_epochs=epochs,
        label_smoothing=0.1,
        hard_example_k=hard_example_k,
        freeze_backbone_epochs=freeze_backbone_epochs,
        optimizer_name=optimizer_name,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        mixup_prob=mixup_prob,
        cutmix_prob=cutmix_prob,
    )
    logger = TensorBoardLogger("tb_logs", name="effnet_single_split")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_f1_macro",
        mode="max",
        save_top_k=3,  # keep best 3 models
        save_last=True,  # also keep last epoch
        filename="epoch{epoch:02d}-val_f1_macro{val_acc:.4f}",  # detailed filename
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            checkpoint_cb
        ],
        accelerator=DEVICE,
        devices=1,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,  # Gradient clipping for stability
        deterministic=True,  # Reproducibility
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
        default="data/train_data_preprocessed",
        help="Directory with preprocessed training images (already masked and cropped)",
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
        "--img-size", type=int, default=384, help="Image resize size (square) - EfficientNetV2-S uses 384x384"
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

    # Augmentation
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.4,
        help="Mixup alpha parameter (set to 0 to disable mixup)",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=1.0,
        help="CutMix alpha parameter (set to 0 to disable cutmix)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.5,
        help="Probability of applying mixup when enabled",
    )
    parser.add_argument(
        "--cutmix-prob",
        type=float,
        default=0.5,
        help="Probability of applying cutmix when enabled (applied if mixup not chosen)",
    )
    parser.add_argument(
        "--use-randaugment",
        action="store_true",
        help="Use RandAugment policy-based augmentation",
    )
    parser.add_argument(
        "--randaugment-n",
        type=int,
        default=2,
        help="RandAugment number of operations",
    )
    parser.add_argument(
        "--randaugment-m",
        type=int,
        default=7,
        help="RandAugment magnitude of operations",
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
            img_size=args.img_size,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
            use_weighted_sampler=args.use_weighted_sampler,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            optimizer_name=args.optimizer,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            cutmix_prob=args.cutmix_prob,
            use_randaugment=args.use_randaugment,
            randaugment_n=args.randaugment_n,
            randaugment_m=args.randaugment_m,
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
            img_size=args.img_size,
            patience=args.patience,
            hard_example_k=args.hard_example_k,
            use_weighted_sampler=args.use_weighted_sampler,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
            optimizer_name=args.optimizer,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            cutmix_prob=args.cutmix_prob,
            use_randaugment=args.use_randaugment,
            randaugment_n=args.randaugment_n,
            randaugment_m=args.randaugment_m,
        )
