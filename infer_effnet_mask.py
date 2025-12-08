import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class TestMaskedImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.filenames = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and "img" in f.lower()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_id = os.path.splitext(fname)[0].split("_")[1]
        mask_path = os.path.join(self.mask_dir, f"mask_{mask_id}.png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = np.array(img)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)
        x = np.concatenate([img, mask], axis=-1)

        # Convert to tensor manually (ToTensor doesn't handle 4 channels)
        x = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, normalize

        if self.transform:
            x = self.transform(x)

        return x, fname


class EfficientNetV2S_4ch(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = efficientnet_v2_s(weights=None)

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
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.base.features[0][0] = new_conv

        in_features = self.base.classifier[1].in_features
        self.base.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)


def load_label_mapping(train_csv):
    df = pd.read_csv(train_csv)
    unique_labels = list(df["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return idx_to_label


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 inference with masks as 4th channel")

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.ckpt or .pth)")
    parser.add_argument("--train-csv", type=str, default="data/train_labels.csv", help="Train CSV to recover label mapping")
    parser.add_argument("--img-dir", type=str, default="data/test_data", help="Directory with test images")
    parser.add_argument("--mask-dir", type=str, default="data/test_data", help="Directory with mask images")
    parser.add_argument("--output-csv", type=str, default="submission.csv", help="Path to save predictions CSV")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--img-size", type=int, default=384, help="Image resize size - EfficientNetV2-S uses 384x384")

    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch, "mps") and torch.mps.is_available() else "cpu")

    idx_to_label = load_label_mapping(args.train_csv)
    num_classes = len(idx_to_label)

    model = EfficientNetV2S_4ch(num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")

    if "state_dict" in state:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state["state_dict"].items()}, strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
    ])

    ds = TestMaskedImageDataset(args.img_dir, args.mask_dir, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    records = []
    with torch.no_grad():
        for x, fnames in dl:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            for fname, p in zip(fnames, preds):
                label = idx_to_label[int(p)]
                records.append({"sample_index": fname, "label": label})

    df_out = pd.DataFrame(records)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
