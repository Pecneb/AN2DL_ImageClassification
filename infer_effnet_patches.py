import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


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


def extract_patches(img, patch_size=384, stride=None, min_overlap=0.5):
    """
    Extract overlapping patches from full-resolution image using sliding window.
    
    Args:
        img: PIL Image at full resolution
        patch_size: Size of each patch (square)
        stride: Step size for sliding window. If None, uses patch_size // 2 for 50% overlap
        min_overlap: Minimum overlap ratio (only used if stride is None)
    
    Returns:
        patches: List of PIL Image patches
        positions: List of (left, top, right, bottom) coordinates
    """
    if stride is None:
        stride = int(patch_size * (1 - min_overlap))
    
    w, h = img.size
    patches = []
    positions = []
    
    # Handle small images
    if w < patch_size or h < patch_size:
        # Resize entire image
        resized = img.resize((patch_size, patch_size), Image.BILINEAR)
        patches.append(resized)
        positions.append((0, 0, w, h))
        return patches, positions
    
    # Sliding window across image
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            right = left + patch_size
            bottom = top + patch_size
            
            patch = img.crop((left, top, right, bottom))
            patches.append(patch)
            positions.append((left, top, right, bottom))
    
    # Handle right edge (if image width doesn't divide evenly)
    if (w - patch_size) % stride != 0:
        left = w - patch_size
        for top in range(0, h - patch_size + 1, stride):
            right = w
            bottom = top + patch_size
            
            patch = img.crop((left, top, right, bottom))
            patches.append(patch)
            positions.append((left, top, right, bottom))
    
    # Handle bottom edge (if image height doesn't divide evenly)
    if (h - patch_size) % stride != 0:
        top = h - patch_size
        for left in range(0, w - patch_size + 1, stride):
            right = left + patch_size
            bottom = h
            
            patch = img.crop((left, top, right, bottom))
            patches.append(patch)
            positions.append((left, top, right, bottom))
    
    # Handle bottom-right corner
    if (w - patch_size) % stride != 0 and (h - patch_size) % stride != 0:
        left = w - patch_size
        top = h - patch_size
        right = w
        bottom = h
        
        patch = img.crop((left, top, right, bottom))
        patches.append(patch)
        positions.append((left, top, right, bottom))
    
    return patches, positions


def predict_image_majority_voting(img_path, model, transform, patch_size=384, stride=None, 
                                   device='cpu', class_names=None):
    """
    Predict class for a single image using majority voting on patches.
    
    Args:
        img_path: Path to image
        model: Trained PyTorch model
        transform: Torchvision transforms
        patch_size: Size of patches to extract
        stride: Stride for sliding window
        device: Device to run inference on
        class_names: List of class names for display
    
    Returns:
        final_prediction: Class index (majority vote)
        confidence: Proportion of patches voting for winning class
        all_votes: Dict with vote counts per class
    """
    # Load full-resolution image
    img = Image.open(img_path).convert("RGB")
    
    # Extract patches
    patches, positions = extract_patches(img, patch_size=patch_size, stride=stride)
    
    # Run inference on all patches
    votes = []
    
    for patch in patches:
        # Transform and add batch dimension
        patch_tensor = transform(patch).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(patch_tensor)
            pred = logits.argmax(dim=1).item()
            votes.append(pred)
    
    # Majority voting
    votes = np.array(votes)
    vote_counts = {i: (votes == i).sum() for i in range(len(set(votes)))}
    final_prediction = max(vote_counts, key=vote_counts.get)
    confidence = vote_counts[final_prediction] / len(votes)
    
    return final_prediction, confidence, vote_counts


def predict_image_soft_voting(img_path, model, transform, patch_size=384, stride=None, 
                               device='cpu', class_names=None):
    """
    Predict class for a single image using soft voting (average probabilities) on patches.
    
    Args:
        img_path: Path to image
        model: Trained PyTorch model
        transform: Torchvision transforms
        patch_size: Size of patches to extract
        stride: Stride for sliding window
        device: Device to run inference on
        class_names: List of class names for display
    
    Returns:
        final_prediction: Class index (highest average probability)
        confidence: Average probability for winning class
        avg_probs: Average probabilities for all classes
    """
    # Load full-resolution image
    img = Image.open(img_path).convert("RGB")
    
    # Extract patches
    patches, positions = extract_patches(img, patch_size=patch_size, stride=stride)
    
    # Run inference on all patches and collect probabilities
    all_probs = []
    
    for patch in patches:
        # Transform and add batch dimension
        patch_tensor = transform(patch).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(patch_tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            all_probs.append(probs)
    
    # Average probabilities across all patches
    all_probs = np.array(all_probs)  # [num_patches, num_classes]
    avg_probs = all_probs.mean(axis=0)  # [num_classes]
    
    final_prediction = avg_probs.argmax()
    confidence = avg_probs[final_prediction]
    
    return final_prediction, confidence, avg_probs


def run_inference(
    img_dir,
    output_csv,
    checkpoint_path,
    num_classes=4,
    patch_size=384,
    stride=None,
    voting_method='majority',
    batch_size=16,
    device='auto',
):
    """
    Run inference on all images in a directory using patch-based approach.
    
    Args:
        img_dir: Directory containing test images
        output_csv: Path to save predictions CSV
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes
        patch_size: Size of patches
        stride: Stride for sliding window (None = patch_size // 2)
        voting_method: 'majority' or 'soft'
        batch_size: Batch size for processing patches
        device: 'auto', 'cuda', 'mps', or 'cpu'
    """
    # Setup device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"PATCH-BASED INFERENCE AT FULL RESOLUTION")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Patch size: {patch_size}×{patch_size}")
    print(f"Stride: {stride if stride else patch_size // 2} (overlap: {1 - (stride or patch_size//2)/patch_size:.1%})")
    print(f"Voting method: {voting_method}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = EfficientNetV2S(num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract state dict (handle Lightning checkpoints)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix from Lightning checkpoints
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!\n")
    
    # Transforms (same as validation, no augmentation)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.startswith('img_') and f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Class names
    class_names = ["Luminal A", "Luminal B", "HER2(+)", "Triple Negative"]
    
    # Run inference
    results = []
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(img_dir, img_name)
        
        if voting_method == 'majority':
            pred, conf, votes = predict_image_majority_voting(
                img_path, model, transform, patch_size, stride, device, class_names
            )
        else:  # soft voting
            pred, conf, probs = predict_image_soft_voting(
                img_path, model, transform, patch_size, stride, device, class_names
            )
        
        results.append({
            'sample_index': img_name,
            'label': class_names[pred],
            # 'confidence': conf,
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Inference Complete!")
    print(f"{'='*60}")
    print(f"Processed: {len(image_files)} images")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}\n")
    
    # Print class distribution
    print("Predicted class distribution:")
    for class_name in class_names:
        count = (df['label'] == class_name).sum()
        print(f"  {class_name}: {count} ({count/len(df)*100:.1f}%)")
    
    print(f"\nAverage confidence: {df['confidence'].mean():.3f}")
    print(f"Min confidence: {df['confidence'].min():.3f}")
    print(f"Max confidence: {df['confidence'].max():.3f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Patch-based inference for EfficientNetV2-S at full resolution"
    )
    
    parser.add_argument(
        "--img-dir",
        type=str,
        required=True,
        help="Directory with test images (full resolution preprocessed)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="submission.csv",
        help="Path to save predictions CSV",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=4,
        help="Number of classes",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=384,
        help="Size of patches to extract",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window (default: patch_size // 2 for 50%% overlap)",
    )
    parser.add_argument(
        "--voting-method",
        type=str,
        default="majority",
        choices=["majority", "soft"],
        help="Voting method: majority (hard voting) or soft (average probabilities)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing patches",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for inference",
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    run_inference(
        img_dir=args.img_dir,
        output_csv=args.output_csv,
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        stride=args.stride,
        voting_method=args.voting_method,
        batch_size=args.batch_size,
        device=args.device,
    )
