"""
Preprocessing script for breast cancer histopathology images.

This script:
1. Detects and removes artifacts (green/orange/brown/white markers)
2. Applies masks to isolate tissue regions
3. Crops to ROI (bounding box of cleaned mask)
4. Saves preprocessed images ready for training

Usage:
    python preprocess_dataset.py --input-dir data/train_data --output-dir data/train_data_preprocessed
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess histopathology images with mask application and cropping")
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw images and masks"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save preprocessed images"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to CSV file with labels (optional, will be copied to output dir)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding around ROI bounding box (default: 20)"
    )
    parser.add_argument(
        "--min-tissue-ratio",
        type=float,
        default=0.001,
        help="Minimum ratio of tissue pixels to total pixels (default: 0.001)"
    )
    parser.add_argument(
        "--max-artifact-ratio",
        type=float,
        default=0.15,
        help="Maximum ratio of artifact pixels to total pixels (default: 0.15, rejects Shrek-like images)"
    )
    
    return parser.parse_args()


def get_artifact_masks(hsv_img):
    """Detect artifact colors in HSV space."""
    # Artifact color ranges (expanded to catch more variations)
    LOWER_GREEN = np.array([25, 40, 40])
    UPPER_GREEN = np.array([95, 255, 255])
    
    LOWER_ORANGE = np.array([5, 50, 50])
    UPPER_ORANGE = np.array([35, 255, 255])
    
    LOWER_WHITE = np.array([0, 0, 200])
    UPPER_WHITE = np.array([180, 30, 255])
    
    LOWER_BROWN = np.array([0, 60, 20])
    UPPER_BROWN = np.array([25, 255, 150])
    
    # Detect each artifact type
    m_green = cv2.inRange(hsv_img, LOWER_GREEN, UPPER_GREEN)
    m_orange = cv2.inRange(hsv_img, LOWER_ORANGE, UPPER_ORANGE)
    m_white = cv2.inRange(hsv_img, LOWER_WHITE, UPPER_WHITE)
    m_brown = cv2.inRange(hsv_img, LOWER_BROWN, UPPER_BROWN)
    
    # Combine all artifact masks
    artifacts = cv2.bitwise_or(m_green, m_orange)
    artifacts = cv2.bitwise_or(artifacts, m_white)
    artifacts = cv2.bitwise_or(artifacts, m_brown)
    
    return artifacts


def clean_and_crop_image(img_path, mask_path, padding=20, min_tissue_ratio=0.001, max_artifact_ratio=0.15):
    """
    Apply mask, remove artifacts, and crop to ROI.
    
    Args:
        img_path: Path to input image
        mask_path: Path to mask image
        padding: Pixels to add around ROI
        min_tissue_ratio: Minimum tissue fraction required
        max_artifact_ratio: Maximum artifact fraction allowed (rejects bad images)
    
    Returns:
        final_img: Preprocessed RGB image (cropped to ROI)
        success: Boolean indicating if preprocessing succeeded
        stats: Dict with preprocessing statistics
    """
    # Load image and mask
    img = cv2.imread(img_path)
    if img is None:
        return None, False
    
    # Load mask (or create full mask if missing)
    if os.path.exists(mask_path):
        original_mask = cv2.imread(mask_path, 0)
    else:
        original_mask = np.full(img.shape[:2], 255, dtype=np.uint8)
    
    # Convert to HSV for artifact detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect artifacts
    bad_pixels = get_artifact_masks(hsv)
    
    # Count artifact pixels BEFORE dilation
    total_pixels = img.shape[0] * img.shape[1]
    artifact_count = cv2.countNonZero(bad_pixels)
    artifact_ratio = artifact_count / total_pixels
    
    # Check if too many artifacts (like Shrek images)
    if artifact_ratio > max_artifact_ratio:
        stats = {
            'artifact_pixels': artifact_count,
            'total_pixels': total_pixels,
            'artifact_ratio': artifact_ratio,
            'reason': f'Too many artifacts ({artifact_ratio:.2%} > {max_artifact_ratio:.2%})'
        }
        return None, False, stats
    
    # Dilate artifact mask to catch edges
    dilation_kernel = np.ones((5, 5), np.uint8)
    bad_pixels_expanded = cv2.dilate(bad_pixels, dilation_kernel, iterations=3)
    
    # Clean mask: remove artifacts from tissue mask
    clean_mask = cv2.bitwise_and(original_mask, cv2.bitwise_not(bad_pixels_expanded))
    
    # Apply mask to image (zero out background and artifacts)
    masked_img = cv2.bitwise_and(img, img, mask=clean_mask)
    
    # Find bounding box of tissue region
    y, x = np.where(clean_mask > 0)
    
    if len(y) == 0:
        # No tissue found
        stats = {
            'artifact_pixels': artifact_count,
            'total_pixels': total_pixels,
            'artifact_ratio': artifact_ratio,
            'tissue_pixels': 0,
            'tissue_ratio': 0.0,
            'reason': 'No tissue remaining after cleaning'
        }
        return None, False, stats
    
    # Check if enough tissue remains
    tissue_pixels = len(y)
    tissue_ratio = tissue_pixels / total_pixels
    if tissue_ratio < min_tissue_ratio:
        stats = {
            'artifact_pixels': artifact_count,
            'total_pixels': total_pixels,
            'artifact_ratio': artifact_ratio,
            'tissue_pixels': tissue_pixels,
            'tissue_ratio': tissue_ratio,
            'reason': f'Too little tissue ({tissue_ratio:.4%} < {min_tissue_ratio:.4%})'
        }
        return None, False, stats
    
    # Crop to bounding box with padding
    y_min, y_max = max(0, y.min() - padding), min(img.shape[0], y.max() + padding)
    x_min, x_max = max(0, x.min() - padding), min(img.shape[1], x.max() + padding)
    
    final_img = masked_img[y_min:y_max, x_min:x_max]
    
    # Convert BGR to RGB
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    
    # Success statistics
    stats = {
        'artifact_pixels': artifact_count,
        'total_pixels': total_pixels,
        'artifact_ratio': artifact_ratio,
        'tissue_pixels': tissue_pixels,
        'tissue_ratio': tissue_ratio,
        'crop_size': f'{final_img.shape[0]}x{final_img.shape[1]}',
        'reason': 'Success'
    }
    
    return final_img, True, stats


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Preprocessing Dataset")
    print(f"{'='*60}")
    print(f"Input directory:     {args.input_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Padding:             {args.padding}px")
    print(f"Min tissue ratio:    {args.min_tissue_ratio}")
    print(f"Max artifact ratio:  {args.max_artifact_ratio} (rejects Shrek-like images)")
    print(f"{'='*60}\n")
    
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith('img_') and f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Track statistics
    successful = 0
    failed = 0
    failed_images = []
    rejection_reasons = {
        'high_artifacts': 0,
        'no_tissue': 0,
        'low_tissue': 0,
        'corrupted': 0
    }
    
    # Process each image
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(args.input_dir, img_name)
        
        # Determine mask path (img_XXXX.png -> mask_XXXX.png)
        mask_name = img_name.replace("img_", "mask_")
        mask_path = os.path.join(args.input_dir, mask_name)
        
        # Process image
        try:
            processed_img, success, stats = clean_and_crop_image(
                img_path, 
                mask_path, 
                padding=args.padding,
                min_tissue_ratio=args.min_tissue_ratio,
                max_artifact_ratio=args.max_artifact_ratio
            )
            
            if success and processed_img is not None:
                # Save preprocessed image
                output_path = os.path.join(args.output_dir, img_name)
                cv2.imwrite(output_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                successful += 1
            else:
                failed += 1
                failed_images.append((img_name, stats))
                
                # Categorize failure reason
                if 'Too many artifacts' in stats.get('reason', ''):
                    rejection_reasons['high_artifacts'] += 1
                elif 'No tissue' in stats.get('reason', ''):
                    rejection_reasons['no_tissue'] += 1
                elif 'Too little tissue' in stats.get('reason', ''):
                    rejection_reasons['low_tissue'] += 1
        except Exception as e:
            failed += 1
            failed_images.append((img_name, {'reason': f'Error: {str(e)}'}))
            rejection_reasons['corrupted'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"✅ Successful: {successful}/{len(image_files)} ({successful/len(image_files)*100:.1f}%)")
    print(f"❌ Failed:     {failed}/{len(image_files)} ({failed/len(image_files)*100:.1f}%)")
    
    if failed > 0:
        print(f"\nRejection breakdown:")
        print(f"  🎨 Too many artifacts (Shrek-like): {rejection_reasons['high_artifacts']}")
        print(f"  🚫 No tissue after cleaning:        {rejection_reasons['no_tissue']}")
        print(f"  📉 Too little tissue:               {rejection_reasons['low_tissue']}")
        print(f"  💥 Corrupted/error:                 {rejection_reasons['corrupted']}")
        
        print(f"\nFailed images (showing first 10):")
        for img_name, stats in failed_images[:10]:
            reason = stats.get('reason', 'Unknown')
            artifact_ratio = stats.get('artifact_ratio', 0)
            if artifact_ratio > 0:
                print(f"  - {img_name}: {reason} (artifacts: {artifact_ratio:.2%})")
            else:
                print(f"  - {img_name}: {reason}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")
    
    # Create new cleaned CSV if provided (original CSV is not modified)
    if args.csv_path and os.path.exists(args.csv_path):
        df = pd.read_csv(args.csv_path)
        original_count = len(df)
        
        # Filter out failed images
        col_name = 'sample_index' if 'sample_index' in df.columns else 'image_id'
        failed_image_names = [name for name, _ in failed_images]
        df_clean = df[~df[col_name].isin(failed_image_names)]
        
        # Generate new CSV filename with _cleaned suffix
        csv_basename = os.path.basename(args.csv_path)
        csv_name, csv_ext = os.path.splitext(csv_basename)
        output_csv = os.path.join(args.output_dir, f"{csv_name}_preprocessed{csv_ext}")
        
        # Save new cleaned CSV
        df_clean.to_csv(output_csv, index=False)
        
        print(f"\n{'='*60}")
        print(f"CSV Created:")
        print(f"  Original CSV:     {args.csv_path}")
        print(f"  Original samples: {original_count}")
        print(f"  Cleaned samples:  {len(df_clean)}")
        print(f"  Removed:          {original_count - len(df_clean)}")
        print(f"  New CSV saved to: {output_csv}")
        print(f"{'='*60}")
    
    print(f"\nPreprocessed images saved to: {args.output_dir}")
    print(f"Ready for training!\n")


if __name__ == "__main__":
    main()
