"""
Step 1: Extract Individual Rooftop Crops from Masks
====================================================
This script uses the binary building masks to:
1. Find connected components (individual buildings) in each mask
2. Extract bounding boxes for each building
3. Crop the corresponding region from the JPG image
4. Save individual rooftop crops for manual classification

Only processes images that exist in the Filtered_Images directory
(i.e., images that contain rooftops).
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.join(BASE_DIR, "label", "label")
JPG_DIR = os.path.join(BASE_DIR, "Filtered_Images")
OUTPUT_DIR = os.path.join(BASE_DIR, "Rooftop_Crops")
METADATA_FILE = os.path.join(OUTPUT_DIR, "crop_metadata.json")

# Minimum area threshold (in pixels) to filter out tiny noise regions
MIN_AREA = 400  # ~20x20 pixels minimum
# Padding around bounding box (in pixels) to include some context
PADDING = 10
# Minimum dimension of crop (width or height) to keep
MIN_DIMENSION = 30


def extract_crops():
    """Extract individual rooftop crops from filtered images using masks."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of filtered images (only these have rooftops)
    filtered_images = [f for f in os.listdir(JPG_DIR) if f.lower().endswith('.jpg')]
    print(f"Found {len(filtered_images)} filtered images to process.")
    
    metadata = {}
    total_crops = 0
    skipped_no_mask = 0
    
    for img_idx, jpg_filename in enumerate(sorted(filtered_images)):
        # Derive mask filename: tile1_0_2000.jpg -> tile1_0_2000_label.tif
        base_name = os.path.splitext(jpg_filename)[0]
        mask_filename = f"{base_name}_label.tif"
        mask_path = os.path.join(MASK_DIR, mask_filename)
        jpg_path = os.path.join(JPG_DIR, jpg_filename)
        
        # Check if mask exists
        if not os.path.exists(mask_path):
            skipped_no_mask += 1
            continue
        
        # Read mask (grayscale) and image (color)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(jpg_path)
        
        if mask is None or image is None:
            print(f"Warning: Could not read {jpg_filename} or its mask. Skipping.")
            continue
        
        # Ensure mask is binary (threshold at 127)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        img_h, img_w = image.shape[:2]
        crop_count = 0
        
        # Label 0 is background, start from 1
        for label_id in range(1, num_labels):
            # Get bounding box from stats
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            area = stats[label_id, cv2.CC_STAT_AREA]
            
            # Skip tiny regions (noise)
            if area < MIN_AREA:
                continue
            
            # Skip very small dimensions
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                continue
            
            # Add padding
            x_pad = max(0, x - PADDING)
            y_pad = max(0, y - PADDING)
            w_pad = min(img_w - x_pad, w + 2 * PADDING)
            h_pad = min(img_h - y_pad, h + 2 * PADDING)
            
            # Crop the rooftop region from the original image
            crop = image[y_pad:y_pad + h_pad, x_pad:x_pad + w_pad]
            
            if crop.size == 0:
                continue
            
            # Save crop
            crop_filename = f"{base_name}_roof_{crop_count}.jpg"
            crop_path = os.path.join(OUTPUT_DIR, crop_filename)
            cv2.imwrite(crop_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Store metadata
            metadata[crop_filename] = {
                "source_image": jpg_filename,
                "bbox": [int(x), int(y), int(w), int(h)],
                "bbox_padded": [int(x_pad), int(y_pad), int(w_pad), int(h_pad)],
                "area": int(area),
                "centroid": [float(centroids[label_id][0]), float(centroids[label_id][1])],
                "label": None  # To be filled during manual labeling
            }
            
            crop_count += 1
            total_crops += 1
        
        # Progress update every 50 images
        if (img_idx + 1) % 50 == 0:
            print(f"Processed {img_idx + 1}/{len(filtered_images)} images, "
                  f"{total_crops} crops extracted so far...")
    
    # Save metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 50)
    print("EXTRACTION COMPLETE")
    print("=" * 50)
    print(f"Total images processed: {len(filtered_images) - skipped_no_mask}")
    print(f"Images skipped (no mask): {skipped_no_mask}")
    print(f"Total rooftop crops extracted: {total_crops}")
    print(f"Crops saved to: {OUTPUT_DIR}")
    print(f"Metadata saved to: {METADATA_FILE}")
    
    # Print crop size statistics
    if metadata:
        areas = [v["area"] for v in metadata.values()]
        print(f"\nCrop area statistics:")
        print(f"  Min area: {min(areas)} px")
        print(f"  Max area: {max(areas)} px")
        print(f"  Mean area: {np.mean(areas):.0f} px")
        print(f"  Median area: {np.median(areas):.0f} px")


if __name__ == "__main__":
    extract_crops()
