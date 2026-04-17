import os
import cv2
import shutil

def filter_empty_images(mask_folder, jpg_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all mask files
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith('.tif')]
    total_masks = len(mask_files)
    print(f"I found {total_masks} mask files to process.") # Add this line!
    
    valid_count = 0
    empty_count = 0

    print(f"Scanning {total_masks} mask files...")

    for mask_filename in mask_files:
        mask_path = os.path.join(mask_folder, mask_filename)
        
        # 1. Read the mask in grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Count non-zero (white) pixels
        # If the count is 0, the image is entirely black (no roofs)
        if cv2.countNonZero(mask) == 0:
            empty_count += 1
            continue  # Skip to the next file

        # 3. If it has roofs, find the matching .jpg file
        # This replaces '_label.tif' with '.jpg' to find the exact match
        # Adjust the string replacement if your naming convention is slightly different
        jpg_filename = mask_filename.replace('_label.tif', '.jpg')
        jpg_path = os.path.join(jpg_folder, jpg_filename)

        # 4. Copy the valid .jpg to the new output folder
        if os.path.exists(jpg_path):
            output_path = os.path.join(output_folder, jpg_filename)
            shutil.copy2(jpg_path, output_path)
            valid_count += 1
        else:
            print(f"Warning: Found valid mask but missing JPG: {jpg_filename}")

    print("--- Filtering Complete ---")
    print(f"Images with roofs copied: {valid_count}")
    print(f"Empty images skipped: {empty_count}")
    print(f"Your clean dataset is located in: {output_folder}")

# --- Configuration ---
# Set your folder paths here
MASK_DIR = "label/label"    # Folder with tile1_0_9200_label.tif
JPG_DIR = "jpg_images"  # Folder with tile1_0_9200.jpg
OUTPUT_DIR = "Filtered_Images"   # Where you want the valid images to go

# Run the function
filter_empty_images(MASK_DIR, JPG_DIR, OUTPUT_DIR)