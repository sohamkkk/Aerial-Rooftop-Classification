import os
from PIL import Image

def batch_convert_tif_to_jpg(input_folder, output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.tif', '.tiff'))]
    total_files = len(files)
    
    print(f"Found {total_files} .tif files. Starting conversion...")

    count = 0
    for filename in files:
        input_path = os.path.join(input_folder, filename)
        
        # Create the new filename by replacing .tif with .jpg
        name_without_ext = os.path.splitext(filename)[0]
        new_filename = f"{name_without_ext}.jpg"
        output_path = os.path.join(output_folder, new_filename)

        try:
            # Open the .tif image
            with Image.open(input_path) as img:
                # Convert to RGB (Crucial: TIFs are often RGBA or CMYK, which JPG doesn't support)
                rgb_im = img.convert('RGB')
                
                # Save as JPG with high quality
                rgb_im.save(output_path, 'JPEG', quality=95)
                
            count += 1
            # Print progress every 100 images
            if count % 100 == 0:
                print(f"Converted {count}/{total_files} images...")
                
        except Exception as e:
            print(f"Failed to convert {filename}: {e}")

    print(f"Conversion complete! Successfully converted {count} out of {total_files} images.")

# --- Configuration ---
# Replace these strings with the actual paths on your computer
INPUT_DIR = "images/images" 
OUTPUT_DIR = "jpg_images"

# Run the function
batch_convert_tif_to_jpg(INPUT_DIR, OUTPUT_DIR)