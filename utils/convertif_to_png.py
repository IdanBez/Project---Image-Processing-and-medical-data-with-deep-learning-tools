# Ensure output directory exists
import cv2
import os


# Define input and output directories
input_dir = '/home/eeproj6/Yasmin/medvlsm-main/data/colondb_polyp/images_tif'  # Replace with your input directory
input_dir = '/home/eeproj6/Yasmin/medvlsm-main/data/colondb_polyp/masks_tif'

output_dir = '/home/eeproj6/Yasmin/medvlsm-main/data/colondb_polyp/masks'  # Replace with your output directory
output_format = 'png'  # Change to 'jpeg' if you prefer JPEG format

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Convert TIFF to desired format
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith(('.tif', '.tiff')):
        # Read the TIFF image
        img = cv2.imread(os.path.join(input_dir, file_name), cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Error loading image {file_name}")
            continue

        # Define output file name and path
        base_name, _ = os.path.splitext(file_name)
        output_file = os.path.join(output_dir, f"{base_name}.{output_format}")

        # Save the image in the desired format
        if output_format == 'png':
            cv2.imwrite(output_file, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif output_format == 'jpeg':
            cv2.imwrite(output_file, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            print(f"Unsupported format: {output_format}")
            continue

        print(f"Converted {file_name} to {output_file}")

print("Conversion complete.")
