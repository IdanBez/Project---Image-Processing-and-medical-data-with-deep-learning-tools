import os
import pandas as pd
import json
import cv2
import numpy as np

# Load CSV file (Assuming the CSV is called 'data.csv')
csv_file = '/path_to_your_csv_file.csv'
data = pd.read_csv(csv_file)

# Path to masks folder
mask_folder = '/path_to_mask_folder'


# Define a helper function to calculate the bounding box from the mask
def get_bbox(mask_image):
    # Find non-zero regions in the mask
    mask = cv2.imread(mask_image, cv2.IMREAD_GRAYSCALE)
    coords = cv2.findNonZero(mask)  # Get all non-zero points
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        return [x, y, x + w, y + h]
    else:
        # If no non-zero points found, return an empty bbox
        return [0, 0, 0, 0]


# Create a list to hold all annotations
annotations = []

# Loop through the data rows
for idx, row in data.iterrows():
    # Get the new filename and ImageId
    img_name = row['new_filename']
    img_id = row['ImageId']

    # Find the corresponding mask
    mask_name = os.path.join(mask_folder, img_name.replace('.png', '_Segmentation.png'))

    if os.path.exists(mask_name):
        # Calculate bbox
        bbox = get_bbox(mask_name)

        # Build the annotation structure
        annotation = {
            "bbox": bbox,
            "cat": row['has_pneumo'],
            "segment_id": img_id,
            "img_name": img_name,
            "mask_name": os.path.basename(mask_name),
            "sentences": [
                {
                    "idx": 0,
                    "sent_id": idx,
                    "sent": ""
                }
            ],
            "prompts": {
                "p0": "",
                "p1": "pneumonia" if row['has_pneumo'] == 1 else ""
            },
            "sentences_num": 1
        }

        # Append to the annotations list
        annotations.append(annotation)

# Write to JSON file
output_file = '/path_to_output_annotations.json'
with open(output_file, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f"Annotations saved to {output_file}")
