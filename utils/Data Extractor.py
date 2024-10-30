import os
import json

# Define the paths to the dataset
dataset_path = r'C:\Idan\Courses\Medical Data in Deep Learning\Seminar and Project\Project\Datasets\Breast Ultrasound Images'
output_json_file = 'Breast Ultrasound Images.json'

# Define the prompts for each type of image
prompts = {
    "Normal": "Breast ultrasound image with no tumor or abnormalities.",
    "Malignant": "Breast ultrasound image with malignant tumor (cancerous).",
    "Benign": "Breast ultrasound image with benign tumor (non-cancerous)."
}

# Initialize a dictionary to hold the data
data = {}

# Loop through the categories
for category in ['Normal', 'Malignant', 'Benign']:
    category_path = os.path.join(dataset_path, category)
    # Initialize the list for this category
    data[category] = []
    for file_name in os.listdir(category_path):
        if "mask" not in file_name:
            image_name = file_name
            mask_name = file_name.replace('.png', '_mask.png')
            image_path = 'Breast Ultrasound Images/' + category + '/' + image_name
            mask_path = 'Breast Ultrasound Images/' + category + '/' + mask_name
            data[category].append((image_path, mask_path, prompts[category]))

with open(output_json_file, 'w') as f:
    json.dump(data, f, indent=4)
print(f'JSON file has been written to {output_json_file}')

