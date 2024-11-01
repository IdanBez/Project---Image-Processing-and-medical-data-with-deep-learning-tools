import os
import glob

# Define the path to your dataset directory
dataset_dir = r'C:\Idan\Courses\Medical Data in Deep Learning\Seminar and Project\Project\Datasets\Skin Lesions\Test\Images'

# Use glob to find all files ending with 'superpixels.png'
files_to_delete = glob.glob(os.path.join(dataset_dir, '*superpixels.png'))

# Loop through the list of files and delete them
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("Deletion process complete.")
