import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the .nii file
file_path = 'path_to_your_file.nii'
nii_data = nib.load(file_path)

# Get the data array from the nii file
mask_data = nii_data.get_fdata()

# Check the shape of the mask data
print(f"Shape of the mask data: {mask_data.shape}")

# Get unique labels in the mask (each label corresponds to a different class)
unique_labels = np.unique(mask_data)
print(f"Unique labels in the mask: {unique_labels}")

# Visualize one of the slices of the mask (e.g., slice number 50)
slice_num = 50
plt.imshow(mask_data[:, :, slice_num], cmap='gray')
plt.title(f'Slice {slice_num} of the Mask')
plt.colorbar()
plt.show()
