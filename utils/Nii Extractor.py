import os
import json
import nibabel as nib
import numpy as np
from PIL import Image


def process_dataset(mask_dir, ct_dir, output_masks_dir, output_images_dir, output_json_dir):

    # Define label names
    label_names = {1: 'liver', 2: 'kidney', 3: 'spleen', 4: 'pancreas'}

    # Initialize JSON data
    json_data = []

    # Process each mask file
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.nii') or mask_file.endswith('.nii.gz'):
            mask_path = os.path.join(mask_dir, mask_file)
            if mask_file.endswith('.nii'):
                ct_file = mask_file.replace('.nii', '_0000.nii')
            else:
                ct_file = mask_file.replace('.nii.gz', '_0000.nii.gz')
            ct_path = os.path.join(ct_dir, ct_file)

            # Load mask and CT data
            mask_nii = nib.load(mask_path)
            mask_data = mask_nii.get_fdata()
            ct_nii = nib.load(ct_path)
            ct_data = ct_nii.get_fdata()

            # Process each frame
            for frame in range(mask_data.shape[2]):
                mask_frame = mask_data[:, :, frame]
                unique_labels = np.unique(mask_frame)

                # Check if frame contains only one segmentation label (excluding background)
                if len(unique_labels) == 2 and 0 in unique_labels:
                    label = unique_labels[unique_labels != 0][0]
                    label_name = label_names[label]

                    # Generate filenames
                    mask_filename = f"{os.path.splitext(os.path.splitext(mask_file)[0])[0]}_{frame:03d}_{label_name}.png"
                    ct_filename = mask_filename

                    # Save mask frame as PNG
                    mask_img = Image.fromarray((mask_frame * 255 / label).astype(np.uint8))
                    mask_output_path = os.path.join(output_masks_dir, mask_filename)
                    mask_img.save(mask_output_path)

                    # Save corresponding CT frame as PNG
                    ct_frame = ct_data[:, :, frame]
                    ct_img = Image.fromarray(((ct_frame - ct_frame.min()) / (ct_frame.max() - ct_frame.min()) * 255).astype(np.uint8))
                    ct_output_path = os.path.join(output_images_dir, ct_filename)
                    ct_img.save(ct_output_path)

                    # Add information to JSON data
                    json_data.append({
                        "image_name": ct_filename,
                        "mask_name": mask_filename,
                        "prompt": f"Segment the {label_name} in the abdominal region ",
                        "label": label_name
                    })

                    print(f"Saved frame {frame} from {mask_file} with {label_name} segmentation")

    # Save JSON data
    output_json_path = os.path.join(output_json_dir, "data.json")
    try:
        with open(output_json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=2)
        print(f"JSON file saved to {output_json_path}")
    except PermissionError:
        print(f"Permission denied: Could not write to {output_json_path}")
        # ADDED: Attempt to save to an alternative location
        alternative_path = os.path.join(os.path.expanduser("~"), "Desktop", "abdomen_ct_data.json")
        try:
            with open(alternative_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=2)
            print(f"JSON file saved to alternative location: {alternative_path}")
        except Exception as e:
            print(f"Failed to save JSON file. Error: {str(e)}")

if __name__ == "__main__":
    mask_directory = r"C:\Users\Omer\Desktop\m"
    ct_directory = r"C:\Users\Omer\Desktop\i"
    output_masks_directory = r"C:\Users\Omer\Desktop\masks"
    output_images_directory = r"C:\Users\Omer\Desktop\images"
    output_json_directory = r"C:\Users\Omer\Desktop\anns"

    process_dataset(mask_directory, ct_directory, output_masks_directory, output_images_directory, output_json_directory)