import os
import shutil
import json


def pool_and_merge_annotations(main_dir, output_folder_name="pool_all", output_json_name="combined_annotations.json"):
    # Create the output folder
    pool_all_dir = os.path.join(main_dir, output_folder_name)
    os.makedirs(pool_all_dir, exist_ok=True)

    # Directories for pooled files
    pooled_images_dir = os.path.join(pool_all_dir, 'images')
    pooled_masks_dir = os.path.join(pool_all_dir, 'masks')
    os.makedirs(pooled_images_dir, exist_ok=True)
    os.makedirs(pooled_masks_dir, exist_ok=True)

    combined_annotations = []
    image_index = 0  # To index images

    # Loop through each folder in the main directory (assuming each is a 'db' folder)
    for db_folder in os.listdir(main_dir):
        db_path = os.path.join(main_dir, db_folder)

        if os.path.isdir(db_path):
            anns_dir = os.path.join(db_path, 'anns')
            images_dir = os.path.join(db_path, 'images')
            masks_dir = os.path.join(db_path, 'masks')

            if not (os.path.exists(anns_dir) and os.path.exists(images_dir) and os.path.exists(masks_dir)):
                print(f"Skipping {db_folder} due to missing subdirectories.")
                continue

            # Loop through annotations and corresponding images/masks
            for ann_file in os.listdir(anns_dir):
                ann_path = os.path.join(anns_dir, ann_file)

                # Load the annotation JSON file
                with open(ann_path, 'r') as f:
                    annotation_data_list = json.load(f)

                if not isinstance(annotation_data_list, list):
                    print(f"Skipping file {ann_file} as it does not contain a list of annotations.")
                    continue

                # Iterate over the list of annotations
                for annotation_data in annotation_data_list:
                    # Get the corresponding image and mask filenames
                    original_image_name = annotation_data.get('img_name')
                    original_mask_name = annotation_data.get('mask_name')

                    if original_image_name and original_mask_name:
                        # Create a new filename using the parent folder name and image index
                        new_image_name = f"{db_folder}_{image_index:04d}.png"
                        new_mask_name = f"{db_folder}_{image_index:04d}_mask.png"

                        # Update the annotation with the new filenames
                        annotation_data['img_name'] = new_image_name
                        annotation_data['mask_name'] = new_mask_name

                        # Add to the combined annotations list
                        combined_annotations.append(annotation_data)

                        # Copy and rename the image and mask to the pooled folder
                        src_image_path = os.path.join(images_dir, original_image_name)
                        src_mask_path = os.path.join(masks_dir, original_mask_name)

                        if os.path.exists(src_image_path):
                            shutil.copy(src_image_path, os.path.join(pooled_images_dir, new_image_name))
                        else:
                            print(f"Image {original_image_name} not found in {images_dir}.")

                        if os.path.exists(src_mask_path):
                            shutil.copy(src_mask_path, os.path.join(pooled_masks_dir, new_mask_name))
                        else:
                            print(f"Mask {original_mask_name} not found in {masks_dir}.")

                        image_index += 1
                    else:
                        print(f"Annotation missing image or mask name in file: {ann_file}")

    # Save the combined annotations JSON file
    combined_annotations_path = os.path.join(pool_all_dir, output_json_name)
    with open(combined_annotations_path, 'w') as f:
        json.dump(combined_annotations, f, indent=4)

    print(f"Pooling and merging completed. All files are in {pool_all_dir}")


if __name__ == "__main__":
    main_dir = '/home/eeproj6/Yasmin/medvlsm-main/data'
    pool_and_merge_annotations(main_dir)


