import os
from PIL import Image


def convert_tif_to_tiff(folder_path):
    # Traverse the directory structure
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".tiff"):
                tif_file_path = os.path.join(root, file)

                # Generate the new .tiff file name
                tiff_file_path = os.path.join(root, file[:-4] + '.tiff')

                # Open the .tif file
                with Image.open(tif_file_path) as img:
                    # Convert and save as .tiff
                    img.save(tiff_file_path, format='TIFF')
                    print(f"Converted: {tif_file_path} -> {tiff_file_path}")

                # Optional: If you want to delete the original .tif file, uncomment the following line
                # os.remove(tif_file_path)
                # print(f"Deleted original .tif file: {tif_file_path}")


if __name__ == "__main__":
    folder_path = '/home/eeproj6/Yasmin/medvlsm-main/data/colondb_polyp'
    #folder_path = input("Enter the path to the folder containing .tif images: ")
    convert_tif_to_tiff(folder_path)
    print("Conversion completed.")
