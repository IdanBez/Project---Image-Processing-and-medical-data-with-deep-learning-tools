import json
import random
import os

def split_json(file_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits a JSON dataset into train, validation, and test sets.

    Args:
        file_path (str): Path to the input JSON file.
        train_ratio (float): Proportion of data for training set.
        val_ratio (float): Proportion of data for validation set.
        test_ratio (float): Proportion of data for test set.

    Returns:
        None: Saves three JSON files: train.json, val.json, and test.json.
    """
    # Check that the sum of the ratios equals 1
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1."

    # Read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Shuffle the data to randomize the order
    random.shuffle(data)

    # Calculate the split indices
    total_len = len(data)
    train_end = int(train_ratio * total_len)
    val_end = train_end + int(val_ratio * total_len)

    # Split the data into train, validation, and test sets
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # Define the output filenames
    base_name = os.path.splitext(file_path)[0]  # Get the base name without the .json extension
    #gets directory name:
    folder_path = os.path.dirname(file_path)
    train_file = os.path.join(folder_path,"train.json")
    val_file = os.path.join(folder_path, "val.json")
    test_file = os.path.join(folder_path,  "test.json")

    # Save the split data into separate JSON files
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=4)
    
    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=4)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Data split complete! Files saved as {train_file}, {val_file}, and {test_file}.")

# Example usage
file_path = '/home/eeproj6/Yasmin/medvlsm-main/data/ChestXRay_Pneumothorax/src/Training/anns/all.json'  # Replace with your actual JSON file path
split_json(file_path, train_ratio=0.8, val_ratio=0.2, test_ratio=0)
