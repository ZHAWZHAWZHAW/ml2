import os
from datasets import load_dataset

# Folder to save the datasets
data_folder = 'fine-tuned_model/data'

# Ensure the folder exists
os.makedirs(data_folder, exist_ok=True)

# Function to load and save the dataset
def load_and_save_dataset(dataset_name, subset_name=None):
    # Load dataset
    if subset_name:
        dataset = load_dataset(dataset_name, subset_name)
    else:
        dataset = load_dataset(dataset_name)

    # Split datasets
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    # Paths to save the datasets
    train_path = os.path.join(data_folder, f'{dataset_name}_train')
    valid_path = os.path.join(data_folder, f'{dataset_name}_valid')
    test_path = os.path.join(data_folder, f'{dataset_name}_test')

    # Save datasets
    train_data.save_to_disk(train_path)
    valid_data.save_to_disk(valid_path)
    test_data.save_to_disk(test_path)

    print(f"Datasets saved to {data_folder}")

# Load and save the XSum dataset
load_and_save_dataset('xsum')
