import os
from datasets import load_dataset

# Ordner, in dem die Datensätze gespeichert werden sollen
data_folder = 'fine-tuned_model/data'

# Sicherstellen, dass der Ordner existiert
os.makedirs(data_folder, exist_ok=True)

# Funktion zum Laden und Speichern des Datensatzes
def load_and_save_dataset(dataset_name, subset_name=None):
    # Datensatz laden
    if subset_name:
        dataset = load_dataset(dataset_name, subset_name)
    else:
        dataset = load_dataset(dataset_name)

    # Datensätze splitten
    train_data = dataset['train']
    valid_data = dataset['validation']
    test_data = dataset['test']

    # Pfade zum Speichern der Datensätze
    train_path = os.path.join(data_folder, f'{dataset_name}_train')
    valid_path = os.path.join(data_folder, f'{dataset_name}_valid')
    test_path = os.path.join(data_folder, f'{dataset_name}_test')

    # Datensätze speichern
    train_data.save_to_disk(train_path)
    valid_data.save_to_disk(valid_path)
    test_data.save_to_disk(test_path)

    print(f"Datasets saved to {data_folder}")

# XSum Datensatz laden und speichern
load_and_save_dataset('xsum')
