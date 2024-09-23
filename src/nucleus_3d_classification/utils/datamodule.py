# TODO: FIX THIS, and add docstrings

import os
import json
import torch
from torch.utils.data import DataLoader
# import torch.nn.functional as F
import tifffile as tiff
from torchvision import transforms
# import pytorch_lightning as pl
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
from utils.padding import pad
import lightning as L

def match_labels_to_images(labels_dict, crop_dir, label_to_directory_file):
    """
    Matches label files to image directories more robustly by checking image existence
    and ensuring both directories have matching files.
    
    Args:
        labels_dict (dict): Dictionary with label files and their metadata.
        crop_dir (str): Directory containing cropped images.
        label_to_directory_file (dict): A dictionary mapping label files to image directories: 
            Keys are label files and values are image directories.
            The file is expected to be in the crop_dir.

    Returns:
        label_to_directory (dict): A mapping of label files to image directories.
    """
    label_to_directory = {}

    # Get list of image directories
    image_directories = sorted([d for d in os.listdir(crop_dir) if os.path.isdir(os.path.join(crop_dir, d))])
    print(f"Found {len(image_directories)} image directories.")

    # Iterate over the label files
    print(f"{len(labels_dict)} label files found: {list(labels_dict.keys())}")

    # Open the label_to_directory_file, in the crop_dir
    with open(os.path.join(crop_dir, label_to_directory_file)) as f:
        label_to_directory_file = json.load(f)
        print(f"Loaded label_to_directory_file: {label_to_directory_file}")
        

    for label_file, label_content in labels_dict.items():
        label_name = label_file.split('.')[0]  # Remove file extension
        matched_directory = None

        # Matching based off of label_to_directory_file
        if label_file in label_to_directory_file.keys():
            matched_directory = label_to_directory_file[label_file]
            print(f"Matched label file {label_file} with directory {matched_directory}.")
            label_to_directory[label_file] = matched_directory
            continue
        
        # # Try matching based on folder naming convention
        # for directory in image_directories:
        #     # Check if label name is in directory name
        #     if label_name in directory:
        #         matched_directory = directory
        #         break

        # Validate image existence and log missing images
        if matched_directory:
            image_path = os.path.join(crop_dir, matched_directory)
            if not os.path.exists(image_path):
                print(f"Warning: Image directory {image_path} does not exist.")
            else:
                print(f"Matched label file {label_file} with directory {matched_directory}.")
                label_to_directory[label_file] = matched_directory
        else:
            print(f"Warning: No matching directory found for label file {label_file}.")

    return label_to_directory

def load_labels(label_dir):
    """
    Load all label files in a directory with error handling.
    
    Args:
        label_dir (str): Directory containing the label files.
        
    Returns:
        dict: A dictionary containing label data loaded from JSON files.
    """
    labels_dict = {}

    try:
        for file in os.listdir(label_dir):
            if file.endswith('.json'):
                label_path = os.path.join(label_dir, file)
                try:
                    with open(label_path, 'r') as f:
                        labels_dict[file] = json.load(f)
                except FileNotFoundError:
                    print(f"Error: Label file {label_path} not found.")
                except json.JSONDecodeError as e:
                    print(f"Error: Failed to parse {label_path}. Invalid JSON format. {e}")
                except Exception as e:
                    print(f"Error: An unexpected error occurred while reading {label_path}. {e}")
                    
    except FileNotFoundError:
        print(f"Error: Label directory {label_dir} not found.")
    except PermissionError:
        print(f"Error: Permission denied to access {label_dir}.")
    except Exception as e:
        print(f"Error: An unexpected error occurred while accessing {label_dir}. {e}")
    
    return labels_dict  # Return the labels dictionary even if some files failed to load

# LightningDataModule to handle dataset and dataloading logic
class CustomDataModule(L.LightningDataModule):
    def __init__(self, setup_file=None, root_dir=None, crop_dir=None, label_dir=None, 
                label_to_directory=None, target_size=[34,164,174], batch_size: int = 16,
                num_workers=0, train_image_names='Hoxb5', val_image_names=['c0_0-68_1000', 'c0_0-68_950'],
                test_image_names='c0_0-55'): # TODO: # Change the default to None later ?
        super().__init__()
        self.root_dir = root_dir # root dir is useless here rn TODO: Remove?
        self.crop_dir = crop_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.setup_file = setup_file
        self.label_to_directory = label_to_directory

        self.train_image_names = train_image_names
        self.val_image_names = val_image_names
        self.test_image_names = test_image_names

        self.transform = transforms.Compose([
            pad(self.target_size)
        ])

        self.intensities = {}
        self.labels = {}

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def prepare_data(self):
        """
        Prepares the data by loading setup file if it exists, labels, handling errors, and matching labels to images.
        """
        # Error handling for setup file and directories
        assert self.setup_file is not None or (self.root_dir is not None and self.crop_dir is not None and self.label_dir is not None and self.label_to_directory is not None), \
            'Either setup_file or root_dir, crop_dir, label_dir, and label_to_directory must be provided'

        # Load setup information from file
        if self.setup_file:
            print(f'Loading setup information from {self.setup_file}')
            with open(self.setup_file) as f:
                setup_info = json.load(f)
                self.root_dir = setup_info['root_dir']
                self.crop_dir = setup_info['crop_dir']
                self.label_dir = setup_info['label_dir']
                self.target_size = setup_info['target_size']
                self.batch_size = setup_info['batch_size']
                self.train_image_names = setup_info['train_image_names']
                self.val_image_names = setup_info['val_image_names']
                self.test_image_names = setup_info['test_image_names']
                self.label_to_directory = setup_info['label_to_directory']
                self.num_workers = setup_info['num_workers'] 

        # Load label files with error handling
        labels_dict = load_labels(self.label_dir)

        # Match labels to directories using the robust matching method
        label_to_directory = match_labels_to_images(labels_dict, self.crop_dir, label_to_directory_file=self.label_to_directory)

        # Filter out labels with necessary information and load data
        new_labels_dict = {label_file: {k: v['ctype'] for k, v in content.items()} for label_file, content in labels_dict.items()}
        self._load_data(new_labels_dict, label_to_directory)

    def _load_data(self, new_labels_dict, label_to_directory):
        """
        Load data from the image directories and labels, and store them in the intensities and labels dictionaries.
        Args:
            new_labels_dict (dict): A dictionary containing label data with necessary information.
            label_to_directory (dict): A mapping of label files to image directories.
        """
        # Create intensities and labels directly
        for label_file, image_dir in label_to_directory.items():
            name = label_file.split('.')[0]
            self.intensities[name] = []
            self.labels[name] = []

            for image in os.listdir(f'{self.crop_dir}{image_dir}'):
                # THIS ASSUMES CROPS ARE IN A FOLDER, WHICH HAS SUBFOLDERS FOR EACH IMAGE WITH THE CROPS
                if image.endswith('.tif'):
                    mask_id = (image.split('.')[0].split('_')[-1])
                    if mask_id not in new_labels_dict[label_file].keys(): # Skip if mask_id not in labels
                        continue
                    label = new_labels_dict[label_file][mask_id]
                    if label != 'unknown': # Skip if label is unknown

                        loaded_image = tiff.imread(f'{self.crop_dir}{image_dir}/{image}')
                        loaded_image = torch.from_numpy(loaded_image).float() # Convert to tensor
                        if self.transform:
                            loaded_image = self.transform(loaded_image)
                            # Add dummy channel dimension
                            loaded_image = loaded_image.unsqueeze(0)
                        self.intensities[name].append(loaded_image)

                        # Change label to binary, 0 if negative, 1 if positive
                        label_onehot = 1 if label == 'megakaryocytic' else 0
                        self.labels[name].append(int(label_onehot))

    def setup(self, stage=None):
        """
        Setup the data for training, validation, and testing.
        """
        # Convert intensities and labels to tensors during setup
        if stage == 'fit' or stage is None:
            if isinstance(self.train_image_names, str):
                self.train_data = list(zip(self.intensities[self.train_image_names], self.labels[self.train_image_names]))
            elif len(self.train_image_names) > 1:
                train_intensities = []
                train_labels = []
                for train_image_name in self.train_image_names:
                    train_intensities += self.intensities[train_image_name]
                    train_labels += self.labels[train_image_name]
                self.train_data = list(zip(train_intensities, train_labels))

        if stage == 'validate' or stage is None:
            if isinstance(self.val_image_names, str):
                self.val_data = list(zip(self.intensities[self.val_image_names], self.labels[self.val_image_names]))
            elif len(self.val_image_names) > 1:
                val_intensities = []
                val_labels = []
                for val_image_name in self.val_image_names:
                    val_intensities += self.intensities[val_image_name]
                    val_labels += self.labels[val_image_name]
                self.val_data = list(zip(val_intensities, val_labels))

        if stage == 'test' or stage is None:
            if isinstance(self.test_image_names, str):
                self.test_data = list(zip(self.intensities[self.test_image_names], self.labels[self.test_image_names]))
            elif len(self.test_image_names) > 1:
                test_intensities = []
                test_labels = []
                for test_image_name in self.test_image_names:
                    test_intensities += self.intensities[test_image_name]
                    test_labels += self.labels[test_image_name]
                self.test_data = list(zip(test_intensities, test_labels))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
    
    def train_dataset(self):
        return self.train_data 
    
    def val_dataset(self):
        return self.val_data
    
    def test_dataset(self):
        return self.test_data

def main():
    # Instantiate the DataModule: Alternative way to instantiate the DataModule
    # data_module = CustomDataModule(
    #     root_dir='/Users/agreic/Desktop/Project/Data/Raw',
    #     crop_dir='/Users/agreic/Desktop/Project/Data/Raw/Training/',
    #     label_dir='/Users/agreic/Desktop/Project/Data/Raw/Segmentation/curated_masks/neg_subset_and_mega/curated_before_filter/mask_dicts',
    #     target_size=(34, 164, 174),
    #     batch_size=2, 
    #     train_image_names='Hoxb5',  # Default value
    #     val_image_names=['c0_0-68_1000', 'c0_0-68_950'],  # Default values
    #     test_image_names='c0_0-55', # Default value
    #     label_to_directory='label_to_directory.json'
    # )

    data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')


    # Prepare data and loaders
    data_module.prepare_data()
    data_module.setup()

    # Get the train, validation, and test loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Sanity check, print the first batch of images and labels using the show_slice function
    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels)

if __name__ == '__main__':
    main()