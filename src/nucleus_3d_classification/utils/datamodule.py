# TODO: FIX THIS, and add docstrings

import os
import json
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tifffile as tiff
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from padding import pad


# LightningDataModule to handle dataset and dataloading logic
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, crop_dir, label_dir, target_size, batch_size: int = 16, num_workers=0,
                 train_image_names='Hoxb5', val_image_names=['c0_0-68_1000', 'c0_0-68_950'], test_image_names='c0_0-55'): 
        # TODO: # Change the default to None later
        super().__init__()
        self.root_dir = root_dir
        self.crop_dir = crop_dir
        self.label_dir = label_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        # Load label dictionaries
        labels_dict = {}
        for file in os.listdir(self.label_dir):
            if file.endswith('.json'):
                with open(os.path.join(self.label_dir, file)) as f:
                    labels_dict[file] = json.load(f)
        labels_dict = dict(sorted(labels_dict.items()))

        # Match label files with image directories
        label_to_directory = {}
        directory_list = sorted([d for d in os.listdir(f'{self.crop_dir}') if 'cutout' in d])
        for label_name in labels_dict.keys():
            for directory in directory_list:
                if str(label_name).split('.')[0] in directory:
                    label_to_directory[label_name] = directory
        
        # Filter labels with necessary information
        new_labels_dict = {}
        for label_file, content in labels_dict.items():
            temp_dict = {key: value['ctype'] for key, value in content.items()}
            new_labels_dict[label_file] = temp_dict

        self._load_data(new_labels_dict, label_to_directory)

    def _load_data(self, new_labels_dict, label_to_directory):
        # Create intensities and labels directly
        for label_file, image_dir in label_to_directory.items():
            name = label_file.split('.')[0]
            self.intensities[name] = []
            self.labels[name] = []

            for image in os.listdir(f'{self.crop_dir}{image_dir}'):
                if image.endswith('.tif'):
                    mask_id = (image.split('.')[0].split('_')[-1])
                    if mask_id not in new_labels_dict[label_file].keys():
                        continue
                    label = new_labels_dict[label_file][mask_id]
                    if label != 'unknown':
                        loaded_image = tiff.imread(f'{self.crop_dir}{image_dir}/{image}')
                        loaded_image = torch.from_numpy(loaded_image).float()
                        if self.transform:
                            loaded_image = self.transform(loaded_image)
                        self.intensities[name].append(loaded_image)
                        # Change label to binary, 0 if negative, 1 if positive
                        label_onehot = 1 if label == 'megakaryocytic' else 0
                        self.labels[name].append(int(label_onehot))

    def setup(self, stage=None):
        # Convert intensities and labels to tensors during setup
        if stage == 'fit' or stage is None:
            if len(self.train_image_names) == 1:
                self.train_data = list(zip(self.intensities[self.train_image_name], self.labels[self.train_image_name]))
            elif len(self.train_image_names) > 1:
                train_intensities = []
                train_labels = []
                for train_image_name in self.train_image_names:
                    train_intensities += self.intensities[train_image_name]
                    train_labels += self.labels[train_image_name]
                self.train_data = list(zip(train_intensities, train_labels))

        if stage == 'validate' or stage is None:
            val_intensities = []
            val_labels = []
            for val_image_name in self.val_image_names:
                val_intensities += self.intensities[val_image_name]
                val_labels += self.labels[val_image_name]
            self.val_data = list(zip(val_intensities, val_labels))

        if stage == 'test' or stage is None:
            if len(self.test_image_names) == 1:
                self.test_data = list(zip(self.intensities[self.test_image_name], self.labels[self.test_image_name]))
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

# Instantiate the DataModule
data_module = CustomDataModule(
    root_dir='/Users/agreic/Desktop/Project/Data/Raw',
    crop_dir='/Users/agreic/Desktop/Project/Data/Raw/Training/',
    label_dir='/Users/agreic/Desktop/Project/Data/Raw/Segmentation/curated_masks/neg_subset_and_mega/curated_before_filter/mask_dicts',
    target_size=(34, 164, 174),
    batch_size=2,  # TODO: Change batch size to appropriate value
    train_image_names='Hoxb5',  # Default value
    val_image_names=['c0_0-68_1000', 'c0_0-68_950'],  # Default values
    test_image_names='c0_0-55'  # Default value
)

# Prepare data and loaders
data_module.prepare_data()
data_module.setup(stage='validate')

# Get the train, validation, and test loaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Sanity check, print the first batch of images and labels using the show_slice function
images, labels = next(iter(val_loader))

print(images.shape)