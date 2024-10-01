# TODO: FIX THIS, and add docstrings

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import tifffile as tiff
from torchvision import transforms

from utils.transform import scale, normalize, pad
import lightning as L

#DEBUG
# Debugging high mem usage
from torch.profiler import profile, record_function, ProfilerActivity
import socket
import logging
from datetime import datetime, timedelta

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
                #print(f"Matched label file {label_file} with directory {matched_directory}.")
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

class CustomDataModule(L.LightningDataModule):
    def __init__(self, setup_file=None, root_dir=None, crop_dir=None, label_dir=None, 
                label_to_directory=None, target_size=[34,164,174], batch_size = None,
                num_workers = None, train_image_names='Hoxb5', val_image_names=['c0_0-68_1000', 'c0_0-68_950'],
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
            pad(self.target_size)#,
            # scale(0, 1), # SKIP FOR NOW
            # normalize(0.5, 0.5) # SKIP FOR NOW
        ])

        self.intensities = {}
        self.labels = {}

        self.val_data = []
        self.train_data = []
        self.val_labels = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []

    def prepare_data(self):
        """
        Prepares the data by loading the setup file (if it exists), labels, and matching labels to images.
        """
        # Validate input configuration
        if not (self.setup_file or (self.root_dir and self.crop_dir and self.label_dir and self.label_to_directory)):
            raise ValueError('Either setup_file or root_dir, crop_dir, label_dir, and label_to_directory must be provided')

        # Load setup file if provided
        if self.setup_file:
            print(f'Loading setup information from {self.setup_file}')
            try:
                with open(self.setup_file, 'r') as f:
                    setup_info = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                raise ValueError(f"Error loading setup file: {e}")

            # Validate the presence of necessary keys in setup file
            required_keys = ['root_dir', 'crop_dir', 'label_dir', 'target_size', 'train_image_names', 'val_image_names', 'test_image_names', 'label_to_directory']
            missing_keys = set(required_keys) - setup_info.keys()
            if missing_keys:
                raise KeyError(f"Missing keys in setup file: {missing_keys}")

            # Assign setup parameters with defaults where applicable
            self.root_dir = setup_info['root_dir']
            self.crop_dir = setup_info['crop_dir']
            self.label_dir = setup_info['label_dir']
            self.target_size = setup_info['target_size']
            self.batch_size = setup_info['batch_size'] if self.batch_size is None else self.batch_size #   get('batch_size', self.batch_size)
            self.train_image_names = setup_info['train_image_names']
            self.val_image_names = setup_info['val_image_names']
            self.test_image_names = setup_info['test_image_names']
            self.label_to_directory = setup_info['label_to_directory']
            self.num_workers = setup_info['num_workers'] if self.num_workers is None else self.num_workers #   get('num_workers', self.num_workers)

        # Load labels with error handling
        labels_dict = load_labels(self.label_dir)

        # Match labels to directories
        label_to_directory = match_labels_to_images(
            labels_dict, 
            self.crop_dir, 
            label_to_directory_file=self.label_to_directory
        )

        # Create a filtered dictionary of labels
        new_labels_dict = {
            label_file: {k: v['ctype'] for k, v in content.items()} 
            for label_file, content in labels_dict.items()
        }

        # Load data based on the new labels and matched directories
        self._load_data(new_labels_dict, label_to_directory)

    def _load_data(self, new_labels_dict, label_to_directory):
        """
        Load data from the image directories and labels, and store them in the intensities and labels dictionaries.
        Args:
            new_labels_dict (dict): A dictionary containing label data with necessary information.
            label_to_directory (dict): A mapping of label files to image directories.
        """
        # Iterate over label files and their corresponding directories
        for label_file, image_dir in label_to_directory.items():
            name = label_file.rsplit('.', 1)[0]  # Extract base name without file extension
            self.intensities[name] = []
            self.labels[name] = []

            # Process all .tif images in the directory
            tif_images = [img for img in os.listdir(f'{self.crop_dir}{image_dir}') if img.endswith('.tif')]
            
            idxa = 0
            for image in tif_images:
                idxa += 1
                if idxa % 2 == 0:
                    print(f"Processed {idxa} images for {name}.")
                    break

                mask_id = image.rsplit('_', 1)[-1].split('.')[0]

                # Skip if mask_id is not present in labels or label is unknown
                label = new_labels_dict[label_file].get(mask_id, 'unknown')
                if label == 'unknown':
                    continue

                # Load and process the image
                image_path = f'{self.crop_dir}{image_dir}/{image}'
                loaded_image = torch.from_numpy(tiff.imread(image_path)).float()
                if self.transform:
                   loaded_image = self.transform(loaded_image).unsqueeze(0)  # Apply transform and add dummy channel dimension

                # Store processed image and label (binary: 1 for megakaryocytic, 0 for others)
                self.intensities[name].append(loaded_image)
                self.labels[name].append(1*(label == 'megakaryocytic'))

        # Convert string to list if only one image name is provided
        if isinstance(self.train_image_names, str):
            self.train_image_names = [self.train_image_names]
        
        if isinstance(self.val_image_names, str):
            self.val_image_names = [self.val_image_names]
        
        if isinstance(self.test_image_names, str):
            self.test_image_names = [self.test_image_names]
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:

            for val_name in self.val_image_names:
                if val_name in self.intensities and val_name in self.labels:
                    self.val_data.extend(self.intensities[val_name])
                    self.val_labels.extend(self.labels[val_name])
            print(f"Validation dataset created with {len(self.val_data)} samples.")
            self.val_zip = list(zip(self.val_data, self.val_labels))

            for train_name in self.train_image_names:
                if train_name in self.intensities and train_name in self.labels:
                    self.train_data.extend(self.intensities[train_name])
                    self.train_labels.extend(self.labels[train_name])
            print(f"Training dataset created with {len(self.train_data)} samples.")
            self.train_zip = list(zip(self.train_data, self.train_labels))
        
        elif stage =='validate': # or stage is None: # We already have the validation data using the 'fit' stage if stage is None
            for val_name in self.val_image_names:
                if val_name in self.intensities and val_name in self.labels:
                    self.val_data.extend(self.intensities[val_name])
                    self.val_labels.extend(self.labels[val_name])
            print(f"Validation dataset created with {len(self.val_data)} samples.")
            self.val_zip = list(zip(self.val_data, self.val_labels))

        if stage == 'test' or stage is None:
            for test_name in self.test_image_names:
                if test_name in self.intensities and test_name in self.labels:
                    self.test_data.extend(self.intensities[test_name])
                    self.test_labels.extend(self.labels[test_name])
            print(f"Test dataset created with {len(self.test_data)} samples.")
            self.test_zip = list(zip(self.test_data, self.test_labels))

        if stage == 'predict' or stage is None:
            ...
        
        # TODO: Check stage argument whether in predict i can pass it to get diff dataloaders, as I think I removed the functionality rn

    def train_dataloader(self):
        #dataset = CustomDataset(self.train_data, self.train_labels)
        return DataLoader(self.train_zip, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        #dataset = CustomDataset(self.val_data, self.val_labels)
        return DataLoader(self.val_zip, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        #dataset = CustomDataset(self.test_data, self.test_labels)
        return DataLoader(self.test_zip, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        ...

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)
    
    def train_dataset(self):
        return self.train_zip#self.train_data, self.train_labels
    
    def val_dataset(self):
        return self.val_zip#self.val_data, self.val_labels
    
    def test_dataset(self):
        return self.test_zip#self.test_data, self.test_labels
    
    def predict_dataset(self):
        ...
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data[0]  # Assuming this is a list or tensor of images
        self.labels = data[1]  # Assuming this is a list or tensor of labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx] # Get the image tensor
        label = self.labels[idx] # Get the label tensor
        return image, label  # Return the image and label

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

    data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json', batch_size=2, num_workers=0)
    
    data_module.prepare_data()
    data_module.setup()

    # # Prepare data and loaders with enhanced memory profiling
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              profile_memory=True, 
    #              record_shapes=True, 
    #              with_stack=True) as prof:
    #     with record_function("datamodule_prep"):
    #         data_module.prepare_data()
    #         data_module.setup()
    
    # # Print more detailed memory profiling information
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))

    # Get the train, validation, and test loaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # Sanity check, print the first batch of images and labels using the show_slice function
    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)

if __name__ == '__main__':
    main()

#DEBUG:
'''
With no transforms:
STAGE:2024-10-01 14:44:30 7306:195153 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-10-01 14:44:30 7306:195153 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
    aten::empty_strided         0.15%       2.352ms         0.15%       2.352ms       0.523us     191.92 Mb     191.92 Mb          4500  
            aten::copy_         2.73%      42.728ms         2.73%      42.728ms       9.495us     163.59 Mb     151.31 Mb          4500  
               aten::to         0.42%       6.497ms         3.49%      54.617ms      12.137us     308.05 Mb      38.60 Mb          4500  
       aten::lift_fresh         0.00%       4.000us         0.00%       4.000us       0.001us           0 b           0 b          4500  
        aten::unsqueeze         0.34%       5.267ms         0.35%       5.440ms       1.209us           0 b           0 b          4500  
       aten::as_strided         0.01%     173.000us         0.01%     173.000us       0.038us           0 b           0 b          4500  
         aten::_to_copy         0.72%      11.287ms         3.38%      52.853ms      11.745us     308.05 Mb     -30.90 Mb          4500  
        datamodule_prep        95.64%        1.497s       100.00%        1.565s        1.565s     308.05 Mb     -42.90 Mb             1  
-----------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.565s

Only padding:
STAGE:2024-10-01 14:45:53 7432:196733 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-10-01 14:45:53 7432:196733 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              aten::empty         0.38%      16.205ms         0.38%      16.205ms       3.601us      11.74 Gb      11.74 Gb          4500  
              aten::fill_        40.52%        1.733s        40.52%        1.733s     385.121us       7.23 Gb       7.19 Gb          4500  
                aten::pad         5.04%     215.465ms        44.66%        1.910s     424.393us      16.26 Gb       1.83 Gb          4500  
      aten::empty_strided         0.09%       3.918ms         0.09%       3.918ms       0.871us     218.78 Mb     218.78 Mb          4500  
              aten::copy_         1.89%      80.793ms         1.89%      80.793ms       8.977us     130.48 Mb     130.02 Mb          9000  
                 aten::to         0.17%       7.476ms         1.20%      51.378ms      11.417us     308.05 Mb      39.76 Mb          4500  
         aten::lift_fresh         0.00%      59.000us         0.00%      59.000us       0.013us           0 b           0 b          4500  
             aten::narrow         0.70%      30.093ms         1.35%      57.922ms       2.146us           0 b           0 b         26994  
              aten::slice         0.60%      25.638ms         0.67%      28.858ms       1.069us           0 b           0 b         26994  
         aten::as_strided         0.09%       3.777ms         0.09%       3.777ms       0.120us           0 b           0 b         31494  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 4.277s

Improved code (Padding):
STAGE:2024-10-01 16:38:42 15229:312555 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-10-01 16:38:42 15229:312555 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
              aten::empty         0.16%       8.374ms         0.16%       8.374ms       1.861us       2.89 Gb       2.89 Gb          4500  
              aten::fill_        14.59%     768.199ms        14.59%     768.199ms     170.711us       1.48 Gb       1.48 Gb          4500  
                aten::pad         1.21%      63.573ms        17.71%     932.281ms     207.174us       4.07 Gb     376.59 Mb          4500  
         aten::lift_fresh         0.00%     165.000us         0.00%     165.000us       0.037us           0 b           0 b          4500  
             aten::narrow         0.68%      35.797ms         1.27%      66.866ms       2.477us           0 b           0 b         26994  
              aten::slice         0.54%      28.529ms         0.61%      32.137ms       1.191us           0 b           0 b         26994  
         aten::as_strided         0.08%       4.259ms         0.08%       4.259ms       0.135us           0 b           0 b         31494  
              aten::copy_         0.51%      26.760ms         0.51%      26.760ms       5.947us           0 b           0 b          4500  
          aten::unsqueeze         0.24%      12.711ms         0.25%      13.062ms       2.903us           0 b           0 b          4500  
    aten::constant_pad_nd         1.07%      56.513ms        17.62%     927.344ms     206.076us       4.07 Gb    -310.89 Mb          4500  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 5.263s

Improved code (Padding + Scale + Normalize):
STAGE:2024-10-01 16:41:43 15493:316082 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-10-01 16:41:43 15493:316082 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::sub         6.34%     926.439ms         6.59%     962.202ms      71.274us      20.33 Gb      19.05 Gb         13500  
                aten::add         8.48%        1.239s         8.66%        1.265s     281.069us      16.25 Gb      15.68 Gb          4500  
                aten::div        12.65%        1.848s        18.76%        2.740s     304.425us      33.74 Gb      14.47 Gb          9000  
                aten::mul         4.73%     690.585ms         4.93%     719.972ms     159.994us      16.26 Gb      14.31 Gb          4500  
      aten::empty_strided         0.27%      39.613ms         0.27%      39.613ms       1.467us      12.29 Gb      12.29 Gb         27000  
              aten::copy_         5.30%     774.337ms         5.30%     774.337ms      24.582us       5.84 Gb       5.72 Gb         31500  
                 aten::to         0.54%      79.495ms         6.11%     891.994ms      33.037us      19.41 Gb       4.49 Gb         27000  
              aten::empty         0.17%      24.886ms         0.17%      24.886ms       1.383us       3.08 Gb       3.08 Gb         18000  
    aten::constant_pad_nd         0.36%      52.752ms         2.52%     368.755ms      81.946us       4.07 Gb     741.15 Mb          4500  
           aten::_to_copy         0.82%     119.361ms         5.92%     864.459ms      32.017us      18.05 Gb     584.80 Mb         27000  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 14.606s

'''