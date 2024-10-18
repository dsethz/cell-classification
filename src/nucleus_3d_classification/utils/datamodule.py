# TODO: FIX THIS, and add docstrings

import os
import json
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import tifffile as tiff
from torchvision import transforms

from utils.transform import scale, normalize, pad, rotate_invert
import lightning as L

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, label_file, crop_dir, target_size=[34,164,174]):
        
        self.label_file = label_file
        self.crop_dir = crop_dir
        self.target_size = target_size

        self.transform = transforms.Compose([
            rotate_invert(rotate_prob = 0, invert_z_prob = .5, invert_x_prob = .5, invert_y_prob = .5),
            pad(self.target_size),
            scale(0, 1),
            normalize(0.5, 0.5)
        ])
        
        self.files = [file for file in os.listdir(self.crop_dir) if file.endswith('.tif')]
        
        # Process labels
        self.labels = process_labels(self.label_file)

        # Create a list of dictionaries to hold file, mask_id, and label together
        self.data = []
        idx = 0
        for file in self.files:
            mask_id = file.rsplit('_', 1)[-1].split('.')[0]
            label = self.labels.get(mask_id, None)  # Fetch the label based on the mask_id
            if label is None or label == 'unknown':
                print(f"Warning: No label found for mask_id {mask_id} in {self.label_file}")
                # Skip appending
                continue
            self.data.append({
                'idx': idx,
                'file': file,
                'mask_id': mask_id,
                'label': label
            })
            idx += 1
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the file, mask_id, and label for the given index
        img_path = self.data[idx]['file']
        label = self.data[idx]['label']

        if label is None:
            raise ValueError(f"Error: No label found for mask_id {self.data[idx]['mask_id']} in {self.label_file}")
        if not os.path.exists(os.path.join(self.crop_dir, img_path)):
            raise FileNotFoundError(f"Error: Image file {img_path} not found in {self.crop_dir}")
        if not label in [0, 1]:
            raise ValueError(f"Error: Invalid label {label} found for mask_id {self.data[idx]['mask_id']} in {self.label_file}")
        if not img_path.endswith('.tif'):
            raise ValueError(f"Error: Invalid image file {img_path}. Only .tif files are supported.")

        # Load the image and apply transformations
        img = tiff.imread(os.path.join(self.crop_dir, img_path))

        if img is None:
            raise ValueError(f"Error: Failed to load image {img_path}.")
        

        img = torch.from_numpy(img).float()
        if self.transform:
            img = self.transform(img)
        return img.unsqueeze(0), label
    

def process_labels(label_file):
    try:
        with open(label_file, 'r') as f:
            labels = {k: 1*(v['ctype'] == 'megakaryocytic') for k, v in json.load(f).items() if v['ctype'] != 'unknown'}
            return labels
    except FileNotFoundError:
        print(f"Error: Label file {label_file} not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse {label_file}. Invalid JSON format. {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading {label_file}. {e}")
    print(f"Error: Failed to load labels from {label_file}.")

def convert_to_list(data):
    if isinstance(data, str):
        return [data]
    return data

class CustomDataModule(L.LightningDataModule):
    def __init__(self,
                setup_file=None,
                target_size=[34,164,174], # This was CD41
                # target_size=[35,331,216], # This is for all other markers rn
                batch_size:int = None,
                num_workers:int = None,
                stage: str = None):
        super().__init__()

        # Passed
        self.stage = stage
        self.setup_file = setup_file
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Created
        self.test_data = None
        self.training_data = None
        self.validation_data = None
        
        self.test_data_names = None
        self.training_data_names = None
        self.validation_data_names = None
        
        with open(self.setup_file, 'r') as f:
            setup_info = json.load(f)
            self.batch_size = setup_info['batch_size'] if self.batch_size is None else self.batch_size
            self.num_workers = setup_info['num_workers'] if self.num_workers is None else self.num_workers
            self.training_data_names = convert_to_list(setup_info.get('training_data', None))
            self.validation_data_names = convert_to_list(setup_info.get('validation_data', None))
            self.test_data_names = convert_to_list(setup_info.get('test_data', None))
    
    def setup(self, stage=None):
        # Handle if multiple files are being passed for training, validation, and testing
        if stage == 'fit' or stage is None:
            self.training_data = create_combined_dataset(self.training_data_names, self.target_size)
            self.validation_data = create_combined_dataset(self.validation_data_names, self.target_size)
        if stage == 'validate':
            self.validation_data = create_combined_dataset(self.validation_data_names, self.target_size)
        if stage == 'test' or stage is None:
            self.test_data = create_combined_dataset(self.test_data_names, self.target_size)
        if stage == 'predict':
            # TODO: Change to predict data later on.
            self.test_data = create_combined_dataset(self.test_data_names, self.target_size)
    
    def train_dataloader(self):
        return DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.validation_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self):
        # TODO: Change to predict data later on.
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def teardown(self, stage: str) -> None:
        return super().teardown(stage)

    def train_dataset(self):
        return self.training_data
    
    def val_dataset(self):
        return self.validation_data
    
    def test_dataset(self):
        return self.test_data    
    
    def predict_dataset(self):
        # TODO: Change to predict data later on.
        return self.test_data
    
def create_combined_dataset(data_list, target_size):
    """Helper function to create and combine datasets."""
    dataset_list = []
    
    # Check if data_list is a single file path or a list of file paths
    data_list = convert_to_list(data_list)

    # Step 1: Create individual CustomDataset objects and add to dataset_list
    for idx, item in enumerate(data_list):
        print(f"Creating dataset {idx + 1} for {item['label_file']} and {item['crop_dir']}")
        
        try:
            dataset = CustomDataset(
                label_file=item['label_file'],
                crop_dir=item['crop_dir'],
                target_size=target_size
            )
            dataset_list.append(dataset)
            print(f"Dataset {idx + 1} created successfully with length: {len(dataset)}")
        except Exception as e:
            print(f"Error creating dataset {idx + 1}: {e}")

    # Step 2: Check if datasets are successfully created
    if len(dataset_list) == 0:
        raise ValueError("No datasets were created. Please check your dataset paths or initialization.")
    
    # Step 3: Concatenate datasets if more than one, otherwise return the single dataset
    if len(dataset_list) > 1:
        combined_dataset = torch.utils.data.ConcatDataset(dataset_list)
        print(f"Combined dataset created with total length: {len(combined_dataset)}")
        return combined_dataset
    else:
        combined_dataset = dataset_list[0]
        print(f"Only one dataset provided, no concatenation required.")
    
    return combined_dataset

def main():
    #data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json', batch_size=2, num_workers=4)
    return
    # data_module.prepare_data()
    # data_module.setup()

    # # Prepare data and loaders with enhanced memory profiling
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
    #              profile_memory=True, 
    #              record_shapes=True, 
    #              with_stack=True) as prof:
    #     with record_function("datamodule_prep"):
    #         data_module.prepare_data()
    #         data_module.setup()
    
    # Print more detailed memory profiling information
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=10))

    # # Get the train, validation, and test loaders
    # train_loader = data_module.train_dataloader()
    # val_loader = data_module.val_dataloader()
    # test_loader = data_module.test_dataloader()

    # # Sanity check, print the first batch of images and labels using the show_slice function
    # images, labels = next(iter(train_loader))

    # print(images.shape)
    # print(labels.shape)

if __name__ == '__main__':
    main()
