# This script generates cutouts from 3D image data based on provided mask coordinates. 
# It takes mask coordinate data from a pickled file and applies it to a TIFF image (nucleus channel), 
# extracting the relevant 3D regions as mask cutouts and saving them as TIFF files. 
# Additionally, it calculates and saves metadata about the cutout sizes and properties into a JSON file.

# Argument parser setup
# Defines command-line arguments for:
# - `mask_coordinates_path`: Path to the pickled mask coordinate dictionary.
# - `image_path`: Path to the TIFF image that contains the 3D data (nucleus channel).
# - `out_path`: Output directory where cutout files and metadata will be saved.


import os
import numpy as np
import pickle
from tqdm import tqdm
import argparse
import tifffile
import json

# Argument parser setup
parser = argparse.ArgumentParser(description='Generate cutouts from mask coordinates.')
parser.add_argument(
    '--mask_coordinates_path',
    type=str,
    required=True,
    help='Filepath where the pickled mask coordinate dictionary is saved. (Including filename and extension).'
)
parser.add_argument(
    '--image_path',
    type=str,
    required=True,
    help='Filepath where the tif image (nucleus channel) is saved. (Including filename and extension).'
)
parser.add_argument(
    '--out_path',
    type=str,
    required=True,
    help='Folder where to save the cutouts (.tif) to.'
)

args = parser.parse_args()

# Paths and filenames
mask_coordinates_path = args.mask_coordinates_path
mask_file = args.image_path
out_path = os.path.join(args.out_path, os.path.basename(mask_coordinates_path).split('.')[0] + '_cutouts/')
filename = os.path.basename(mask_coordinates_path).split('.')[0]

# Ensure output directory exists
os.makedirs(out_path, exist_ok=True)

# Load the mask coordinate dictionary
with open(mask_coordinates_path, 'rb') as f:
    mask_dict = pickle.load(f)
print(f"Mask dictionary loaded from {mask_coordinates_path}")
print(f'\n### Successfully loaded pickled mask coordinate file containing {len(mask_dict)} masks.')

# Remove zero-key if it exists, potentially background mask
if 0 in mask_dict:
    mask_dict.pop(0)
    print(f'\n### Successfully removed the zero-key in the dictionary.')

# Load the mask array
mask_array = tifffile.imread(mask_file)
print(f'\n### Mask of shape {mask_array.shape} loaded.')

# Variables to store various parameters for max size cutout
max_z_size = 0 # max(z_max - z_min) globally, NOT PADDED
max_y_size = 0 # max(y_max - y_min) globally, NOT PADDED
max_x_size = 0 # max(x_max - x_min) globally, NOT PADDED

# Variables to store various parameters for max size cutout + padding
max_z_size_padded = 0 # max(z_max - z_min) globally, PADDED
max_y_size_padded = 0 # max(y_max - y_min) globally, PADDED
max_x_size_padded = 0 # max(x_max - x_min) globally, PADDED

# Biggest mask variables
biggest_mask_pixels = 0
biggest_mask = None
bm_z_size = 0 # z_max - z_min for biggest mask
bm_y_size = 0 # y_max - y_min for biggest mask
bm_x_size = 0 # x_max - x_min for biggest mask

# Biggest mask + padding variables
biggest_mask_pixels_padded = 0
biggest_mask_padded = None
bmp_z_size = 0 # z_max - z_min for biggest (mask + padding), PADDING FIRST
bmp_y_size = 0 # y_max - y_min for biggest (mask + padding), PADDING FIRST
bmp_x_size = 0 # x_max - x_min for biggest (mask + padding), PADDING FIRST

# Generate crops
for key in tqdm(mask_dict.keys(), desc='Creating mask crops'):
    z_indices, y_indices, x_indices = mask_dict[key]

    mask_z_min = min(z_indices)
    mask_z_max = max(z_indices)

    mask_z_size = mask_z_max - mask_z_min

    if mask_z_size > max_z_size:
        max_z_size = mask_z_size
    
    mask_z_min_padded = max(0, mask_z_min - 1) # Z cannot be lower than 0.
    mask_z_max_padded = min(mask_array.shape[0], mask_z_max + 1) # Z cannot be higher than max Z.

    if (mask_z_max_padded - mask_z_min_padded) > max_z_size_padded:
        max_z_size_padded = mask_z_max_padded - mask_z_min_padded

    mask_y_min = min(y_indices)
    mask_y_max = max(y_indices)

    mask_y_size = mask_y_max - mask_y_min

    if mask_y_size > max_y_size:
        max_y_size = mask_y_size

    mask_y_min_padded = max(0, mask_y_min - 2) # Y cannot be lower than 0.
    mask_y_max_padded = min(mask_array.shape[1], mask_y_max + 2) # Y cannot be higher than max Y.
    
    if (mask_y_max_padded - mask_y_min_padded) > max_y_size_padded:
        max_y_size_padded = mask_y_max_padded - mask_y_min_padded

    mask_x_min = min(x_indices)
    mask_x_max = max(x_indices)

    mask_x_size = mask_x_max - mask_x_min

    if mask_x_size > max_x_size:
        max_x_size = mask_x_size
    
    mask_x_min_padded = max(0, mask_x_min - 2) # X cannot be lower than 0.
    mask_x_max_padded = min(mask_array.shape[2], mask_x_max + 2) # X cannot be higher than max X.

    if (mask_x_max_padded - mask_x_min_padded) > max_x_size_padded:
        max_x_size_padded = mask_x_max_padded - mask_x_min_padded

    # Calculate sizes for biggest mask
    pxls = len(z_indices)
    if pxls > biggest_mask_pixels:
        biggest_mask_pixels = pxls
        biggest_mask = key
        bm_z_size = mask_z_size
        bm_y_size = mask_y_size
        bm_x_size = mask_x_size
    
    # Calculate sizes for biggest mask + padding
    pxls_padded = (mask_z_max_padded - mask_z_min_padded) * (mask_y_max_padded - mask_y_min_padded) * (mask_x_max_padded - mask_x_min_padded)
    if pxls_padded > biggest_mask_pixels_padded:
        biggest_mask_pixels_padded = pxls_padded
        biggest_mask_padded = key
        bmp_z_size = mask_z_max_padded - mask_z_min_padded
        bmp_y_size = mask_y_max_padded - mask_y_min_padded
        bmp_x_size = mask_x_max_padded - mask_x_min_padded

    mask_cutout = mask_array[mask_z_min_padded:mask_z_max_padded, mask_y_min_padded:mask_y_max_padded, mask_x_min_padded:mask_x_max_padded]
    tifffile.imsave(os.path.join(out_path, f'{filename}_mask_{key}.tif'), mask_cutout)

# Save the information of the sizes to json 

info = {
    'max_z_size': int(max_z_size),                 # Maximum depth of a cutout without padding
    'max_y_size': int(max_y_size),                 # Maximum height of a cutout without padding
    'max_x_size': int(max_x_size),                 # Maximum width of a cutout without padding
    'max_z_size_padded': int(max_z_size_padded),   # Maximum depth of a cutout with padding
    'max_y_size_padded': int(max_y_size_padded),   # Maximum height of a cutout with padding
    'max_x_size_padded': int(max_x_size_padded),   # Maximum width of a cutout with padding
    'biggest_mask_pixels': int(biggest_mask_pixels),   # Number of pixels in the largest mask
    'biggest_mask': int(biggest_mask),             # ID of the largest mask
    'bm_z_size': int(bm_z_size),                   # Depth of the largest mask
    'bm_y_size': int(bm_y_size),                   # Height of the largest mask
    'bm_x_size': int(bm_x_size),                   # Width of the largest mask
    'biggest_mask_pixels_padded': int(biggest_mask_pixels_padded),   # Number of pixels in the largest mask with padding
    'biggest_mask_padded': int(biggest_mask_padded),   # ID of the largest mask with padding
    'bmp_z_size': int(bmp_z_size),                 # Depth of the largest mask with padding
    'bmp_y_size': int(bmp_y_size),                 # Height of the largest mask with padding
    'bmp_x_size': int(bmp_x_size)                  # Width of the largest mask with padding
}

with open(os.path.join(out_path, f'{filename}_info.json'), 'w') as f:
    json.dump(info, f, indent=4)