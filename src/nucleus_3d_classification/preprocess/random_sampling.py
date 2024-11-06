'''
This script performs random sampling of masks from a given mask coordinate dictionary and saves the sampled masks.
Functions:
    reconstruct_arr_from_dict(arr_shape, coordinates):
        Reconstructs the mask array from the dictionary of mask coordinates.
    main():
        Main function to parse arguments, load data, perform random sampling, and save the sampled masks.
Arguments:
    --mask_coordinates_path (str): Filepath where the pickled mask coordinate dictionary is saved (after filtering).
    --mask_array_path (str): Filepath where the mask array (after manual curation) is saved.
    --n_count (int): Number of masks to sample.
    --out_path (str): Path to output directory.
Usage:
    python random_sampling.py --mask_coordinates_path <path_to_mask_coordinates> --mask_array_path <path_to_mask_array> --n_count <number_of_masks> --out_path <output_directory>
'''

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import tifffile
from tqdm import tqdm
import random

def reconstruct_arr_from_dict(arr_shape, coordinates):
# This function is taken from: 'https://github.com/CSDGroup/3d_segmentation/blob/main/scripts/filter_through_masks.py'.
# It reconstructs the mask array from the dictionary of mask coordinates.
    filtered_mask_array = np.zeros(arr_shape)
    for key in tqdm(coordinates.keys(), desc=f'\n### Reconstructing mask.tif file from mask-coordinates.pkl file.'):
        for i in range(0, coordinates[key][0].shape[0]):
            filtered_mask_array[coordinates[key][0][i], coordinates[key][1][i], coordinates[key][2][i]] = key
    return filtered_mask_array

def main():

    parser = argparse.ArgumentParser()

    # INPUTS:

    parser.add_argument(
        '--mask_coordinates_path',
        type = str,
        required=True,
        help = 'Filepath where the pickled mask coordinate dictionary is saved (after filtering). (Including filename and extension).'
    )

    parser.add_argument(
        '--mask_array_path',
        type = str,
        required=True,
        help = 'Filepath where the mask array (after manual curation) is saved. (Including filename and extension).'
    )

    parser.add_argument(
        '--n_count',
        type = int,
        required=True,
        help = 'Number of masks to sample.'
    )

    # OUTPUTS:

    parser.add_argument(
        '--out_path',
        type = str,
        required=True,
        help = 'Path to output directory.'
    )

    args =  parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    path_out = args.out_path
    mask_coordinates_path = args.mask_coordinates_path
    mask_array_filtered_path = args.mask_array_path
    n_count = args.n_count

    # Load the filtered mask array
    mask_array = tifffile.imread(mask_array_filtered_path)
    mask_shape = mask_array.shape
    print(f'\n### Mask of shape {mask_array.shape} loaded.')

    # Loads the mask coordinate dict
    with open(mask_coordinates_path, 'rb') as f:
        coordinates = pickle.load(f)
    print(f'\n### Successfully loaded pickled mask coordinate file containing {int(len(coordinates))} masks.')

    #remove the zero-key in the dictionary
    if (0 in coordinates.keys()) == True:
        coordinates.pop(0)
    
    masks_list = list(coordinates.keys())

    # Randomly sample n items
    sampled_items = random.sample(masks_list, n_count)

    #print(f'\n### Randomly sampled {len(sampled_items)} masks.')

    # Keep only sampled items
    masks_subset = {key: coordinates[key] for key in sampled_items}

    print(f'\n### Number of masks sampled {len(masks_subset)}, this should be equal to {n_count}.')

############ SAVE THE MASKS ############

    # Save all the masks, including the megakaryocyte masks and the negative masks subset:
    file_name = os.path.basename(mask_array_filtered_path)
    file_name = file_name.split('.')[0]

    # Save the megakaryocyte masks
    out_masks_path_pkl = os.path.join(path_out, f'{file_name}_sampled.pkl')
    out_masks_path_tif = os.path.join(path_out, f'{file_name}_sampled.tif')

    with open(out_masks_path_pkl, 'wb') as f:
        pickle.dump(masks_subset, f)

    out_masks_tif = reconstruct_arr_from_dict(mask_shape, masks_subset)
    tifffile.imwrite(out_masks_path_tif, out_masks_tif.astype(np.int32), compression=('zlib', 1))

if __name__ == '__main__':
    main()