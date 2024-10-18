# This script visualizes a histogram of the sizes of the masks, and allows the user to set a
# threshold for the size of the masks to be considered as megakaryocytes.
# The script then filters the masks based on the threshold, and saves
# the megakaryocyte masks and negative examples as separate dictionaries,
# as well as a combined dictionary of the two after randomly sampling a subset of the negative examples.

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
    for key in tqdm(coordinates.keys(), desc=f'\n### Reconstructing mask.tif file with filtered mask-coordinates.pkl file.'):
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

    # VARIABLES:

    parser.add_argument(
        '--multiplier',
        type = int,
        required=True,
        help = 'The number of negative masks per megakaryocyte to sample. (eg. a value of 20 would result in 20 negative masks for each megakaryocyte sampled. Note: megakaryocytes are chosen based off of size, thus this multiplier is not fully accurate).'
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
    megakaryocyte_masks_path = args.mega_path
    negative_masks_path = args.negative_all_path
    negative_masks_subset_path = args.negative_subset_path
    combined_masks_path = args.neg_subset_and_mega_path
    multiplier = args.multiplier

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

    # Initialize a dictionary to store the mask sizes
    mask_sizes = {}

    # initialize dictionary to store the max area of a mask on any z-layer
    mask_max_area = {}


    for key in tqdm(coordinates.keys(), desc='\n### Calculating mask sizes.'):
        # Essentially, each coordinate refers to one single voxel in the mask.
        size = len(coordinates[key][2])
        # Add the size to the mask sizes dictionary with key as the mask id.
        mask_sizes[key] = size

        # z is channel 0, y is channel 1, x is channel 2

        # Additionally check the max area for a mask on any of the z-layers
        z_coords = coordinates[key][0] # Get the z-coordinates of the mask
        z_coords = np.array(z_coords) # Convert to numpy array
        unique_z_layers = np.unique(z_coords) # Get the unique z-layers
        
        counts = np.zeros(len(unique_z_layers)) # Initialize an array to store the counts of the mask on each z-layer
        for i, z in enumerate(unique_z_layers):
            counts[i] = (z_coords == z).sum()

        max_area = counts.max()
        # Save the max area for this mask
        mask_max_area[key] = max_area


    # Plot the histogram of the mask max areas
    plt.figure(figsize=(12, 8))
    plt.hist(mask_max_area.values(), bins=100, color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title('Histogram of mask max areas (voxel count of a single z-layer)')
    plt.xlabel('Max area of mask on any z-layer')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    # Set a threshold for the size of the masks to be considered megakaryocytes:
    # The threshold will be visualized on the histogram.
    # Plot the histogram of the mask sizes
    plt.figure(figsize=(12, 8))
    plt.hist(mask_sizes.values(), bins=100, color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title('Histogram of mask sizes (voxel count)')
    plt.xlabel('Size of mask (voxel count)')
    plt.ylabel('Frequency (log scale)')
    plt.show()

    while True:
        while True:
            threshold = input('Enter the threshold for the size of the masks to be considered as megakaryocytes: ')
            if not threshold.isdigit():
                print('Please enter a valid number.')

            else:
                threshold = int(threshold)
                break
        
        # Display the histogram with the threshold
        plt.close()
        histogram = plt.hist(mask_sizes.values(), bins=100, color='blue', alpha=0.7)
        plt.yscale('log')
        plt.title('Histogram of mask sizes')
        plt.xlabel('Size of mask')
        plt.ylabel('Frequency (log scale)')
        plt.axvline(x=threshold, color='red', linestyle='--')
        plt.show()
        
        confirm = input(f'You have entered {threshold}. Are you satisfied with this threshold? (yes/no): ')
        if confirm.lower() == 'yes':
            break
        else: # Reprompt for another value
            print('Please enter another threshold value.')


    # Filter the masks based on the threshold
    megakaryocyte_masks = {k: v for k, v in mask_sizes.items() if v > threshold}
    print(f'\n### Number of megakaryocyte masks: {len(megakaryocyte_masks)}')

    # The remaining masks are considered as negative examples:
    negative_masks = {k: v for k, v in mask_sizes.items() if v <= threshold}
    print(f'\n### Number of negative masks: {len(negative_masks)}')

    # Convert to original dictionary items
    for positive_mask in megakaryocyte_masks.keys():
        megakaryocyte_masks[positive_mask] = coordinates[positive_mask]

    for negative_mask in negative_masks.keys():
        negative_masks[negative_mask] = coordinates[negative_mask]

    # Randomly select a subset of the negative masks (*multiplier) the number of megakaryocyte masks:
    n_count = int(len(megakaryocyte_masks))*multiplier

    # Randomly sample the negative masks
        # Convert dictionary items to a list of tuples
    negative_masks_list = list(negative_masks.items())

        # Randomly sample n items
    sampled_items = random.sample(negative_masks_list, n_count)

        # Convert the list of tuples back to a dictionary
    negative_masks_subset = dict(sampled_items)

    print(f'\n### Number of negative masks subset: {len(negative_masks_subset)}')

############ SAVE THE MASKS ############

    # Save all the masks, including the megakaryocyte masks and the negative masks subset:
    file_name = os.path.basename(mask_array_filtered_path)
    file_name = file_name.split('.')[0]

    # Save the megakaryocyte masks
    megakaryocyte_masks_path_pkl = os.path.join(path_out, f'{file_name}_megas.pkl')
    megakaryocyte_masks_path_tif = os.path.join(path_out, f'{file_name}_megas.tif')

    with open(megakaryocyte_masks_path_pkl, 'wb') as f:
        pickle.dump(megakaryocyte_masks, f)

    megakaryocyte_masks_tif = reconstruct_arr_from_dict(mask_shape, megakaryocyte_masks)
    tifffile.imwrite(megakaryocyte_masks_path_tif, megakaryocyte_masks_tif.astype(np.int32), compression=('zlib', 1))

   # Save the negative subset and full masks
    negative_masks_subset_path_pkl = os.path.join(path_out, f'{file_name}_neg_sub.pkl')
    negative_masks_subset_path_tif = os.path.join(path_out, f'{file_name}_neg_sub.tif')

    with open(negative_masks_subset_path_pkl, 'wb') as f:
        pickle.dump(negative_masks_subset, f)

    negative_masks_subset_tif = reconstruct_arr_from_dict(mask_shape, negative_masks_subset)
    tifffile.imwrite(negative_masks_subset_path_tif, negative_masks_subset_tif.astype(np.int32), compression=('zlib', 1))

    negative_masks_path_pkl = os.path.join(path_out, f'{file_name}_neg.pkl')
    negative_masks_path_tif = os.path.join(path_out, f'{file_name}_neg.tif')

    with open(negative_masks_path_pkl, 'wb') as f:
        pickle.dump(negative_masks, f)

    negative_masks_tif = reconstruct_arr_from_dict(mask_shape, negative_masks)
    tifffile.imwrite(negative_masks_path_tif, negative_masks_tif.astype(np.int32), compression=('zlib', 1))

    # Save the combined dictionary of megakaryocyte masks and negative masks subset
    negative_masks_subset.update(megakaryocyte_masks)
    combined_masks_path_pkl = os.path.join(path_out, f'{file_name}_comb.pkl')
    combined_masks_path_tif = os.path.join(path_out, f'{file_name}_comb.tif')

    with open(combined_masks_path_pkl, 'wb') as f:
        pickle.dump(negative_masks_subset, f)

    combined_masks_tif = reconstruct_arr_from_dict(mask_shape, negative_masks_subset)
    tifffile.imwrite(combined_masks_path_tif, combined_masks_tif.astype(np.int32), compression=('zlib', 1))


if __name__ == '__main__':
    main()