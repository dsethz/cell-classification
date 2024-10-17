"""
This script generates 2D and 3D features from images, their masks (.tif), and mask coordinate dictionaries (.pkl).
The mask coordinate dictionaries are generated using the script 'generate_mask_coord_dict.py', which is available at:
https://github.com/CSDGroup/3d_segmentation/blob/main/scripts/generate_mask_coord_dict.py.
Different files are matched to each other using their filenames and sorting them using the .sort() method.
Usage:
    python extract_2D_3D_features.py --image_directory <image_directory> --mask_directory <mask_directory> --filetype <filetype> --out_dir <out_dir> --coord_directory <coord_directory> --spacing <spacing> --same_columns <same_columns>
Arguments:
    --image_directory: Directory containing the images.
    --mask_directory: Directory containing the masks.
    --filetype: File type of the images and masks. Only .tif files are supported at the moment.
    --out_dir: Output directory for the features.
    --coord_directory: Directory containing the coordinate files.
    --spacing: Voxel spacing for the images.
    --same_columns: Whether to only keep the same columns in 2D and 3D features.
The script performs the following steps:
1. Parses the input arguments.
2. Checks if the specified file type is supported.
3. Creates the output directory if it doesn't exist.
4. Detects all filenames in the user-specified input directories and sorts them.
5. Loads pairs of image files and corresponding mask files.
6. Extracts 2D features from each z-layer of the image and mask.
7. Groups the 2D features by label and computes the mean across z layers.
8. Extracts 3D features from the entire image and mask.
9. Ensures common columns between 2D and 3D features if specified.
10. Aligns the mask ids in both 2D and 3D features.
11. Saves the features to disk.
"""

# Description: Generate 2D and 3D features from images, their masks (.tif), labels (.json) and mask coord dictionaries (.pkl).
# The mask coord dictionaries are generated using the script 'generate_mask_coord_dict.py', which is available at: https://github.com/CSDGroup/3d_segmentation/blob/main/scripts/generate_mask_coord_dict.py
# Different files are matched to each other using their filenames and sorting them using .sort() method.

# Import necessary libraries
import numpy as np
import glob
import os
from skimage import measure
import pickle
import pandas as pd
import json
from skimage.measure import regionprops_table
import imageio.v2 as imageio
import tifffile
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate 2D and 3D features from images, their masks (.tif), labels (.json) and mask coord dictionaries (.pkl).')
parser.add_argument('--image_directory', type=str, default='/Users/agreic/Desktop/Project/Data/Raw/Images/nucleus', help='Directory containing the images.')
parser.add_argument('--mask_directory', type=str, default='/Users/agreic/Desktop/testing_dir/mask_dir/', help='Directory containing the masks.')
parser.add_argument('--filetype', type=str, default='.tif', help='File type of the images and masks.')
parser.add_argument('--out_dir', type=str, default='/Users/agreic/Desktop/testing_dir/out_dir/', help='Output directory for the features.')
# parser.add_argument('--label_directory', type=str, default='/Users/agreic/Desktop/testing_dir/label_dir/', help='Directory containing the label files.')
parser.add_argument('--coord_directory', type=str, default='/Users/agreic/Desktop/testing_dir/coord_dir/', help='Directory containing the coordinate files.')
parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 0.24, 0.24), help='Voxel spacing for the images.')
parser.add_argument('--same_columns', type=bool, default=False, help='Whether to only keep the same columns in 2D and 3D features.')

args = parser.parse_args()

# Assign arguments to variables
image_directory = args.image_directory
mask_directory = args.mask_directory
filetype = args.filetype
out_dir = args.out_dir
# label_directory = args.label_directory
coord_directory = args.coord_directory
same_columns_flag = args.same_columns
spacing = tuple(args.spacing)

# Check for filetpye
if filetype != '.tif':
    raise ValueError('Only .tif files are supported at the moment.')


# Create output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Properties to calculate

properties = [
    "area",  # Area of the region i.e. number of pixels of the region scaled by pixel-area.
    "area_bbox",  # Area of the bounding box i.e. number of pixels of the bounding box scaled by pixel-area.
    "area_convex",  # Area of the convex hull image, the smallest convex polygon enclosing the region.
    "area_filled",  # Area of the region with all holes filled in.
    "axis_major_length",  # Length of the major axis of the ellipse that matches the second central moments of the region.
    "axis_minor_length",  # Length of the minor axis of the ellipse that matches the second central moments of the region.
#    "bbox",  # Bounding box (min_row, min_col, max_row, max_col) of the region.
#    "centroid",  # Centroid coordinate tuple (row, col) of the region.
#    "centroid_local",  # Centroid coordinate tuple (row, col) relative to the region bounding box.
#    "centroid_weighted",  # Centroid weighted with intensity values, giving (row, col) coordinates.
#    "centroid_weighted_local",  # Intensity-weighted centroid relative to the region bounding box.
#    "coords_scaled",  # Coordinates of the region scaled by spacing, in (row, col) format.
#    "coords",  # Coordinates of the region in (row, col) format.
    "eccentricity",  # Eccentricity of the ellipse matching the second moments of the region (range [0, 1), with 0 being circular).
    "equivalent_diameter_area",  # Diameter of a circle with the same area as the region.
    "euler_number",  # Euler characteristic: number of connected components minus the number of holes in the region.
    "extent",  # Ratio of the region’s area to the area of the bounding box (area / (rows * cols)).
    "feret_diameter_max",  # Maximum Feret's diameter: longest distance between points on the convex hull of the region.
#    "image",  # Binary region image, same size as the bounding box.
#    "image_convex",  # Binary convex hull image, same size as the bounding box.
#    "image_filled",  # Binary region image with holes filled, same size as the bounding box.
#    "image_intensity",  # Intensity image inside the region's bounding box.
    "inertia_tensor",  # Inertia tensor for rotation around the region's center of mass.
    "inertia_tensor_eigvals",  # Eigenvalues of the inertia tensor, in decreasing order.
    "intensity_max",  # Maximum intensity value in the region.
    "intensity_mean",  # Mean intensity value in the region.
    "intensity_min",  # Minimum intensity value in the region.
    # "intensity_std",  # Standard deviation of the intensity values in the region. # Doesnt work.
    "label",  # Label of the region in the labeled input image.
    "moments",  # Spatial moments up to the 3rd order.
    "moments_central",  # Central moments (translation-invariant) up to the 3rd order.
    "moments_hu",  # Hu moments (translation, scale, and rotation-invariant).
    "moments_normalized",  # Normalized moments (translation and scale-invariant) up to the 3rd order.
    "moments_weighted",  # Intensity-weighted spatial moments up to the 3rd order.
    "moments_weighted_central",  # Intensity-weighted central moments (translation-invariant) up to the 3rd order.
    "moments_weighted_hu",  # Intensity-weighted Hu moments (translation, scale, and rotation-invariant).
    "moments_weighted_normalized",  # Intensity-weighted normalized moments up to the 3rd order.
    "num_pixels",  # Number of foreground pixels in the region.
    "orientation",  # Orientation of the major axis of the ellipse that matches the second moments of the region.
    "perimeter",  # Perimeter of the object, approximated using a 4-connectivity contour.
    "perimeter_crofton",  # Perimeter estimated by the Crofton formula, based on 4 directions.
#    "slice",  # Slice object to extract the region from the source image.
    "solidity"  # Solidity: ratio of region area to convex hull area.
] # The removed properties are not informative as they relate to absolute positions in the image


# Detect all filenames in the user-specified input directories
img_filenames = glob.glob(os.path.join(image_directory, f'*{filetype}'))
mask_filenames = glob.glob(os.path.join(mask_directory, f'*{filetype}'))
# label_filenames = glob.glob(os.path.join(label_directory, '*.json'))
coord_filenames = glob.glob(os.path.join(coord_directory, '*.pkl'))

# Files are sorted to ensure that the corresponding files are matched
img_filenames.sort()
mask_filenames.sort()
# label_filenames.sort()
coord_filenames.sort()

# Prepare pd dataframe to store the features generated from slices
features_2D = pd.DataFrame()
features_3D = pd.DataFrame()

print(f'\n### Detected {len(img_filenames)} images, {len(mask_filenames)} mask-files and {len(coord_filenames)} coord-files.')

# Check if the number of images, masks and labels are the same
assert len(img_filenames) == len(mask_filenames) == len(coord_filenames), f'Error: Detected different amounts of files: {len(img_filenames)} images, {len(mask_filenames)} masks and {len(coord_filenames)} coord dicts.'

idx = 0 # Index for the images

# Load pairs of img-file and corresponding mask-file
for img_name, mask_name, coord_name in zip(img_filenames, mask_filenames, coord_filenames):

    print(f'\n### Processing image {idx+1}/{len(img_filenames)}.')
    idx += 1
    
    # Load image to numpy array
    img_path = os.path.join(image_directory, img_name)
    img = imageio.imread(img_path)
    img = np.array(img)
    image = tifffile.imread(img_path)
    print(f'\n### Image of shape {img.shape} loaded. Name: {img_name}')

    img_name = os.path.basename(img_name).split('.')[0]

    # Load masks to numpy array
    mask_path = os.path.join(mask_directory, mask_name)
    mask = imageio.imread(mask_path)
    mask = np.array(mask, dtype=np.uint32)
    print(f'\n### Mask array of shape {mask.shape} loaded. Name: {mask_name}')

    # Load the corresponding label file
    # with open(label_name, 'r') as f:
    #     labels = json.load(f)
    # print(f'\n### Label file {label_name} loaded.')

    # Load the corresponding coord file
    with open(coord_name, 'rb') as f:
        coords = pickle.load(f)
    print(f'\n### Coord file {coord_name} loaded.')
    
    # Loop for 2D feature extraction
    for z_layer in range(img.shape[0]):
        img_z = image[z_layer]
        mask_z = mask[z_layer]

        mask_2D_features = {}
        for prop in properties:
            try:
                # Extract properties one by one
                mask_2D_prop = regionprops_table(label_image=mask_z, intensity_image=img_z, properties=[prop]) # not using spacing=[1.0, 0.24, 0.24] for 2D
                for key, value in mask_2D_prop.items():
                    mask_2D_features[key] = value
            except Exception as e:
                print(f'Error calculating 2D property {prop} on z-layer {z_layer}: {e}')
        
        mask_2D_features_df = pd.DataFrame(mask_2D_features)
        features_2D = pd.concat([features_2D, mask_2D_features_df], ignore_index=True)

    # Group the 2D features by label, compute the mean of label across z layers
    features_2D['label'] = features_2D['label'].astype(str)
    features_2D = features_2D.groupby('label').mean(numeric_only=True).reset_index()

    # 3D feature extraction
    mask_3D_features = {}
    for prop in properties:
        try:
            mask_3D_prop = measure.regionprops_table(mask, intensity_image=img, spacing=[1.0, 0.24, 0.24], properties=[prop])
            for key, value in mask_3D_prop.items():
                mask_3D_features[key] = value
        except Exception as e:
            print(f'Error calculating 3D property {prop}: {e}')
    
    mask_3D_features_df = pd.DataFrame(mask_3D_features)
    features_3D = pd.concat([features_3D, mask_3D_features_df], ignore_index=True)

    if same_columns_flag:
        # Ensure common columns between 2D and 3D
        common_columns = features_2D.columns.intersection(features_3D.columns)
        features_2D = features_2D[common_columns]
        features_3D = features_3D[common_columns]

    # Align the labels in both 2D and 3D
    features_3D['label'] = features_3D['label'].astype(str)
    features_2D['label'] = features_2D['label'].astype(str)

    labels_2D = set(features_2D['label'])
    labels_3D = set(features_3D['label'])

    in_2D_not_in_3D = labels_2D - labels_3D
    in_3D_not_in_2D = labels_3D - labels_2D

    if in_2D_not_in_3D:
        print(f"Masks in 2D but not in 3D: {in_2D_not_in_3D}")
    if in_3D_not_in_2D:
        print(f"Masks in 3D but not in 2D: {in_3D_not_in_2D}")

    # Drop rows with labels that are not in labels, if provided label file
    # features_2D = features_2D[features_2D['label'].isin(labels)]
    # features_3D = features_3D[features_3D['label'].isin(labels)]

    # Rename the label column to mask_id
    features_2D.rename(columns={'label': 'mask_id'}, inplace=True)
    features_3D.rename(columns={'label': 'mask_id'}, inplace=True)

    # # Add mask_id to the features
    # features_2D['mask_id'] = features_2D['label']
    # features_3D['mask_id'] = features_3D['label']

    # # Replace labels with cell types
    # mask_id_label = {mask_id: labels[mask_id]['ctype'] for mask_id in labels}
    # id_to_drop = [str(mask_id) for mask_id, label in mask_id_label.items() if label == 'unknown']
    # mask_id_label = {str(mask_id): label for mask_id, label in mask_id_label.items() if label != 'unknown'}
    
    # features_2D = features_2D[~features_2D['label'].isin(id_to_drop)]
    # features_3D = features_3D[~features_3D['label'].isin(id_to_drop)]

    # mask_id_label = {mask_id: 1 if label == 'megakaryocytic' else 0 for mask_id, label in mask_id_label.items()}
    # features_2D['label'] = features_2D['label'].map(mask_id_label)
    # features_3D['label'] = features_3D['label'].map(mask_id_label)
    
    # Save the features to disk
    features_2D.to_csv(os.path.join(out_dir, f'features_2D_{img_name}.csv'), index=False)
    features_3D.to_csv(os.path.join(out_dir, f'features_3D_{img_name}.csv'), index=False)

    print(f'\n### 2D features shape: {features_2D.shape}.')
    print(f'\n### 3D features shape: {features_3D.shape}.')
    print(f'\n### Features saved to {out_dir}.')

    # Reset dataframes for the next iteration
    features_2D = pd.DataFrame()
    features_3D = pd.DataFrame()

print('### Done.')
