'''
This script is used to visualize the 3D images and segmentation masks using napari, and to draw a rectangle
around the existing crop region. This script is a remnant of the initial exploration phase of the project and
is not used in the final pipeline. It is kept here for reference purposes.
'''

import napari
import numpy as np
import tifffile as tiff
import os


# Paths to the image directories
image_dir = "/Users/agreic/Desktop/Project/Data/Raw/Images/nucleus/"
segmentation_dir = ""
sca1_dir = "/Users/agreic/Desktop/Project/Data/Raw/Images/Sca1/"

# Load the images and segmentation masks
image = tiff.imread(os.path.join(image_dir, "sca1_extra_mask.tif"))
sca1 = tiff.imread(os.path.join(sca1_dir, "240930_C5_SCA1.tif"))

viewer = napari.Viewer()
viewer.add_image(image, colormap='gray')
viewer.add_image(sca1, colormap='green', blending='additive')
# viewer.add_labels(segmentation)

# Draw a rectangle filled with red to signify existing crop around: 50-130_3940-6672_4100-6661 (z,y,x
# Define the rectangle coordinates using four corner vertices
ymin = 3940
ymax = 6672
xmin = 4100
xmax = 6661

layer= viewer.add_shapes(np.array([[3940,4100],[6672,6661]]), shape_type='rectangle')

# # Define the rectangle coordinates using four corner vertices
# rectangle_coords = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])

# # Add the rectangle to the viewer
# layer2 = viewer.add_shapes(rectangle_coords, shape_type='rectangle', edge_color='red', edge_width=5, face_color='royalblue', opacity=0.5)


# Start the napari event loop
napari.run()