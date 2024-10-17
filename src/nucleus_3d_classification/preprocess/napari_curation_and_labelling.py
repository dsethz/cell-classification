# Mask Coord Dict is a dictionary that stores the coordinates of the masks in the labels layer,
# generated from 'generate_mask_coord_dict.py', found at:
# 'https://github.com/CSDGroup/3d_segmentation/blob/main/scripts/generate_mask_coord_dict.py'.
# This file is not required, but it greatly improves the efficiency of the curation process by
# allowing the user to jump to the next mask directly, speeding up the process.

# Mask Dictionary is a dictionary that stores the cell type (under 'ctype') and centroid of each mask.
# The cell type is assigned by the user, and the centroid is calculated from the mask coordinates (from the pickle dictionary file, if provided!).

import json
import os
import pickle
import argparse

import napari
import numpy as np
import tifffile
from magicgui import magicgui
from qtpy.QtWidgets import (
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Add argparse options for the script - either curation or labelling mode
parser = argparse.ArgumentParser(description="Napari Curation and Labelling Tool")

parser.add_argument(
                '--mode', 
                type=str,
                choices=['curation', 'labelling'],
                default='curation',
                help="Select the mode for the tool (curation or labelling)"
)
parser.add_argument(
                '--save_mask_array_bool',
                type=bool,
                default=False,
                help="Save the mask array every 10th mask? (True/False)"
)


args = parser.parse_args()
mode = args.mode
save_mask_array_bool = args.save_mask_array_bool

# Initialize the Napari viewer
viewer = napari.Viewer()

# Global variables to store mask data
mask_dict = {}
mask_ids = []
current_mask_idx = -1
sdict = {}  # Dictionary to store mask IDs, cell types, and centroids
id_current = 0
filtered_mask_ids = []  # List to store IDs of masks with the selected label
current_filtered_idx = -1
selected_label = None
loaded_mask_dict = False
total_len = 0
temp_path = None
brush_size = 1
loaded_labels_layer = None

def update_brush_size(size):
    """Update the brush size of the active labels layer."""
    global brush_size, loaded_labels_layer
    if loaded_labels_layer is not None:
        brush_size = size
        loaded_labels_layer.brush_size = brush_size
        print(f"Brush size updated to {size}")

# Save the mask_array every time next mask is called
def save_mask_array(labels_layer):
    global temp_path, loaded_mask_dict
    if loaded_mask_dict:
        head, _ = os.path.split(temp_path)
        mask_array_path = head + '/temp_mask_array.tif'
        # Saving takes up a lot of time, need to find better alternative.
        tifffile.imwrite(mask_array_path, labels_layer.data, compression=('zlib', 1))
        print(f"Mask array saved to {mask_array_path}")

# Function to select the labels layer
@magicgui(call_button="Select Labels Layer", labels_layer={"choices": []})
def select_labels_layer(labels_layer: napari.layers.Labels):
    global mask_ids, current_mask_idx, sdict, total_len, loaded_labels_layer
    mask_ids = []
    total_len = 0
    if labels_layer.data is not None:
        loaded_labels_layer = labels_layer
        unique_labels = np.unique(labels_layer.data)

        # Remove 0 from the unique labels
        unique_labels = unique_labels[unique_labels != 0]
        total_len = len(unique_labels)
        mask_ids = unique_labels.tolist()

        # Create a dictionary entry for each mask ID
        for mask_id in mask_ids:
            sdict[mask_id] = {"ctype": "unknown", "centroid": None}

    update_brush_size(brush_size)
    current_mask_idx = -1  # Reset index for curation
    print(f"Found {len(mask_ids)} mask(s) in the selected labels layer.")

# Function to center on a specific mask and update its centroid
def center_on_mask(labels_layer, mask_id):
    global sdict, id_current, current_mask_idx, save_mask_array_bool
    id_current = mask_id

    # Retrieve the coordinates from the dictionary
    if loaded_mask_dict:
        if mask_id in mask_dict:
            z_indices, y_indices, x_indices = mask_dict[mask_id]
            print(f'Mask ID {mask_id} found in the mask coord dictionary.')

            # Save every idx%10
            if save_mask_array_bool:
                if current_mask_idx % 10 == 0:
                    save_mask_array(labels_layer)

        else:
            z_indices, y_indices, x_indices = np.where(labels_layer.data == mask_id)
            print(f"Mask ID {mask_id} not found in the mask dictionary. Using the labels layer instead.")
    else:
        z_indices, y_indices, x_indices = np.where(labels_layer.data == mask_id)

    if len(z_indices) == 0:
        print(f"Mask ID {mask_id} has no coordinates.")
        return

    # Calculate centroid
    centroid = (
        int(np.mean(x_indices)),
        int(np.mean(y_indices)),
        int(np.mean(z_indices))
    )

    # Update the dictionary with the centroid
    #print(f"Updating mask {mask_id} with centroid {centroid}")
    # print(f"Mask ID in sdict.keys(): {mask_id in sdict.keys()}") # True, when not loading a json mask file, interesting.

    # Update or create dictionary entry
    if mask_id not in sdict:
        sdict[mask_id] = {"ctype": "unknown", "centroid": None}
        print(f"Created new entry for mask {mask_id} with centroid {centroid}, set ctype to unknown.")
    sdict[mask_id]["centroid"] = centroid

    # if mask_id in sdict.keys(): # TODO: Something is wrong here, the mask_id is never in the dictionary
    #     sdict[(mask_id)]["centroid"] = centroid
    #     print(f"Updated mask {mask_id} with centroid {centroid}")
    # else:
    #     print(f"Mask ID {mask_id} not found in the mask (json) dictionary, creating a new entry.")
    #     sdict[(mask_id)] = {"ctype": "unknown", "centroid": centroid}

    # Center viewer on the mask
    viewer.dims.set_point(0, centroid[2])
    viewer.camera.center = (centroid[1], centroid[0])
    viewer.camera.zoom = 8

    # Set the selected label in the labels layer to the current mask
    labels_layer.selected_label = mask_id

    print(f"Centered on mask {mask_id} at {centroid}")
    return centroid

# Function to move to the next mask
def next_mask(event=None):
    global current_mask_idx
    print(f"Current mask index: {current_mask_idx + 1}/{total_len}")
    if current_mask_idx < len(mask_ids) - 1:
        current_mask_idx += 1
    else:
        current_mask_idx = 0  # Loop back to the first mask
    mask_id = mask_ids[current_mask_idx]
    center_on_mask(select_labels_layer.labels_layer.value, mask_id)

# Function to move to the previous mask
def previous_mask(event=None):
    global current_mask_idx
    if current_mask_idx > 0:
        current_mask_idx -= 1
    else:
        current_mask_idx = len(mask_ids) - 1  # Loop to the last mask
    mask_id = mask_ids[current_mask_idx]
    center_on_mask(select_labels_layer.labels_layer.value, mask_id)

# Function to validate and jump to the mask
def jump_to_mask():
    global current_mask_idx
    try:
        mask_id = int(mask_id_input.text())
        if mask_id in mask_ids:
            center_on_mask(select_labels_layer.labels_layer.value, mask_id)
            #print(f"Jumped to mask {mask_id}")
            current_mask_idx = mask_ids.index(mask_id)
        else:
            print(f"Mask ID {mask_id} is not valid. Please enter an existing mask ID.")
            # A non-valid mask is one that doesnt have coordinates,
            # so if it was a negative mask that was not picked in random selection, it wouldnt be valid!
    except ValueError:
        print("Please enter a valid integer for the Mask ID.")

# Function to assign the cell type to the current mask
@magicgui(call_button="Assign Cell Type", cell_type={"choices": ["positive", "negative", "unknown"]}) # "megakaryocytic" changed to "positive"
def assign_cell_type(cell_type: str):
    global current_mask_idx, mask_ids, sdict, id_current
    mask_id = id_current
    if mask_id != 0 and mask_id in sdict:
        sdict[int(mask_id)]["ctype"] = cell_type
        print(f"Assignment: {mask_id}: ctype = {cell_type}, centroid = {sdict[int(mask_id)]['centroid']}")
        next_mask()  # Automatically move to the next mask

# Function to filter masks by label
@magicgui(call_button="Filter Masks by Label", label={"choices": ["negative", "positive", "unknown"]}) # "megakaryocytic" changed to "positive"
def filter_masks_by_label(label="unknown"):
    global filtered_mask_ids, selected_label, current_filtered_idx, sdict
    selected_label = label

    # Find masks with the selected label
    filtered_sdict = {int(k): v for k, v in sdict.items() if v['ctype'] == selected_label}
    filtered_mask_ids = list(filtered_sdict.keys())

    current_filtered_idx = -1  # Reset index
    print(f"Filtered {len(filtered_mask_ids)} masks with label '{selected_label}'.")
    print(f'The filtered mask ids are: {filtered_mask_ids}')

# Modify the function to jump to the next filtered mask
def next_filtered_mask(event=None):
    global current_filtered_idx, filtered_mask_ids
    if len(filtered_mask_ids) > 0:
        current_filtered_idx += 1  # Increment index
        current_filtered_idx %= len(filtered_mask_ids)  # Loop back to the first mask if needed

        mask_id = filtered_mask_ids[current_filtered_idx]
        mask_id_input.setText(str(mask_id))  # Update the jump input field

        print('Jumping to filtered mask id:', mask_id)
        jump_to_mask()  # Use the jump function instead of next_mask to move directly
    else:
        print("No masks found with the selected label.")

def previous_filtered_mask(event=None):
    global current_filtered_idx, filtered_mask_ids
    if len(filtered_mask_ids) > 0:
        current_filtered_idx -=1  # Decrease index
        current_filtered_idx %= len(filtered_mask_ids)  # Loop back to the first mask if needed

        mask_id = filtered_mask_ids[current_filtered_idx]
        mask_id_input.setText(str(mask_id))  # Update the jump input field

        print('Jumping to filtered mask id:', mask_id)
        jump_to_mask()  # Use the jump function instead of next_mask to move directly
    else:
        print("No masks found with the selected label.")

# Function to save the mask dictionary to a JSON file
def save_mask_dict():
    global sdict
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getSaveFileName(caption="Save Mask Dictionary As", filter="JSON Files (*.json)")

    if file_path:
        sdict_converted = {int(k): v for k, v in sdict.items()}
        with open(file_path, 'w') as file:
            json.dump(sdict_converted, file)
        print(f"Mask dictionary saved to {file_path}")
    else:
        print("Save operation canceled.")

# Function to load the mask dictionary from a JSON file
def load_mask_dict():
    global sdict
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(caption="Load Mask Dictionary (JSON)", filter="JSON Files (*.json)")

    if file_path:
        with open(file_path, 'r') as file:
            sdict = json.load(file)
        # Convert keys back to integers
        sdict = {int(k): v for k, v in sdict.items()}
        print(sdict.keys(), sdict.values())
        print(f"Mask (cell type) dictionary loaded from {file_path}. Total masks: {len(sdict)}")
    else:
        print("Load operation canceled.")

# Function to load the mask dictionary from a .pkl file
@magicgui(call_button="Load Mask Coord Dict (Pickle)")
def load_mask_dict_jump():
# This widget uses files generated by the script found at:
# 'https://github.com/CSDGroup/3d_segmentation/blob/main/scripts/generate_mask_coord_dict.py'.
    global mask_dict, loaded_mask_dict, temp_path
    file_dialog = QFileDialog()
    file_path, _ = file_dialog.getOpenFileName(caption="Select Mask Dictionary File", filter="Pickle Files (*.pkl)")
    temp_path = file_path
    if file_path:
        with open(file_path, 'rb') as file:
            mask_dict = pickle.load(file)
        print(f"Mask dictionary (for moving to next mask) loaded from {file_path}.pkl, length is {len(mask_dict)}")
        loaded_mask_dict = True
    else:
        print("No file selected.")

# Add the widget to load the mask dictionary
load_dict_button = QPushButton("Load Mask Coord Dict (Pickle)")
load_dict_button.clicked.connect(load_mask_dict_jump)

# Create buttons for mask navigation and dictionary operations
next_button = QPushButton("Next Mask")
next_button.clicked.connect(next_mask)

prev_button = QPushButton("Previous Mask")
prev_button.clicked.connect(previous_mask)

save_button = QPushButton("Save Mask Dictionary JSON")
save_button.clicked.connect(save_mask_dict)

load_button = QPushButton("Load Mask Dictionary JSON")
load_button.clicked.connect(load_mask_dict)

# Filtered mask navigation
next_filtered_button = QPushButton("Next Filtered Mask")
next_filtered_button.clicked.connect(next_filtered_mask)

prev_filtered_button = QPushButton("Previous Filtered Mask")
prev_filtered_button.clicked.connect(previous_filtered_mask)

# Create a QWidget to hold control buttons
control_widget = QWidget()
layout = QVBoxLayout()
layout.addWidget(load_button)
layout.addWidget(filter_masks_by_label.native)
layout.addWidget(next_filtered_button)
layout.addWidget(prev_filtered_button)
control_widget.setLayout(layout)

# Create a custom widget for jumping to masks
mask_jump_widget = QWidget()
layout = QVBoxLayout()
mask_id_input = QLineEdit()
mask_id_input.setPlaceholderText("Enter Mask ID")
jump_button = QPushButton("Jump to Mask")
jump_button.clicked.connect(jump_to_mask)
layout.addWidget(QLabel("Enter Mask ID:"))
layout.addWidget(mask_id_input)
layout.addWidget(jump_button)
mask_jump_widget.setLayout(layout)

# Add widgets to the viewer
viewer.window.add_dock_widget(load_mask_dict_jump, area='right')
viewer.window.add_dock_widget(select_labels_layer, area='right')
viewer.window.add_dock_widget(assign_cell_type, area='right')
viewer.window.add_dock_widget(next_button, area='right')
viewer.window.add_dock_widget(prev_button, area='right')
viewer.window.add_dock_widget(save_button, area='right')
viewer.window.add_dock_widget(mask_jump_widget, area='right')
viewer.window.add_dock_widget(control_widget, area='right')

# Bind hotkeys for next and previous mask
viewer.bind_key('r', next_mask)
viewer.bind_key('f', previous_mask)

if mode == 'labelling':
    viewer.bind_key("q", lambda event: assign_cell_type(cell_type="negative"))
    viewer.bind_key("w", lambda event: assign_cell_type(cell_type="positive"))
    viewer.bind_key("e", lambda event: assign_cell_type(cell_type="unknown"))

elif mode == 'curation':
    # Bind hotkeys to increase brush size to 10, 20 and 40:
    viewer.bind_key('q', lambda _: update_brush_size(10))
    viewer.bind_key('w', lambda _: update_brush_size(20))
    viewer.bind_key('e', lambda _: update_brush_size(40))

# viewer.bind_key('s', lambda event: save_mask_array(select_labels_layer.labels_layer.value))

# Start the Napari viewer event loop
napari.run()
