# This script fits models to the 2D and 3D features extracted using the script 'extract_2D_3D_features.py'.
# This scrip will take in a single parameter, a json file, stating which data to use for training, validation and testing.
# Based on this, it will generate the output - result.csv files, containing the accuracies and other metrics for the models on both, the validation and test sets.
# Additionally, all the models generated will be saved in a directory, that will be specified in the json file (based on the label, eg cd41, gata1, etc.).

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import json
import argparse


# Function to load the CSV data from a list of multiple CSV files,
# match the multiple files to the multiple JSON files,
# and finally return one long dataframe with the data and labels.
def load_data(data, labels):
    # Ensure that the input is a list, even if a single string is provided
    if isinstance(data, str):
        data = [data]
    if isinstance(labels, str):
        labels = [labels]

    assert len(data) == len(labels), "Data and labels must be of the same length."

    # Check if files exist
    for i in range(len(data)):
        assert os.path.exists(data[i]), f"Data file {data[i]} does not exist."
        assert os.path.exists(labels[i]), f"Label file {labels[i]} does not exist."

    # Initialize a list to hold the DataFrames
    data_list = []

    for i in range(len(data)):
        # Load CSV data
        temp_data = pd.read_csv(data[i])

        # Load corresponding JSON labels
        with open(labels[i], 'r') as label_file:
            temp_label = json.load(label_file)

        # Create mask_id to label mapping, filtering out 'unknown' labels
        mask_id_to_label = {
            str(mask_id): str(temp_label[mask_id]['ctype'])
            for mask_id in temp_label if temp_label[mask_id]['ctype'] != 'unknown'
        }

        # Ensure that 'mask_id' exists in temp_data
        if 'mask_id' in temp_data.columns:
            # Convert 'mask_id' in the DataFrame to string for consistency
            temp_data['mask_id'] = temp_data['mask_id'].astype(str)

            # Filter out rows with mask_ids that have 'unknown' label
            temp_data = temp_data[temp_data['mask_id'].isin(mask_id_to_label.keys())]

            # Map the 'mask_id' to its corresponding label and create a new 'label' column
            temp_data['label'] = temp_data['mask_id'].map(
                lambda x: 1 if mask_id_to_label[x] == 'positive' or mask_id_to_label[x] == 'megakaryocytic' else 0
            )
        else:
            raise KeyError("The column 'mask_id' does not exist in the DataFrame.")

        # Append temp_data to the list
        data_list.append(temp_data)

    # Concatenate all DataFrames into one
    df = pd.concat(data_list, ignore_index=True)

    print(f"Data shape: {df.shape}")
    return df

# Clean the data, in case there are unexpected columns:
def trim(x):
    # Remove columns that contain 'Identifier' or 'mask_id' in their names
    x = x.loc[:, ~x.columns.str.contains('Identifier', case=False, na=False)]
    x = x.loc[:, ~x.columns.str.contains('mask_id', case=False, na=False)]

    # Remove feret_diameter_max, as this can cause issues for some models.
    # x = x.loc[:, ~x.columns.str.contains('feret_diameter_max', case=False, na=False)]

    # Remove rows where the 'label' column is missing
    x = x.dropna(subset=['label'])

    # Replace infinite values with NaN
    x = x.replace([np.inf, -np.inf], np.nan)

    # Remove columns that contain any NaN values
    x = x.dropna(axis=1, how='any')

    print(f"Data shape after trimming: {x.shape}")
    return x

def prep_trim_scale_data(train_data, val_data, test_data):

    # Remove the columns that are not useful
    train_data = trim(train_data)
    val_data = trim(val_data)
    test_data = trim(test_data)

    train_X = train_data.drop(columns=['label'])
    # train_X = scaler.fit_transform(train_X)
    # Add back col names.
    train_X = pd.DataFrame(train_X, columns = train_data.drop(columns=['label']).columns)
    train_y = train_data['label']
    val_X = val_data.drop(columns=['label'])
    # val_X = scaler.transform(val_X)
    val_X = pd.DataFrame(val_X, columns = val_data.drop(columns=['label']).columns)
    val_y = val_data['label']
    test_X = test_data.drop(columns=['label'])
    # test_X = scaler.transform(test_X)
    test_X = pd.DataFrame(test_X, columns = test_data.drop(columns=['label']).columns)
    test_y = test_data['label']

    return train_X, train_y, val_X, val_y, test_X, test_y

def fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type= '3D'):

    # Print out the classes and their counts before fitting
    print("Classes and their counts before fitting:")
    print("Train:", train_y.value_counts())
    print("Validation:", val_y.value_counts())
    print("Test:", test_y.value_counts())

def main():

    parser = argparse.ArgumentParser(description='Fit models to the 2D and 3D features extracted using the script extract_2D_3D_features.py.')
    parser.add_argument('--setup', type=str, default="/Users/agreic/Documents/GitHub/cell-classification/Analysis/cd41_fit.json", help='File containing the neccessary setup for the model fitting.')

    args = parser.parse_args()

    setup = json.load(open(args.setup, 'r'))
    label_name = setup['label_name']
    output_dir = setup['output_directory']
    param_grid = setup['param_grid']

    # Data are CSV files of the features extracted from the images and masks. They contain column names.
    train_data = []
    val_data = []
    test_data = []

    # Load the data
    train_data = load_data(setup['train_data'], setup['train_labels'])
    val_data = load_data(setup['val_data'], setup['val_labels'])
    test_data = load_data(setup['test_data'], setup['test_labels'])

    train_X, train_y, val_X, val_y, test_X, test_y = prep_trim_scale_data(train_data, val_data, test_data)

    fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type= '3D')

    # Load the 2D data, if it was provided. This is optional, but they do share the same labels.
    if ('train_data_2d' in setup) and 'val_data_2d' in setup and 'test_data_2d' in setup:

        train_data = load_data(setup['train_data_2d'], setup['train_labels'])
        val_data = load_data(setup['val_data_2d'], setup['val_labels'])
        test_data = load_data(setup['test_data_2d'], setup['test_labels'])

        train_X, train_y, val_X, val_y, test_X, test_y = prep_trim_scale_data(train_data, val_data, test_data)
        fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type= '2D')

if __name__ == "__main__":
    main()