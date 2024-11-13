# This script fits baseline models to the 2D and 3D features extracted using the script 'extract_2D_3D_features.py'.
# This scrip will take in a single parameter, a json file, stating which data to use for training, validation and testing.
# Based on this, it will generate the output - result.csv files, containing the accuracies and other metrics for the models on both, the validation and test sets.
# Additionally, all the models generated will be saved in a directory, that will be specified in the json file (based on the label, eg cd41, gata1, etc.).

# The json files are available in the same directory as this script, and are named as follows:
# cd41_fit.json
# gata1_fit.json
# pu1_fit.json
# sca_fit.json

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, RocCurveDisplay, roc_auc_score, precision_recall_curve, auc, f1_score
import pickle
from sklearn.preprocessing import StandardScaler
from itertools import product
from numba import prange

# Function to detect diff columns and remove them, printing it out too.
def remove_diff_columns(df1, df2, df3):
    """Remove columns that are not present in all dataframes."""
    
    # Print out the columns to be removed:
    columns_to_remove = (set(df1.columns) - set(df2.columns)).union(
                        set(df1.columns) - set(df3.columns)).union(
                        set(df2.columns) - set(df1.columns)).union(
                        set(df2.columns) - set(df3.columns)).union(
                        set(df3.columns) - set(df1.columns)).union(
                        set(df3.columns) - set(df2.columns))
    # Check if the set is empty:
    if columns_to_remove:
        print("Columns being removed:", columns_to_remove)

    # Find the common columns
    common_columns = df1.columns.intersection(df2.columns).intersection(df3.columns)

    # Remove the different columns
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    df3 = df3[common_columns]

    # Print out the shape of the dataframes
    print(f"Data shape after removing different columns: {df1.shape}, {df2.shape}, {df3.shape}")
    return df1, df2, df3

# Function to load the CSV data from a list of multiple CSV files,
# match the multiple files to the multiple JSON files,
# and finally return one long dataframe with the data and labels.
def load_data(data, labels):
    """Load CSV data and match it to JSON labels."""
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
    """Remove columns that are not useful."""
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
    """Prepare, trim, and scale the data."""
    # Remove the columns that are not useful
    train_data = trim(train_data)
    val_data = trim(val_data)
    test_data = trim(test_data)

    train_data, val_data, test_data = remove_diff_columns(train_data, val_data, test_data)

    scaler = StandardScaler()

    train_X = train_data.drop(columns=['label'])
    train_X = scaler.fit_transform(train_X)
    # Add back col names.
    train_X = pd.DataFrame(train_X, columns = train_data.drop(columns=['label']).columns)
    train_y = train_data['label']
    val_X = val_data.drop(columns=['label'])
    val_X = scaler.transform(val_X)
    val_X = pd.DataFrame(val_X, columns = val_data.drop(columns=['label']).columns)
    val_y = val_data['label']
    test_X = test_data.drop(columns=['label'])
    test_X = scaler.transform(test_X)
    test_X = pd.DataFrame(test_X, columns = test_data.drop(columns=['label']).columns)
    test_y = test_data['label']

    return train_X, train_y, val_X, val_y, test_X, test_y

def calc_metrics(y_true, model, X):
    """Calculate metrics for the model."""
    y_pred = model.predict(X)
    test_proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_true, model.predict_proba(X)[:, 1])    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, test_proba)
    pr_auc = auc(recall_curve, precision_curve)

    f1_score_ = f1 # Doing this twice because i was debugging, TODO: remove one

    confusion_matrix_ = cm
    tn, fp, fn, tp = confusion_matrix_.ravel()
    confusion_matrix_ = f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}'

    return acc, cm, precision, recall, f1, roc_auc, pr_auc, f1_score_, confusion_matrix_


def fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type= '3D'):
    """Fit and evaluate a logistic regression model."""
    logreg_dir = os.path.join(output_dir, 'logreg')
    logreg_2d_dir = os.path.join(logreg_dir, '2d')
    logreg_3d_dir = os.path.join(logreg_dir, '3d')

    if not os.path.exists(logreg_dir):
        os.makedirs(logreg_dir)
    if not os.path.exists(logreg_2d_dir):
        os.makedirs(logreg_2d_dir)
    if not os.path.exists(logreg_3d_dir):
        os.makedirs(logreg_3d_dir)
    
    # Print out the classes and their counts before fitting
    print("Classes and their counts before fitting:")
    print("Train:", train_y.value_counts())
    print("Validation:", val_y.value_counts())
    print("Test:", test_y.value_counts())

    # For Feature analysis, we will save the train_X to a csv_file.
    if feature_type == '3D':
        # Save as csv
        train_X_df = pd.DataFrame(train_X)
        train_X_df.to_csv(os.path.join(logreg_3d_dir, 'train_3D.csv'))
        # Save labels
        train_y_df = pd.DataFrame(train_y, columns=['label'])
        train_y_df.to_csv(os.path.join(logreg_3d_dir, 'train_3D_labels.csv'))
    elif feature_type == '2D':
        # Save as csv
        train_X_df = pd.DataFrame(train_X)
        train_X_df.to_csv(os.path.join(logreg_2d_dir, 'train_2D.csv'))
        # Save labels
        train_y_df = pd.DataFrame(train_y, columns=['label'])
        train_y_df.to_csv(os.path.join(logreg_2d_dir, 'train_2D_labels.csv'))

    model = LogisticRegression(max_iter=100000, class_weight='balanced')
    model.fit(train_X, train_y)

    # Calculate the metrics
    train_acc, train_cm, train_precision, train_recall, train_f1, train_roc_auc, train_pr_auc, train_f1_score_, train_confusion_matrix_ = calc_metrics(train_y, model, train_X)
    val_acc, val_cm, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, val_f1_score_, val_confusion_matrix_ = calc_metrics(val_y, model, val_X)
    test_acc, test_cm, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, test_f1_score, test_confusion_matrix_ = calc_metrics(test_y, model, test_X)

    # Save the model and metrics.
    if feature_type == '3D':
        with open(os.path.join(logreg_3d_dir, 'logreg_3d.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(logreg_3d_dir, 'metrics_3d.txt'), 'w') as f:
            f.write(f"Train Accuracy: {train_acc}\n")
            f.write(f"Train Precision: {train_precision}\n")
            f.write(f"Train Recall: {train_recall}\n")
            f.write(f"Train F1: {train_f1}\n")
            f.write(f"Train ROC AUC: {train_roc_auc}\n")
            f.write(f"Train PR AUC: {train_pr_auc}\n")
            f.write(f"Train Confusion Matrix: {train_confusion_matrix_}\n")

            f.write(f"Validation Accuracy: {val_acc}\n")
            f.write(f"Validation Precision: {val_precision}\n")
            f.write(f"Validation Recall: {val_recall}\n")
            f.write(f"Validation F1: {val_f1}\n")
            f.write(f"Validation ROC AUC: {val_roc_auc}\n")
            f.write(f"Validation PR AUC: {val_pr_auc}\n")
            f.write(f"Validation Confusion Matrix: {val_confusion_matrix_}\n")

            f.write(f"Test Accuracy: {test_acc}\n")
            f.write(f"Test Precision: {test_precision}\n")
            f.write(f"Test Recall: {test_recall}\n")
            f.write(f"Test F1: {test_f1}\n")
            f.write(f"Test ROC AUC: {test_roc_auc}\n")
            f.write(f"Test PR AUC: {test_pr_auc}\n")
            f.write(f"Test Confusion Matrix: {test_confusion_matrix_}\n")

    elif feature_type == '2D':
        with open(os.path.join(logreg_2d_dir, 'logreg_2d.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(logreg_2d_dir, 'metrics_2d.txt'), 'w') as f:
            f.write(f"Train Accuracy: {train_acc}\n")
            f.write(f"Train Precision: {train_precision}\n")
            f.write(f"Train Recall: {train_recall}\n")
            f.write(f"Train F1: {train_f1}\n")
            f.write(f"Train Confusion Matrix: {train_cm}\n")
            f.write(f"Train ROC AUC: {train_roc_auc}\n")
            f.write(f"Train PR AUC: {train_pr_auc}\n")
            f.write(f"Train F1 Score: {train_f1_score_}\n")
            f.write(f"Train Confusion Matrix: {train_confusion_matrix_}\n")

            f.write(f"Validation Accuracy: {val_acc}\n")
            f.write(f"Validation Precision: {val_precision}\n")
            f.write(f"Validation Recall: {val_recall}\n")
            f.write(f"Validation F1: {val_f1}\n")
            f.write(f"Validation Confusion Matrix: {val_cm}\n")
            f.write(f"Validation ROC AUC: {val_roc_auc}\n")
            f.write(f"Validation PR AUC: {val_pr_auc}\n")
            f.write(f"Validation F1 Score: {val_f1_score_}\n")
            f.write(f"Validation Confusion Matrix: {val_confusion_matrix_}\n")

            f.write(f"Test Accuracy: {test_acc}\n")
            f.write(f"Test Precision: {test_precision}\n")
            f.write(f"Test Recall: {test_recall}\n")
            f.write(f"Test F1: {test_f1}\n")
            f.write(f"Test Confusion Matrix: {test_cm}\n")
            f.write(f"Test ROC AUC: {test_roc_auc}\n")
            f.write(f"Test PR AUC: {test_pr_auc}\n")
            f.write(f"Test F1 Score: {test_f1_score}\n")
            f.write(f"Test Confusion Matrix: {test_confusion_matrix_}\n")

def rf_train(param_grid, output_dir, train_X, train_y, val_X, val_y, test_X, test_y, iters:int=3, feature_type='3D'):
    """Train and evaluate random forest models, based on provided iterations, hyperparameter grid."""
    rf_dir = os.path.join(output_dir, 'rf')
    rf_2d_dir = os.path.join(rf_dir, '2d')
    rf_3d_dir = os.path.join(rf_dir, '3d')

    if not os.path.exists(rf_dir):
        os.makedirs(rf_dir)
    if not os.path.exists(rf_2d_dir):
        os.makedirs(rf_2d_dir)
    if not os.path.exists(rf_3d_dir):
        os.makedirs(rf_3d_dir)

    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['criterion'],
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
        param_grid['bootstrap'],
        param_grid['max_features'],
        ['balanced']  # Include class_weight manually
    ))

    val_results = {}
    test_results = {}

    if feature_type == '3D':
        save_dir = rf_3d_dir
    elif feature_type == '2D':
        save_dir = rf_2d_dir

    # Iterate through outer loops (replicates) and train models
    for index in prange(iters):
        for param_set in param_combinations:
            # Unpack the hyperparameters
            n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, bootstrap, max_features, class_weight = param_set
            
            # Initialize RandomForestClassifier with the current set of hyperparameters
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                bootstrap=bootstrap,
                max_features=max_features,
                class_weight=class_weight
            )

            # Train the model
            rf.fit(train_X, train_y)

            # Save model using a unique name based on the combination
            model_name = f'rf_model_{index}_{n_estimators}_{criterion}_{max_depth}_{min_samples_split}_{min_samples_leaf}_{bootstrap}_{max_features}_{class_weight}.pkl'
            
            # Ensure the save directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            with open(os.path.join(save_dir, model_name), 'wb') as f:
                pickle.dump(rf, f)
            
            accuracy, _, precision, recall, f1_score_, roc_auc, pr_auc, _, confusion_matrix_ = calc_metrics(val_y, rf, val_X)
            # Store the results
            val_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1_score_,
            'confusion_matrix': confusion_matrix_,
            'hyperparameters': rf.get_params()
            }

            accuracy, _, precision, recall, f1_score_, roc_auc, pr_auc, _, confusion_matrix_ = calc_metrics(test_y, rf, test_X)
            # Store the results
            test_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'f1_score': f1_score_,
            'confusion_matrix': confusion_matrix_,
            'hyperparameters': rf.get_params()
            }
    
    # Order the models based on a metric.
    val_results = dict(sorted(val_results.items(), key=lambda x: x[1]['accuracy'], reverse=True))
    test_results = dict(sorted(test_results.items(), key=lambda x: x[1]['accuracy'], reverse=True))

    val_results_df = pd.DataFrame(val_results).T
    test_results_df = pd.DataFrame(test_results).T

    val_results_df.to_csv(os.path.join(rf_dir, f'{feature_type}_val_results.csv'))
    test_results_df.to_csv(os.path.join(rf_dir, f'{feature_type}test_results.csv'))

def main():

    parser = argparse.ArgumentParser(description='Fit models to the 2D and 3D features extracted using the script extract_2D_3D_features.py.')
    parser.add_argument('--setup', type=str, default="/Users/agreic/Documents/GitHub/cell-classification/Analysis/cd41_fit.json", help='File containing the neccessary setup for the model fitting.')

    args = parser.parse_args()

    setup = json.load(open(args.setup, 'r'))
    label_name = setup['label_name']
    output_dir = setup['output_directory']
    param_grid = setup['param_grid']
 
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create subdirectory for label name within output directory (if label_name already isnt the last subdirectory)
    if output_dir.split('/')[-1] != label_name:
        output_dir = os.path.join(output_dir, label_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # Data are CSV files of the features extracted from the images and masks. They contain column names.
    train_data = []
    val_data = []
    test_data = []

    # Labels are json files, with nested dictionaries, containing the structure mask_id : {'ctype': label, 'centroid': [x, y, z]}
    # We will need to extract the ctype for each mask_id and use it as the label for the model.
    # For this, we will first extract ctype and mask_id from json, then match them to the csv data by the mask_id label.
    train_labels = []
    val_labels = []
    test_labels = []

    # Load the data
    train_data = load_data(setup['train_data'], setup['train_labels'])
    val_data = load_data(setup['val_data'], setup['val_labels'])
    test_data = load_data(setup['test_data'], setup['test_labels'])

    train_X, train_y, val_X, val_y, test_X, test_y = prep_trim_scale_data(train_data, val_data, test_data)

    # Save the column names and the shape of each df to a text file.
    with open(os.path.join(output_dir, '3D_data_info.txt'), 'w') as f:

        f.write(f"Train data shape: {train_data.shape}\n")
        f.write(f"Data columns: {train_data.columns}\n")

        f.write(f"Val data shape: {val_data.shape}\n")
        f.write(f"Data columns: {val_data.columns}\n")

        f.write(f"Test data shape: {test_data.shape}\n")
        f.write(f"Data columns: {test_data.columns}\n")

        f.write(f"train_X shape: {train_X.shape}\n")
        f.write(f"train_y shape: {train_y.shape}\n")

        f.write(f"val_X shape: {val_X.shape}\n")
        f.write(f"val_y shape: {val_y.shape}\n")

        f.write(f"test_X shape: {test_X.shape}\n")
        f.write(f"test_y shape: {test_y.shape}\n")
    
    # Fit the logistic regression model to the 3D data
    fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type='3D')
    rf_train(param_grid, output_dir, train_X, train_y, val_X, val_y, test_X, test_y, iters=5, feature_type='3D')


    # Load the 2D data, if it was provided. This is optional, but they do share the same labels.
    if ('train_data_2d' in setup) and 'val_data_2d' in setup and 'test_data_2d' in setup:

        train_data = load_data(setup['train_data_2d'], setup['train_labels'])
        val_data = load_data(setup['val_data_2d'], setup['val_labels'])
        test_data = load_data(setup['test_data_2d'], setup['test_labels'])

        train_X, train_y, val_X, val_y, test_X, test_y = prep_trim_scale_data(train_data, val_data, test_data)

        # Save the column names and the shape of each df to a text file.
        with open(os.path.join(output_dir, '2D_data_info.txt'), 'w') as f:

            f.write(f"Train data shape: {train_data.shape}\n")
            f.write(f"Data columns: {train_data.columns}\n")

            f.write(f"Val data shape: {val_data.shape}\n")
            f.write(f"Data columns: {val_data.columns}\n")

            f.write(f"Test data shape: {test_data.shape}\n")
            f.write(f"Data columns: {test_data.columns}\n")

            f.write(f"train_X shape: {train_X.shape}\n")
            f.write(f"train_y shape: {train_y.shape}\n")

            f.write(f"val_X shape: {val_X.shape}\n")
            f.write(f"val_y shape: {val_y.shape}\n")

            f.write(f"test_X shape: {test_X.shape}\n")
            f.write(f"test_y shape: {test_y.shape}\n")

        # Fit the logistic regression model to the 2D data
        fit_eval_logreg(train_X, train_y, val_X, val_y, test_X, test_y, output_dir, feature_type='2D')
        rf_train(param_grid, output_dir, train_X, train_y, val_X, val_y, test_X, test_y, iters=5, feature_type='2D')


if __name__ == "__main__":
    main()