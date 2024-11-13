'''
Module for calculating class weights and creating (un-)weighted loss functions for the model.
The balanced weighting calculates the inverse of class frequencies from the training labels.
Functions:
    calculate_class_weights(setup_file):
        Calculate and normalize class weights from a setup file containing training label information.
    create_loss_fn_with_weights(setup_file, loss_fn, weight=None):
        Create a weighted loss function using class weights derived from the training label file within the setup file.

Old, unused functions:       
    _calculate_class_weights(dataloader):
        Calculate and normalize class weights from a dataloader's label distribution.
    _create_loss_fn_with_weights(dataloader, loss_fn, weight=None):
        Create a weighted loss function using class weights derived from a dataloader.
'''

import torch.nn.functional as F
import json
import numpy as np
import torch



def calculate_class_weights(setup_file):
    """
    Calculate class weights based on the frequency of labels in the training dataset.

    Args:
        setup_file (str): Path to the setup file containing the label file for training data. (This would be your --data argument)

    Returns:
        torch.Tensor: A tensor containing normalized class weights.
    """
    # Load the setup file
    with open(setup_file, 'r') as f:
        setup = json.load(f)

    training_data_list = setup["training_data"]

    # Dictionary to store the count of each label
    label_counts = {
        'positive': 0,
        'negative': 0
    }

    for label_crop in training_data_list:
        label_file = label_crop["label_file"]

        # Load and extract the counts of each label
        with open(label_file, 'r') as f:
            labels = json.load(f)

        for key, value in labels.items():
            if value['ctype'] == 'megakaryocytic': # This is a remnant from having a 'megakaryocytic' label
                label_counts['positive'] += 1
            if value['ctype'] == 'positive':
                label_counts['positive'] += 1
            if value['ctype'] == 'negative':
                label_counts['negative'] += 1

    print("Unique labels and their counts are:", label_counts)

    # Calculate class weights based on label frequencies
    label_values = np.array(list(label_counts.values()))
    class_weights = 1.0 / label_values
    class_weights = class_weights / class_weights.sum()

    print("Calculated class weights are:", class_weights)
    return torch.tensor(class_weights, dtype=torch.float)


def create_loss_fn_with_weights(setup_file, loss_fn, weight=None):
    """
    Create a weighted loss function based on class distribution in the dataloader.

    Args:
        setup_file (str): Path to the setup file containing the label file for training data.
        loss_fn (str): Type of loss function to create ('cross_entropy', 'bce', 'mse').
        weight (str or None): Weighting scheme for the loss ('balanced' or None).

    Returns:
        function: A weighted loss function (cross_entropy, bce, or mse) with calculated class weights.
    """
    if weight == 'balanced':
        weight = calculate_class_weights(setup_file)
        print("Calculated class weights are:", weight)

    if loss_fn == 'cross_entropy':
        def weighted_loss(pred, target):
            return F.cross_entropy(pred, target, weight=weight if weight == 'balanced' else None)

    elif loss_fn == 'bce':
        def weighted_loss(pred, target):
            return F.binary_cross_entropy(pred, target, weight=weight if weight == 'balanced' else None)

    elif loss_fn == 'mse':
        def weighted_loss(pred, target):
            return F.mse_loss(pred, target, weight=weight if weight == 'balanced' else None)

    else:
        raise ValueError(f"Invalid loss_fn argument: {loss_fn}")

    return weighted_loss

def _calculate_class_weights(dataloader):
    """
    Calculate class weights based on the frequency of labels in the training dataset.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader for the training dataset.

    Returns:
        torch.Tensor: A tensor containing normalized class weights.
    """
    # List to hold all labels in the dataset
    all_labels = []

    # Iterate through the entire dataloader to gather all labels
    for batch in dataloader:
        _, labels = batch  # Assuming batch returns (inputs, labels)
        all_labels.append(labels)

    # Concatenate all labels into one tensor
    all_labels = torch.cat(all_labels)
    
    # Use bincount to count occurrences of each class
    class_counts = torch.bincount(all_labels)
    
    # Calculate class weights (inverse of class frequencies)
    class_weights = 1.0 / class_counts.float()
    
    # Normalize weights to sum to 1
    class_weights = class_weights / class_weights.sum()

    return class_weights

def _create_loss_fn_with_weights(dataloader, loss_fn, weight=None):
    """
    Create a weighted loss function based on class distribution in the dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Training dataloader for calculating class weights.
        loss_fn (str): Type of loss function to create ('cross_entropy', 'bce', 'mse').
        weight (str or None): Weighting scheme for the loss ('balanced' or None).

    Returns:
        function: A weighted loss function (cross_entropy, bce, or mse) with calculated class weights.
    """
    if weight == 'balanced':
        weight = calculate_class_weights(dataloader)
        print("Calculated class weights are:", weight)

    if loss_fn == 'cross_entropy':
        def weighted_loss(pred, target):
            return F.cross_entropy(pred, target, weight=weight if weight == 'balanced' else None)

    elif loss_fn == 'bce':
        def weighted_loss(pred, target):
            return F.binary_cross_entropy(pred, target, weight=weight if weight == 'balanced' else None)

    elif loss_fn == 'mse':
        def weighted_loss(pred, target):
            return F.mse_loss(pred, target, weight=weight if weight == 'balanced' else None)

    else:
        raise ValueError(f"Invalid loss_fn argument: {loss_fn}")

    return weighted_loss


# def main():
    # # Usage example:
    # data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
    # data_module.prepare_data()
    # data_module.setup()
    # train_dataloader = data_module.train_dataloader()

    # # Create a loss function with dynamic class weights
    # loss_fn = create_loss_fn_with_weights(train_dataloader)
    # print(loss_fn, "and the class weights are:", calculate_class_weights(train_dataloader))

# if __name__ == "__main__":
#     main()
