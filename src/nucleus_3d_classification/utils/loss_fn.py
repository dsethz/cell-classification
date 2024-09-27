import torch
import numpy as np
import torch.nn.functional as F
    

def calculate_class_weights(dataloader):
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

def create_loss_fn_with_weights(dataloader, loss_fn, weight=None):
    """
    Create a weighted loss function based on class distribution in the dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Training dataloader for calculating class weights.

    Returns:
        function: A weighted loss function (cross_entropy) with calculated class weights.
    """

    match weight:
        case 'balanced':
            weight = calculate_class_weights(dataloader)

            print("Calculated class weights are:", weight)

            match loss_fn:
                case 'cross_entropy':
                    def weighted_cross_entropy(pred, target):
                        return F.cross_entropy(pred, target, weight=weight)
                case 'bce':
                    def weighted_bce(pred, target):
                        return F.binary_cross_entropy(pred, target, weight=weight)
                case 'mse':
                    def weighted_mse(pred, target):
                        return F.mse_loss(pred, target, weight=weight)
                case _:
                    raise ValueError(f"Invalid loss_fn argument: {loss_fn}")

        case None:
            match loss_fn:

                case 'cross_entropy':
                    def weighted_cross_entropy(pred, target):
                        return F.cross_entropy(pred, target)
                case 'bce':
                    def weighted_bce(pred, target):
                        return F.binary_cross_entropy(pred, target)
                case 'mse':
                    def weighted_mse(pred, target):
                        return F.mse_loss(pred, target)
                case _:
                    raise ValueError(f"Invalid weight argument: {weight}")
                
        case _:
            raise ValueError(f"Invalid weight argument: {weight}")
            
    return ValueError(f"Ivalid dataloader argument: {dataloader}")

# Import customdata module
# from datamodule import CustomDataModule
# from padding import pad

def main():
    # Usage example:
    data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
    data_module.prepare_data()
    data_module.setup()
    train_dataloader = data_module.train_dataloader()

    # Create a loss function with dynamic class weights
    loss_fn = create_loss_fn_with_weights(train_dataloader)
    print(loss_fn, "and the class weights are:", calculate_class_weights(train_dataloader))

if __name__ == "__main__":
    main()
