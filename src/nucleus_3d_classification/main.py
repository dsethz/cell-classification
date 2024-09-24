######################################################################################################################
# This script coordinates training and testing of cell segmentation with Pytorch Lightning                           #
# Author:               Aurimas Greicius, Daniel Schirmacher                                                         #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                              #
# Python Version:       3.12.2                                                                                       #
# PyTorch Version:      2.3.1                                                                                        #
# PyTorch Lightning Version: 2.3.1                                                                                   #
######################################################################################################################

import os
import sys
from typing import List, Optional
import argparse
import lightning as L
import pytorch_lightning as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import json
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import Callback

# Tensorboard
import tensorboard

from utils.testdatamodule import BaseDataModule
from models.testmodels import BaseNNModel, BaseNNModel2, testBlock, get_BaseNNModel, get_BaseNNModel2

import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch

from utils.datamodule import CustomDataModule
from models.ResNet import ResNet50, ResNet101, ResNet152, ResNet, Block, ResNet_custom_layers

'''
NotImplementedError: The operator 'aten::max_pool3d_with_indices' is not currently implemented 
for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, 
please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment 
variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than 
running natively on MPS.

Before running use terminal command:
export PYTORCH_ENABLE_MPS_FALLBACK=1

Otherwise maxpool will not work on MPS device
'''

class SklearnModelWrapper:
    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
def create_dir_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_model(model_name: str, class_weight: Optional[str] = None, max_iter: int = 1000):
    if model_name == "rf":
        return SklearnModelWrapper(RandomForestClassifier(class_weight=class_weight))
    elif model_name == "logreg":
        return SklearnModelWrapper(LogisticRegression(class_weight=class_weight, max_iter=max_iter))

def get_nn_model(model_name: str, extra_args: dict = None):

    if extra_args is None:
        extra_args = 'None'

    print(f"Creating model: {model_name} with extra_args: {extra_args}")

    if model_name == "BaseNNModel":
        model = get_BaseNNModel()
        print(f'Created model instance: {model}')
        print(f'Model is of type: {type(model)}')
        return model
    elif model_name == "BaseNNModel2":
        if 'Block' not in extra_args or 'layers' not in extra_args:
            raise ValueError("BaseNNModel2 requires 'Block' and 'layers' in extra_args.")

        model = get_BaseNNModel2(extra_args['Block'], extra_args['layers'])

        return model
    elif model_name == "ResNet50":
        return ResNet50(ceil_mode=True, num_classes=extra_args['num_classes'], image_channels=extra_args['image_channels'], padding_layer_sizes=extra_args['padding_layer_sizes'])
    elif model_name == "ResNet101":
        return ResNet101(ceil_mode=True, num_classes=extra_args['num_classes'], image_channels=extra_args['image_channels'], padding_layer_sizes=extra_args['padding_layer_sizes'])
    elif model_name == "ResNet152":
        return ResNet152(ceil_mode=True, num_classes=extra_args['num_classes'], image_channels=extra_args['image_channels'], padding_layer_sizes=extra_args['padding_layer_sizes'])
    elif model_name == "ResNet_custom_layers":
        return ResNet_custom_layers(layers=extra_args['layers'], ceil_mode=True, num_classes=extra_args['num_classes'], image_channels=extra_args['image_channels'], padding_layer_sizes=extra_args['padding_layer_sizes'])
    

def get_nn_model_class(model_name: str):
    if model_name == "ResNet50":
        return ResNet50
    elif model_name == "ResNet101":
        return ResNet101
    elif model_name == "ResNet152":
        return ResNet152
    elif model_name == "ResNet_custom_layers":
        return ResNet_custom_layers
    elif model_name == "BaseNNModel":
        return BaseNNModel
    elif model_name == "BaseNNModel2":
        return BaseNNModel2

def load_data(data_dir:str, data_file: str, target: str):
    # Implement data loading for RF and LogReg
    if target is None:
        target = 'label'
    # Load data from data_file
    if data_dir is None:
        data_dir = "./data"
    if data_file.endswith('.csv'):
        data = pd.read_csv(os.path.join(data_dir, data_file))
        # Remove columns with missing values
        data = data.dropna(axis=1)
    # Split into X and y
    X = data.drop(columns=[target])
    y = data[target]
    return X, y

def load_predict_data(data_dir:str, data_file: str, remove_label: bool = True):
    # Implement data loading for RF and LogReg
    if data_dir is None:
        data_dir = "./data"
    
    if not data_file.endswith('.csv'):
        raise ValueError("Prediction data must be a CSV file")

    if data_file.endswith('.csv'):
        data = pd.read_csv(os.path.join(data_dir, data_file))
        # Remove columns with missing values
        data = data.dropna(axis=1)
        # If theres a 'label' column, remove it, and print that it was removed by default!
        if 'label' in data.columns:
            print("Warning: 'label' column found in prediction data. Removing it by default. Set remove_label=False to keep it.")
            data = data.drop(columns=['label'])
    return data

def train(model, X, y, save_name: str = None, save_dir: str = "./models"):
    if isinstance(model, SklearnModelWrapper):
        model.fit(X, y)
        # Save the model
        if save_name:
            create_dir_if_not_exists(save_dir)
            with open(os.path.join(save_dir, f"{save_name}.pkl"), 'wb') as f:
                pickle.dump(model, f)

def predict(model, X, save_name: str = None, save_dir: str = "./predictions", save_type: str = 'csv'):
    if isinstance(model, SklearnModelWrapper):
        # Predict using the model
        y = model.predict(X)
        # Save the predictions
        if save_name:
            create_dir_if_not_exists(save_dir)
            if save_type == 'csv':
                pd.DataFrame(y).to_csv(os.path.join(save_dir, f"{save_name}.csv"), index=False)
            elif save_type == 'pkl':
                with open(os.path.join(save_dir, f"{save_name}.pkl"), 'wb') as f:
                    pickle.dump(y, f)
        return model.predict(X)

## Experimenting with Pytorch Lightning ####################################################################################################

    # TODO: Implement the following functions

def define_callbacks(callback_names: List[str]) -> List[pl.Callback]:
    callbacks = []
    for callback_name in callback_names:
        if callback_name == "early_stopping":
            callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss'))
        elif callback_name == "model_checkpoint":
            callbacks.append(pl.callbacks.ModelCheckpoint(monitor='val_loss'))
        elif callback_name == "lr_monitor":
            callbacks.append(pl.callbacks.LearningRateMonitor())
    return callbacks

def define_trainer(
        trainer_class, profiler: bool = False, 
        max_epochs: int = 10, 
        default_root_dir: str = "./logs",
        logger: pl.loggers.Logger = TensorBoardLogger,
        devices: str = "auto",
        accelerator: str = "auto",
        enable_checkpointing: bool = True,
        enable_early_stopping: bool = True,
        accumulate_grad_batches: int = 1,
        fast_dev_run: bool = False,
        limit_train_batches: int = 1.0,
        limit_val_batches: int = 1.0,
        limit_test_batches: int = 1.0,
        limit_predict_batches: int = 1.0,
        log_every_n_steps: int = 5,
        callbacks: List[pl.Callback] = None
        ):
    # Define the trainer based on the arguments
    trainer = trainer_class(
        profiler=profiler,
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        logger=logger,
        devices=devices,
        accelerator=accelerator,
        enable_checkpointing=enable_checkpointing,
        enable_early_stopping=enable_early_stopping,
        accumulate_grad_batches=accumulate_grad_batches,
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        limit_predict_batches=limit_predict_batches,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks
    )
    return trainer

######################################################################################################################


def main(args=None):
    parser = argparse.ArgumentParser(description="Flexible ML model training and prediction")

    # Model type argument (nn, logreg, rf)
    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Model type to use ('nn', 'logreg', 'rf')")

    # Subparser for neural networks
    nn_parser = subparsers.add_parser('nn', help="Neural network model options")
    
        # Neural network-specific arguments - trainer options
    nn_parser.add_argument("--profiler", action="store_true", help="Enable PyTorch Lightning profiler")
    nn_parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs for training")
    nn_parser.add_argument("--default_root_dir", type=str, default="./logs", help="Default root directory for logs")
    nn_parser.add_argument("--devices", type=str, default="auto", help="Devices to use for training")
    nn_parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator to use for training")
    nn_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches")
    nn_parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development run")
    nn_parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit train batches")
    nn_parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit validation batches")
    nn_parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Limit test batches")
    nn_parser.add_argument("--limit_predict_batches", type=float, default=1.0, help="Limit predict batches")
    nn_parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log every n steps")
        # Neural network-specific arguments - callbacks
    nn_parser.add_argument("--callbacks", nargs='+', help="Callbacks to use: (early_stopping, model_checkpoint, lr_monitor)")

        # Subparsers for 'train' and 'predict' under 'nn'
    nn_subparsers = nn_parser.add_subparsers(dest="command", required=True, help="Command to execute ('train' or 'predict')")

    # Neural network-specific training options
    nn_train_parser = nn_subparsers.add_parser("train", help="Train a neural network model")
    nn_train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    nn_train_parser.add_argument("--data_module", type=str, default="BaseDataModule", help="Data module to use")
    # nn_train_parser.add_argument("--data_module_setup", type=str, help="Data module setup file (json) (full path)") # TODO: Implement this   
    #nn_train_parser.add_argument("--profiler", action="store_true", help="Enable PyTorch Lightning profiler")
    #nn_train_parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs for training")
    
    nn_train_parser.add_argument("--model_class", type=str, default='BaseNNModel2', choices=[
        'ResNet50', 'ResNet101', 'ResNet152', 'ResNet_custom_layers', 'BaseNNModel', 'BaseNNModel2'
    ], help="Model class to use")

    # Additional ResNet-specific arguments
    nn_train_parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (ResNet)")
    nn_train_parser.add_argument("--image_channels", type=int, default=1, help="Image channels (ResNet)")
    nn_train_parser.add_argument("--padding_layer_sizes", type=tuple, default=(2, 2, 4, 3, 20, 19), help="Padding layers for ResNet")

    # Check what argument model_class is and add the appropriate arguments
    # To check, we first have to parse the arguments
    nn_train_args, _ = nn_train_parser.parse_known_args()
    if nn_train_args.model_class in ["ResNet_custom_layers"]:
        nn_train_parser.add_argument("--layers", nargs='+', help="Number of layers (for ResNet_custom_layers)")
    elif nn_train_args.model_class == "BaseNNModel2":
        nn_train_parser.add_argument("--layers", type=int, default=3, help="Number of layers (for BaseNNModel2)")
        nn_train_parser.add_argument("--block", type=str, default='testBlock', help="Block type (for BaseNNModel2)")

    # Neural network-specific prediction options
    nn_predict_parser = nn_subparsers.add_parser("predict", help="Predict using a neural network model")
    nn_predict_parser.add_argument("--model_class", type=str, default='BaseNNModel2', choices=[
        'ResNet50', 'ResNet101', 'ResNet152', 'ResNet_custom_layers', 'BaseNNModel', 'BaseNNModel2'
    ], help="Model class to use")
    nn_predict_parser.add_argument("--data_module", type=str, default="BaseDataModule", help="Data module to use")
    #nn_predict_parser.add_argument("--data_module_setup", type=str, help="Data module setup file (json) (full path)") # TODO: Implement this   
    nn_predict_parser.add_argument("--model_file", type=str, required=True, help="Model file for prediction")
    nn_predict_parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
    nn_predict_parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save predictions")
    nn_predict_parser.add_argument("--save_name", type=str, default="Prediction", help="Filename for saved predictions")
    nn_predict_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction") # TODO: Doesnt work when you use a setup file!
    nn_predict_parser.add_argument("--save_type", type=str, choices=['csv', 'pkl'], default='pkl', help="Save predictions as CSV or pickle")

    # Subparser for logistic regression and random forest
    for model_type in ['logreg', 'rf']:
        model_parser = subparsers.add_parser(model_type, help=f"{model_type.upper()} model options")

        # Subparsers for 'train' and 'predict' under 'logreg' or 'rf'
        model_subparsers = model_parser.add_subparsers(dest="command", required=True, help="Command to execute ('train' or 'predict')")

        # Training options for scikit-learn models
        train_parser = model_subparsers.add_parser("train", help=f"Train a {model_type.upper()} model")
        train_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
        train_parser.add_argument("--data", type=str, required=True, help="Data file")
        train_parser.add_argument("--save_dir", type=str, default="./models", help="Model save directory")
        train_parser.add_argument("--target", type=str, default="label", help="Target column for prediction")
        train_parser.add_argument("--class_weight", type=str, choices=["balanced", None], default="balanced", help="Class weight for classification")
        if model_type == "logreg":
            train_parser.add_argument("--save_name", type=str, default='logreg_model', help="Filename for saved model")
            train_parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for logistic regression")
        elif model_type == "rf":
            train_parser.add_argument("--save_name", type=str, default='rf_model', help="Filename for saved model")

        # Prediction options for scikit-learn models
        predict_parser = model_subparsers.add_parser("predict", help=f"Predict using a {model_type.upper()} model")
        predict_parser.add_argument("--model_file", type=str, required=True, help="Model file to use for prediction")
        predict_parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
        predict_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
        predict_parser.add_argument("--data", type=str, required=True, help="Data file")
        predict_parser.add_argument("--save_name", type=str, default="Prediction", help="Filename for saved predictions")
        predict_parser.add_argument("--save_type", type=str, choices=['csv', 'pkl'], default='csv', help="Save predictions as CSV or pickle")
        predict_parser.add_argument("--remove_label", type=bool, default=True, help="Remove 'label' column from prediction data, if it exists")

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    parsed_args = parser.parse_args(args)

    # If Namespace object has no attribute 'max_iter', set it to 1
    if not hasattr(parsed_args, 'max_iter'):
        parsed_args.max_iter = 1

    # Neural network training logic
    if parsed_args.model_type == "nn" and parsed_args.command == "train":
        model_class = parsed_args.model_class

        # ResNet models
        if model_class.startswith("ResNet"):

            if not hasattr(parsed_args, 'layers'):
                layers = None
            else:
                layers = parsed_args.layers
            
            # layers = parsed_args.layers if parsed_args.layers else [1, 1, 1, 1]

            model = get_nn_model(model_class, {
                "num_classes": parsed_args.num_classes,
                "image_channels": parsed_args.image_channels,
                "padding_layer_sizes": parsed_args.padding_layer_sizes,
                "layers": layers
            })
        elif model_class in ["BaseNNModel", "BaseNNModel2"]:
            # BaseNNModel and BaseNNModel2

            if model_class == "BaseNNModel2":
                block = parsed_args.block or 'testBlock' # Default to testBlock for BaseNNModel2
                if block == "testBlock":
                    block = testBlock
                layers = parsed_args.layers or 3 # Default to 3 layers for BaseNNModel2
                model = get_nn_model(model_class, {'layers': layers, 'Block': block})
            elif model_class == "BaseNNModel":
                model = get_nn_model(model_class)

        # Data module setup for neural network
        if parsed_args.data_module == "BaseDataModule":
            data_module = BaseDataModule(data_dir="./data", batch_size=parsed_args.batch_size)
            data_module.setup()

        elif parsed_args.data_module == "CustomDataModule":
            data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
            print("Preparing data")
            data_module.prepare_data()
            print("Setting up data")
            data_module.setup()

        # Instantiate callbacks
        callbacks_ = define_callbacks(parsed_args.callbacks)
        # Instantiate trainer
            # Trainer specific arguments
        profiler = parsed_args.profiler
        max_epochs = parsed_args.max_epochs
        default_root_dir = parsed_args.default_root_dir
        devices = parsed_args.devices
        accelerator = parsed_args.accelerator
        accumulate_grad_batches = parsed_args.accumulate_grad_batches
        fast_dev_run = parsed_args.fast_dev_run
        limit_train_batches = parsed_args.limit_train_batches
        limit_val_batches = parsed_args.limit_val_batches
        limit_test_batches = parsed_args.limit_test_batches
        limit_predict_batches = parsed_args.limit_predict_batches
        log_every_n_steps = parsed_args.log_every_n_steps
            # Trainer instantiation
        trainer = define_trainer(
            trainer_class=L.Trainer,
            profiler=profiler,
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
            devices=devices,
            accelerator=accelerator,
            accumulate_grad_batches=accumulate_grad_batches,
            fast_dev_run=fast_dev_run,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks_
        )

        print("Training with the following configuration:", parsed_args)
        trainer.fit(model, datamodule=data_module)

    # Neural network prediction logic
    elif parsed_args.model_type == "nn" and parsed_args.command == "predict":

        # Check what argument model_class is and add the appropriate arguments
        # To check, we first have to parse the arguments
        nn_predict_args, _ = nn_predict_parser.parse_known_args()
        if nn_predict_args.model_class == "ResNet_custom_layers":
            nn_predict_parser.add_argument("--layers", nargs='+', help="Number of layers (for ResNet_custom_layers)")
        elif nn_predict_args.model_class == "BaseNNModel2":
            nn_predict_parser.add_argument("--layers", type=int, default=3, help="Number of layers (for BaseNNModel2)")
            nn_predict_parser.add_argument("--block", type=str, default='testBlock', help="Block type (for BaseNNModel2)")
        
        nn_predict_args, _ = nn_predict_parser.parse_known_args()
        # Get model class
        model_class = nn_predict_args.model_class
        # Load model
        model = get_nn_model_class(model_class).load_from_checkpoint(os.path.join(nn_predict_args.model_dir, f"{nn_predict_args.model_file}.ckpt"))
        
            # // TODO: Currently, we cannot return class with a different than default block, because we cannot pass the block to the model

        # Load data module for prediction
        if parsed_args.data_module == "BaseDataModule":
            data_module = BaseDataModule(data_dir="./data", batch_size=parsed_args.batch_size)
            data_module.setup()

        elif parsed_args.data_module == "CustomDataModule": # TODO: Allow this to be passed in separate file?
            
            data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
            print("Preparing data")
            data_module.prepare_data()

            print("Setting up data")
            data_module.setup()
        
        # Instantiate callbacks
        callbacks_ = define_callbacks(parsed_args.callbacks)
        # Instantiate trainer
            # Trainer specific arguments
        profiler = parsed_args.profiler
        max_epochs = parsed_args.max_epochs
        default_root_dir = parsed_args.default_root_dir
        devices = parsed_args.devices
        accelerator = parsed_args.accelerator
        accumulate_grad_batches = parsed_args.accumulate_grad_batches
        fast_dev_run = parsed_args.fast_dev_run
        limit_train_batches = parsed_args.limit_train_batches
        limit_val_batches = parsed_args.limit_val_batches
        limit_test_batches = parsed_args.limit_test_batches
        limit_predict_batches = parsed_args.limit_predict_batches
        log_every_n_steps = parsed_args.log_every_n_steps
            # Trainer instantiation
        trainer = define_trainer(
            trainer_class=L.Trainer,
            profiler=profiler,
            max_epochs=max_epochs,
            default_root_dir=default_root_dir,
            devices=devices,
            accelerator=accelerator,
            accumulate_grad_batches=accumulate_grad_batches,
            fast_dev_run=fast_dev_run,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            limit_predict_batches=limit_predict_batches,
            log_every_n_steps=log_every_n_steps,
            callbacks=callbacks_
        )

        # Predict using neural network
        predictions = trainer.predict(model, datamodule=data_module)
        create_dir_if_not_exists(parsed_args.save_dir)

        # Save predictions with the specified save type
        if parsed_args.save_type == 'csv':
            pd.DataFrame(predictions).to_csv(os.path.join(parsed_args.save_dir, f"{parsed_args.save_name}.csv"), index=False)
        elif parsed_args.save_type == 'pkl':
            with open(os.path.join(parsed_args.save_dir, f"{parsed_args.save_name}.pkl"), 'wb') as f:
                pickle.dump(predictions, f)

    # Scikit-learn training logic (Random Forest / Logistic Regression)
    elif parsed_args.model_type in ["rf", "logreg"] and parsed_args.command == "train":
        model = get_model(parsed_args.model_type, class_weight=parsed_args.class_weight, max_iter=parsed_args.max_iter)
        X, y = load_data(parsed_args.data_dir, parsed_args.data, parsed_args.target)
        train(model, X, y, save_name=parsed_args.save_name, save_dir=parsed_args.save_dir)

    # Scikit-learn prediction logic (Random Forest / Logistic Regression)
    elif parsed_args.model_type in ["rf", "logreg"] and parsed_args.command == "predict":
        model_file = os.path.join(parsed_args.model_dir, f"{parsed_args.model_file}.pkl")
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        X = load_predict_data(parsed_args.data_dir, parsed_args.data, remove_label=parsed_args.remove_label)
        predictions = predict(model, X, save_name=parsed_args.save_name, save_type=parsed_args.save_type)
        print(predictions)

if __name__ == "__main__":
    # model = ResNet50(num_classes=2, image_channels=1, padding_layer_sizes=(2,2,4,3,20,19))
    # #print(model)

    # # tensor = torch.randn(2, 1, 34, 164, 174) # Batch, Channel, Depth, Height, Width
    # #output = model(tensor)
    # #print(output.shape)
    # #print(output)

    # data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
    # data_module.prepare_data()
    # data_module.setup()

    # train_loader = data_module.train_dataloader()

    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)
    # print(images.shape)

    # y = model(images)
    # print(y.shape)
    # print(y)

    # from utils.show_slice import show_slice
    # show_slice(images[0, 0, :, :, :])
    
    main()
