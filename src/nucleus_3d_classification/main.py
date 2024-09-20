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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from models.BaseNNModels import BaseNNModel, BaseNNModel2, Block

import argparse
import os
import sys
from datetime import datetime

import pytorch_lightning as pl
import torch

from lightning.pytorch.cli import LightningCLI

from models import model_wrapper
from models.ResNet import ResNet, Block
from models.testmodels import BaseNNModel, BaseNNModel2, testBlock


class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # These will be set in the setup() method
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Generate random synthetic data: 1000 samples, 10 features
        num_samples = 1000
        num_features = 10
        num_classes = 2

        # Create random tensors for features (X) and labels (y)
        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))

        # Wrap data into a TensorDataset
        dataset = TensorDataset(X, y)

        # Split into training, validation, and test datasets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)  # Example: using test dataset for prediction

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

def get_nn_model(model_name: str):
    if model_name == "BaseNNModel":
        return BaseNNModel()
    elif model_name == "BaseNNModel2":
        parser.add_argument("--layers", type=int, default=3, help="Number of layers to use")
        extra_args = parser.parse_args(args)  # Reparse with additional options

        return BaseNNModel2(Block=Block, layers=extra_args.layers) # TODO: Fix this, so we can pass in layers

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

def main(args: List[str] = None):
    parser = argparse.ArgumentParser(description="Flexible ML model training and prediction")
    
    # Universal arguments
    parser.add_argument("command", type=str, choices=['train', 'predict'], help="Command to execute")
    parser.add_argument("--model_type", type=str, default='logreg', help="Model to use (rf, logreg, or nn), default is nn")

    # Parse initial arguments
    if args is None:
        args = sys.argv[1:]

    parsed_args, remaining_args = parser.parse_known_args(args)

    if parsed_args.model_type == 'nn':
        
        parser.add_argument("--batch_size", type=int, default=32, help="Batch size to use")
        parser.add_argument("--data_module", type=str, default="BaseDataModule", help="Data module to use")
        parser.add_argument("--profiler", type=bool, default=False, help="Enable the PyTorch Lightning profiler")
        parser.add_argument("--model_class", type=str, choices=("BaseNNModel", "BaseNNModel2"), help="Name of the model to use")

        if parsed_args.command == 'train':
        # Additional arguments for Lightning models
            parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs to train for")

            parsed_args = parser.parse_args(args)  # Reparse with additional options
            
            model = get_nn_model(parsed_args.model_class)

            # Load the data module
            if parsed_args.data_module == 'BaseDataModule':
                data_module = BaseDataModule(data_dir="./data", batch_size=parsed_args.batch_size)
                data_module.setup()
            else:
                # Implement custom data module loading
                pass

            # Simple profiler
            if parsed_args.profiler:
                profiler = L.profiler.SimpleProfiler()
                trainer = L.Trainer(profiler=profiler, max_epochs=parsed_args.max_epochs)
            else:
                profiler = None
                trainer = L.Trainer(max_epochs=parsed_args.max_epochs)

            print("Parsed arguments for train:", parsed_args)

            trainer.fit(model, datamodule=data_module)

        elif parsed_args.command == 'predict':
            parser.add_argument("--model_file", type=str, required=True, help="Model file to use for prediction")
            parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
            parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save the predictions in")
            parser.add_argument("--save_name", type=str, default='Prediction', help="Name to save the predictions as")

            parsed_args = parser.parse_args(args)  # Reparse with additional options

            # Load the respective model and checkpoint # TODO: Fix this
            # model = BaseModel.load_from_checkpoint(os.path.join(parsed_args.model_dir, f"{parsed_args.model_file}.ckpt"))

            # Load the data module
            if parsed_args.data_module == 'BaseDataModule':
                data_module = BaseDataModule(data_dir="./data", batch_size=parsed_args.batch_size)
                data_module.setup()
            else:
                # Implement custom data module loading
                pass

            # Simple profiler
            if parsed_args.profiler:
                profiler = L.profiler.SimpleProfiler()
                trainer = L.Trainer(profiler=profiler)
            else:
                profiler = None
                trainer = L.Trainer()

            predictions = trainer.predict(model, datamodule=data_module)

            # Save predictions to csv
            create_dir_if_not_exists(parsed_args.save_dir)
            pd.DataFrame(predictions).to_csv(os.path.join(parsed_args.save_dir, f"{parsed_args.save_name}.csv"), index=False)
            
            print(predictions)

    if parsed_args.model_type in ['rf', 'logreg']:
        # Additional arguments for scikit-learn models
        parser.add_argument("--data_dir", type=str, default="./data", help="Data directory to use")
        parser.add_argument("--data", type=str, required=True, help="Data file to use")

        if parsed_args.command == 'train':
            parser.add_argument("--save_dir", type=str, default="./models", help="Directory to save the model in")
            parser.add_argument("--target", type=str, default='label', help="Target column to predict")
            parser.add_argument("--class_weight", type=str, choices=('balanced', None), default='balanced', help="Class weight to use")

            if parsed_args.model_type == 'logreg':
                parser.add_argument("--save_name", type=str, default='Logistic_regression_model', help="Name to save the model as")
                parser.add_argument("--max_iter", type=int, default=1000, help="Maximum number of iterations for Logistic Regression")

            if parsed_args.model_type == 'rf':
                parser.add_argument("--save_name", type=str, default='Random_forest_model', help="Name to save the model as")

            parsed_args = parser.parse_args(args)  # Reparse with additional options
            
            # Handling missing max_iter for RF
            if parsed_args.model_type == 'rf':
                parsed_args.max_iter = 1000

            print("Parsed arguments for train:", parsed_args)

            model = get_model(parsed_args.model_type, class_weight=parsed_args.class_weight, max_iter=parsed_args.max_iter)
            X, y = load_data(parsed_args.data_dir, parsed_args.data, parsed_args.target)
            train(model, X, y, save_name=parsed_args.save_name, save_dir=parsed_args.save_dir)

        if parsed_args.command == 'predict':
            parser.add_argument("--save_name", type=str, default='Prediction', help="Name to save the predictions as")
            parser.add_argument("--model_file", type=str, required=True, help="Model file to use for prediction")
            parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
            parser.add_argument("--save_type", type=str, choices=['csv', 'pkl'], default='csv', help="Type to save the predictions as")
            parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save the predictions in")
            parser.add_argument("--remove_label", type=bool, default=True, help="Remove 'label' column from prediction data")

            parsed_args = parser.parse_args(args)  # Reparse with additional options
            
            print("Parsed arguments for predict:", parsed_args)
            
            # Load the model from the file
            with open(os.path.join(parsed_args.model_dir, f"{parsed_args.model_file}.pkl"), 'rb') as f:
                model = pickle.load(f)
            # Load the prediction data
            X = load_predict_data(parsed_args.data_dir, parsed_args.data, remove_label=parsed_args.remove_label)
            predictions = predict(model, X, save_name=parsed_args.save_name, save_type=parsed_args.save_type)
            print(predictions)

if __name__ == "__main__":
    # model2 = BaseNNModel2(Block=Block)
    # print(model2)

    # x = torch.randn(10)
    # print(model2(x).shape)

    model = ResNet(block=Block, layers=[1,1,1,1], num_classes=2, image_channels=1, padding_layer_sizes=(2,2,4,3,20,19))
    #print(model)

    tensor = torch.randn(2, 1, 34, 164, 174) # Batch, Channel, Depth, Height, Width
    output = model(tensor)
    print(output.shape)
    #print(output)

    # main()