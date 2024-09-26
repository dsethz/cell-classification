"""
Flexible ML Model Argument Parser

This script provides a flexible command-line interface for training and predicting with various machine learning models,
including neural networks (nn), logistic regression (logreg), and random forests (rf).

Usage:
    python script_name.py <model_type> <command> [options]

Model Types:
    - nn: Neural Network
    - logreg: Logistic Regression
    - rf: Random Forest

Commands:
    - train: Train a new model
    - predict: Make predictions using a trained model

Examples:
    1. Train a neural network with custom ResNet layers:
       python script_name.py nn train --model_class ResNet_custom_layers --layers 4 1 2 2

    2. Train a neural network with BaseNNModel2:
       python script_name.py nn train --model_class BaseNNModel2 --layers 3

    3. Train a logistic regression model:
       python script_name.py logreg train --data train_data.csv

    4. Make predictions with a trained random forest model:
       python script_name.py rf predict --model_file rf_model.pkl --data test_data.csv

For more detailed information on available options for each model type and command,
use the --help flag:
    python script_name.py <model_type> <command> --help
"""

import os
import sys
import argparse
import pickle
import pandas as pd
from typing import Optional

import lightning as L
import pytorch_lightning as pl

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from utils.testdatamodule import BaseDataModule
from utils.datamodule import CustomDataModule
from models.testmodels import BaseNNModel, BaseNNModel2, testBlock, get_BaseNNModel, get_BaseNNModel2
from models.ResNet import ResNet50, ResNet101, ResNet152, ResNet_custom_layers

from lightning.pytorch.callbacks import BatchSizeFinder, ModelCheckpoint

import tensorboard
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
        
        if not isinstance(model, BaseEstimator):
            raise ValueError("Model must be a scikit-learn estimator")

        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def create_dir_if_not_exists(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_model(model_name: str, class_weight: Optional[str] = None, max_iter: int = 1000):
    class_weight = None if class_weight == "None" else class_weight
    if model_name == "rf":
        return SklearnModelWrapper(RandomForestClassifier(class_weight=class_weight))
    elif model_name == "logreg":
        return SklearnModelWrapper(LogisticRegression(class_weight=class_weight, max_iter=max_iter))

# NN model instance creation
def get_nn_model(model_name: str, extra_args: dict = None):
    if extra_args is None:
        extra_args = {}

    print(f'{type(extra_args['layers'])}')

    # if hasattr(extra_args, 'block'):
    #     print(f"Block found in extra_args: {extra_args['block']},\n Block type: {type(extra_args['block'])}")
    #     extra_args['block'] = globals()[extra_args['block']]

    print(f"Creating model: {model_name} with extra_args: {extra_args}")

    if model_name == "BaseNNModel":
        return BaseNNModel() # TODO: Check if this is correct, do we have to get model = ...()?
    elif model_name == "BaseNNModel2":
        return BaseNNModel2(layers = extra_args['layers'])
    elif model_name in ["ResNet50", "ResNet101", "ResNet152"]:
        return globals()[model_name](
            ceil_mode=extra_args['ceil_mode'],
            num_classes=extra_args['num_classes'],
            image_channels=extra_args['image_channels'],
            padding_layer_sizes=extra_args['padding_layer_sizes']
        )
    elif model_name == "ResNet_custom_layers":
        return ResNet_custom_layers(
            layers=extra_args['layers'],
            ceil_mode=extra_args['ceil_mode'],
            num_classes=extra_args['num_classes'],
            image_channels=extra_args['image_channels'],
            padding_layer_sizes=extra_args['padding_layer_sizes']
        )

def get_nn_model_class(model_name: str):
    if model_name not in globals():
        raise ValueError(f"Model class {model_name} not found, or input is not valid.\nPlease use one of the following: {list(globals().keys())}") # TODO: Check if this is correct
    return globals()[model_name]

def load_data(data_dir: str, data_file: str, target: str = 'label'):
    data_path = os.path.join(data_dir, data_file)
    if data_file.endswith('.csv'):
        data = pd.read_csv(data_path)
        data = data.dropna(axis=1) # Drop columns with NaN values
    else:
        raise ValueError("Unsupported file format. Please use CSV.")
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in data.")
    y = data[target]
    X = data.drop(columns=[target])
    return X, y

def load_predict_data(data_dir: str, data_file: str, remove_label: bool = True):
    data_path = os.path.join(data_dir, data_file)
    if not data_file.endswith('.csv'):
        raise ValueError("Prediction data must be a CSV file")

    data = pd.read_csv(data_path)
    data = data.dropna(axis=1)

    if remove_label and 'label' in data.columns:
        print("Warning: 'label' column found in prediction data. Removing it by default.\nIf you want to keep the 'label' column, use the --remove_label flag.")
        data = data.drop(columns=['label'])
    
    return data

def train_sklearn_model(model, X, y, save_name: str, save_dir: str = "./models"):
    if not isinstance(model, SklearnModelWrapper):
        raise ValueError("Model must be a SklearnModelWrapper instance")
    model.fit(X, y)
    if save_name:
        create_dir_if_not_exists(save_dir)
        with open(os.path.join(save_dir, f"{save_name}.pkl"), 'wb') as f:
            pickle.dump(model, f)

def predict_sklearn_model(model, X, save_name: str, save_dir: str = "./predictions", save_type: str = 'csv'):
    if not isinstance(model, SklearnModelWrapper):
        raise ValueError("Model must be a SklearnModelWrapper instance")
    y = model.predict(X)
    create_dir_if_not_exists(save_dir)
    save_path = os.path.join(save_dir, f"{save_name}.{save_type}")
    
    if save_type == 'csv':
        pd.DataFrame(y).to_csv(save_path, index=False)
    elif save_type == 'pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(y, f)
            
    print(f"Predictions saved to {save_path} as {save_type}")
    
    return y

def define_callbacks(args, callback_names: list):

    # Making the checkpointing callback here:


    callbacks = []
    for callback_name in callback_names if callback_names is not None else []:
        if callback_name == "early_stopping":
            callbacks.append(pl.callbacks.EarlyStopping(monitor='val_loss'))
        elif callback_name == "model_checkpoint":
            callbacks.append(pl.callbacks.ModelCheckpoint(monitor='val_loss'))
        elif callback_name == "lr_monitor":
            callbacks.append(pl.callbacks.LearningRateMonitor())
        elif callback_name == "BatchSizeFinder":
            callbacks.append(FineTuneBatchSizeFinder())
    
    checkpoint_callback = create_checkpoint_callback(args)
    callbacks.append(checkpoint_callback)

    return callbacks

def create_checkpoint_callback(args):
    args.filename = replace_filename(args)

    # Dirpath -> check if exists, if not, create it
    create_dir_if_not_exists(args.dirpath)

    return ModelCheckpoint(
        save_top_k=args.save_top_k,
        monitor=args.monitor,
        mode=args.mode,
        dirpath=args.dirpath,
        filename=args.filename # Default is "model_name_data_module_name-{epoch:02d}-{val_loss:.2f}"
    )

def replace_filename(args):
    filename = args.filename
    filename = filename.replace("model_name", args.model_class)
    filename = filename.replace("data_module_name", args.data_module)
    filename = filename.replace("epoch", "{epoch:02d}")
    filename = filename.replace("val_loss", "{val_loss:.2f}")
    return filename

class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in [0, 1, 2]:
            self.scale_batch_size(trainer, pl_module)

def define_trainer(args, callbacks=None):
    trainer_kwargs = {
        'profiler': args.profiler,
        'max_epochs': args.max_epochs,
        'default_root_dir': args.default_root_dir,
        'devices': args.devices,
        'accelerator': args.accelerator,
        'accumulate_grad_batches': args.accumulate_grad_batches,
        'fast_dev_run': args.fast_dev_run,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'limit_test_batches': args.limit_test_batches,
        'limit_predict_batches': args.limit_predict_batches,
        'log_every_n_steps': args.log_every_n_steps,
        'sync_batchnorm': args.sync_batchnorm,
        'enable_checkpointing': args.enable_checkpointing,
        'callbacks': callbacks
    }
    print(f"Trainer kwargs: {trainer_kwargs}")
    return L.Trainer(**trainer_kwargs)

def load_data_module(args):
    if args.data_module == "BaseDataModule":
        data_module = BaseDataModule(data_dir="./data", batch_size=args.batch_size)
    elif args.data_module == "CustomDataModule":
        data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json')
    else:
        raise ValueError(f"Unknown data module: {args.data_module}")
    
    data_module.prepare_data()
    data_module.setup()
    return data_module

def train_nn_model(args, model):
    
    data_module = load_data_module(args)
    callbacks = define_callbacks(args, args.callbacks)
    trainer = define_trainer(args, callbacks)
    
    print("Training with the following configuration:", args)
    trainer.fit(model, datamodule=data_module)

def predict_nn_model(args):
    model_class = get_nn_model_class(args.model_class)
    try:
        print(f"Loading model from {args.model_file}.ckpt")
        model = model_class.load_from_checkpoint(f"{args.model_file}.ckpt")
        print(f"Loaded model")

    except FileNotFoundError:
        print(f"Loading model from {os.path.join(args.model_dir, f'{args.model_file}.ckpt')}")
        model = model_class.load_from_checkpoint(os.path.join(args.model_dir, f"{args.model_file}.ckpt"))
        print(f"Loaded model")
    
    data_module = load_data_module(args)
    trainer = define_trainer(args)

    model.eval()
    model.freeze()

    if args.stage == "predict":
        predictions = trainer.predict(model, datamodule=data_module)
    elif args.stage == "test":
        predictions = trainer.test(model, datamodule=data_module)
    elif args.stage == "validate":
        predictions = trainer.test(model, datamodule=data_module)

    save_predictions(predictions, args)
    print("Predictions complete")

def save_predictions(predictions, args):
    create_dir_if_not_exists(args.save_dir)
    save_path = os.path.join(args.save_dir, f"{args.save_name}.{args.save_type}")

    if args.save_type == 'csv':
        pd.DataFrame(predictions).to_csv(save_path, index=False)
    elif args.save_type == 'pkl':
        with open(save_path, 'wb') as f:
            pickle.dump(predictions, f)
    
    print(f"Predictions saved to {save_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Flexible ML model training and prediction")
    
    # First level: model type
    subparsers = parser.add_subparsers(dest="model_type", required=True, help="Model type to use ('nn', 'logreg', 'rf')")

    # Neural Network parser
    nn_parser = subparsers.add_parser('nn', help="Neural network model options")
    nn_subparsers = nn_parser.add_subparsers(dest="command", required=True, help="Command to execute ('train' or 'predict')")

    # Common NN arguments
    nn_common_parser = argparse.ArgumentParser(add_help=False)
    nn_common_parser.add_argument("--profiler", action="store_true", help="Enable PyTorch Lightning profiler")
    nn_common_parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs for training")
    nn_common_parser.add_argument("--default_root_dir", type=str, default="./logs", help="Default root directory for logs")
    nn_common_parser.add_argument("--devices", type=str, default="auto", help="Devices to use for training")
    nn_common_parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator to use for training")
    nn_common_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches")
    nn_common_parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development run")
    nn_common_parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit train batches")
    nn_common_parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit validation batches")
    nn_common_parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Limit test batches")
    nn_common_parser.add_argument("--limit_predict_batches", type=float, default=1.0, help="Limit predict batches")
    nn_common_parser.add_argument("--log_every_n_steps", type=int, default=5, help="Log every n steps")
    nn_common_parser.add_argument("--callbacks", nargs='+', help="Callbacks to use: (early_stopping, model_checkpoint, lr_monitor)")
    nn_common_parser.add_argument("--data_module", type=str, default="BaseDataModule", help="Data module to use")
    nn_common_parser.add_argument("--model_class", type=str, required=True, choices=['ResNet50', 'ResNet101', 'ResNet152', 'ResNet_custom_layers', 'BaseNNModel', 'BaseNNModel2'], help="Model class to use")
    # nn_common_parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (ResNet)")
    # nn_common_parser.add_argument("--image_channels", type=int, default=1, help="Image channels (ResNet)")
    # nn_common_parser.add_argument("--padding_layer_sizes", type=tuple, default=(2, 2, 4, 3, 7, 7), help="Padding layers for ResNet")
    # nn_common_parser.add_argument("--ceil_mode", action="store_false", help="Ceil mode for ResNet on the maxpool layer, default is True")
    # nn_common_parser.add_argument("--layers", nargs='+', type=int, help="Number of layers for custom models or ResNet_custom_layers")
    nn_common_parser.add_argument("--sync_batchnorm", action="store_true", help="Synchronize batch normalization layers")
    nn_common_parser.add_argument("--enable_checkpointing", action="store_true", help="Enable model checkpointing")
    nn_common_parser.add_argument("--save_top_k", type=int, default=1, help="Save top k models")
    nn_common_parser.add_argument("--monitor", type=str, default="val_loss", help="Monitor metric for model checkpointing")
    nn_common_parser.add_argument("--mode", type=str, default="min", help="Mode for model checkpointing ('min' or 'max')")
    nn_common_parser.add_argument("--dirpath", type=str, default="./models/ckpt", help="Directory to save models using checkpointing")
    nn_common_parser.add_argument("--filename", type=str, default="model_name_data_module_name_epoch_val_loss", help="Filename for saved models, best to not change!")


    # NN Train parser
    nn_train_parser = nn_subparsers.add_parser("train", parents=[nn_common_parser], help="Train a neural network model")
    nn_train_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    nn_train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

    nn_train_parser.add_argument("--layers", nargs='+', type=int, help="Number of layers for custom models or ResNet_custom_layers")
    nn_train_parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (ResNet)")
    nn_train_parser.add_argument("--image_channels", type=int, default=1, help="Image channels (ResNet)")
    nn_train_parser.add_argument("--padding_layer_sizes", type=tuple, default=(2, 2, 4, 3, 7, 7), help="Padding layers for ResNet")
    nn_train_parser.add_argument("--ceil_mode", action="store_false", help="Ceil mode for ResNet on the maxpool layer, default is True")

    # NN Predict parser
    nn_predict_parser = nn_subparsers.add_parser("predict", parents=[nn_common_parser], help="Predict using a neural network model")
    #nn_predict_parser.add_argument("--stage", type=str, default="predict", choices=["predict", "val", "test"], help="DataModule stage to use for prediction")
    nn_predict_parser.add_argument("--batch_size", type=int, default=1, help="Batch size for prediction")
    nn_predict_parser.add_argument("--model_file", type=str, required=True, help="Model file for prediction")
    nn_predict_parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
    nn_predict_parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save predictions")
    nn_predict_parser.add_argument("--save_name", type=str, default="Prediction", help="Filename for saved predictions")
    nn_predict_parser.add_argument("--save_type", type=str, choices=['csv', 'pkl'], default='pkl', help="Save predictions as CSV or pickle")
    nn_predict_parser.add_argument("--stage", type=str, default="predict", choices={"test", "predict", "validate"}, help="Which dataloader to use (as defined in datamodule)?")
    
    # Scikit-learn parsers (logreg and rf)
    for model_type in ['logreg', 'rf']:
        model_parser = subparsers.add_parser(model_type, help=f"{model_type.upper()} model options")
        model_subparsers = model_parser.add_subparsers(dest="command", required=True, help="Command to execute ('train' or 'predict')")
        
        # Train parser
        train_parser = model_subparsers.add_parser("train", help=f"Train a {model_type.upper()} model")
        train_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
        train_parser.add_argument("--data", type=str, required=True, help="Data file")
        train_parser.add_argument("--save_dir", type=str, default="./models", help="Model save directory")
        train_parser.add_argument("--target", type=str, default="label", help="Target column for prediction")
        train_parser.add_argument("--class_weight", type=str, choices=["balanced", "None"], default="balanced", help="Class weight for classification")
        train_parser.add_argument("--save_name", type=str, default=f'{model_type}_model', help="Filename for saved model")
        
        if model_type == "logreg":
            train_parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for logistic regression")
        
        # Predict parser
        predict_parser = model_subparsers.add_parser("predict", help=f"Predict using a {model_type.upper()} model")
        predict_parser.add_argument("--model_file", type=str, required=True, help="Model file to use for prediction")
        predict_parser.add_argument("--model_dir", type=str, default="./models", help="Directory to load the model from")
        predict_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
        predict_parser.add_argument("--data", type=str, required=True, help="Data file")
        predict_parser.add_argument("--target", type=str, default="label", help="Target column for prediction")
        predict_parser.add_argument("--save_name", type=str, default="Prediction", help="Filename for saved predictions")
        predict_parser.add_argument("--save_type", type=str, choices=['csv', 'pkl'], default='pkl', help="Save predictions as CSV or pickle")
        predict_parser.add_argument("--remove_label", action="store_false", help="Remove 'label' column from prediction data")
        predict_parser.add_argument("--save_dir", type=str, default="./predictions", help="Directory to save predictions")

    args = parser.parse_args()
    return args

def process_layers_argument(args):
    """
    Process the --layers argument based on the model_class.
    For ResNet_custom_layers, interpret it as [layer1, layer2, layer3, layer4].
    For BaseNNModel2, interpret it as a single integer for the number of layers.
    """
    if args.model_class == 'ResNet_custom_layers':
        if args.layers is None or len(args.layers) != 4:
            raise ValueError("ResNet_custom_layers requires exactly 4 layer values")
        args.layers = args.layers  # Keep as a list of 4 integers
    elif args.model_class == 'BaseNNModel2':
        if args.layers is None or len(args.layers) != 1:
            raise ValueError("BaseNNModel2 requires a single integer for the number of layers")
        args.layers = args.layers[0]  # Convert to a single integer
    return args

def get_extra_args(args):
    extra_args = {
                'ceil_mode': args.ceil_mode,
                'num_classes': args.num_classes,
                'image_channels': args.image_channels,
                'padding_layer_sizes': args.padding_layer_sizes,
                'layers': args.layers,
                'padding_layer_sizes': args.padding_layer_sizes
                }
    return extra_args

def main():
    args = parse_arguments()
    
    # NN Branch
    if args.model_type == "nn":
        if args.command == "train":
            if args.model_class in ['ResNet_custom_layers', 'BaseNNModel2']:
                args = process_layers_argument(args)
            extra_args = get_extra_args(args)

            train_nn_model(args, model=get_nn_model(args.model_class, extra_args))
        elif args.command == "predict":
            predict_nn_model(args)

    # SKlearn Branch
    elif args.model_type in ["logreg", "rf"]:

        # Train sub-branch
        if args.command == "train":
            # List of arguments that should be checked
            required_args = ['max_iter', 'class_weight']
            # Set any missing arguments to None
            for arg in required_args:
                if not hasattr(args, arg) or getattr(args, arg) is None:
                    setattr(args, arg, None)

            model = get_model(args.model_type, class_weight=args.class_weight, max_iter=args.max_iter)
            X, y = load_data(args.data_dir, args.data, target=args.target)
            train_sklearn_model(model, X, y, args.save_name, save_dir=args.save_dir)
        
        # Predict sub-branch
        elif args.command == "predict":
            model_path = os.path.join(args.model_dir, f"{args.model_file}.pkl")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            X_pred = load_predict_data(args.data_dir, args.data, remove_label=args.remove_label)
            predict_sklearn_model(model, X_pred, args.save_name, save_dir=args.save_dir, save_type=args.save_type)

if __name__ == "__main__":
    main()
