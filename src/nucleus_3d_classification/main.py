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

For NN models, the following options are available:
    - --model_class: Model class to use (ResNet50, ResNet101, ResNet152, ResNet_custom_layers, BaseNNModel, BaseNNModel2)
    - --num_classes: Number of classes to predict (ResNet)
    - --image_channels: Image channels (ResNet)
    - --padding_layer_sizes: Padding layers for ResNet
    - --ceil_mode: Ceil mode for ResNet on the maxpool layer
    - --layers: Number of layers for custom models or ResNet_custom_layers
    - --sync_batchnorm: Synchronize batch normalization layers
    - --enable_checkpointing: Enable model checkpointing
    - --save_top_k: Save top k models
    - --monitor: Monitor metric for model checkpointing
    - --monitor_stop: Monitor metric for early stopping
    - --mode: Mode for model checkpointing ('min' or 'max')
    - --dirpath: Directory to save models using checkpointing
    - --filename: Filename for saved models
    - --strategy: Training strategy (ddp, ddp_spawn, deepspeed, etc.)
    - --loss_fn: Loss function to use (cross_entropy, bce, mse)
    - --loss_weight: Class weight for classification
    - --num_workers: Number of workers for dataloader
    - --batch_size: Batch size for dataloader
    - --swa_args: StochasticWeightAveraging arguments, example: --swa_args swa_lrs=3e-4 swa_epoch_start=15 annealing_epochs=10 annealing_strategy=cos
    - --callbacks: Callbacks to use: (early_stopping, model_checkpoint, lr_monitor), example: --callbacks early_stopping model_checkpoint lr_monitor
    - --data_module: Data module to use (BaseDataModule (testing purposes), CustomDataModule (for custom data loading))
    - --setup_file: Setup file for CustomDataModule
    - --profiler: Enable PyTorch Lightning profiler, choices are simple, advanced
    - --enable_progress_bar: Enables the progress bar
    - --max_epochs: Max epochs for training
    - --default_root_dir: Default root directory for logs
    - --devices: Devices to use for training, expects int, default is auto
    - --accelerator: Accelerator to use for training
    - --accumulate_grad_batches: Accumulate gradient batches
    - --fast_dev_run: Run a fast development run
    - --limit_train_batches: Limit train batches
    - --limit_val_batches: Limit validation batches
    - --limit_test_batches: Limit test batches
    - --limit_predict_batches: Limit predict batches
    - --log_every_n_steps: Log every n steps


# SWA args (--swa_args) example:
--swa_args swa_lrs=3e-4 swa_epoch_start=15 annealing_epochs=10 annealing_strategy=cos
--swa_args swa_lrs=float swa_epoch_start=int annealing_epochs=int annealing_strategy=str

# LRF args (--lrf_args) example:    
--lrf_args min_lr=1e-6 max_lr=1e-1 num_training_steps=100 mode=cos milestones=[0, 1]
--lrf_args min_lr=float max_lr=float num_training_steps=int mode=str milestones=[int, int, int]
"""

import os
import sys
import argparse
import re
import pickle
import pandas as pd
from typing import Optional

import lightning as L
import pytorch_lightning as pl

# Debugging high mem usage
from torch.profiler import profile, record_function, ProfilerActivity
import socket
import logging
from datetime import datetime, timedelta
import torch

from lightning.pytorch import seed_everything
seed_everything(42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator

from utils.testdatamodule import BaseDataModule
from utils.datamodule import CustomDataModule
from utils.loss_fn import create_loss_fn_with_weights, calculate_class_weights
from models.testmodels import BaseNNModel, BaseNNModel2, testBlock, get_BaseNNModel, get_BaseNNModel2
from models.ResNet import ResNet50, ResNet101, ResNet152, ResNet_custom_layers, testNet

from lightning.pytorch.callbacks import BatchSizeFinder, ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateFinder, StochasticWeightAveraging
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

# import tensorboard
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

    print(f"Creating model: {model_name} with extra_args: {extra_args}")
    print(f"extra_args.loss_fn: {extra_args.get('loss_fn', None)}")

    if model_name == "BaseNNModel":
        return BaseNNModel()
    elif model_name == "BaseNNModel2":
        return BaseNNModel2(layers = extra_args['layers'])
    elif model_name in ["ResNet50", "ResNet101", "ResNet152", "testNet"]:
        return globals()[model_name](
            ceil_mode=extra_args['ceil_mode'],
            num_classes=extra_args['num_classes'],
            image_channels=extra_args['image_channels'],
            padding_layer_sizes=extra_args['padding_layer_sizes'],
            learning_rate=extra_args['learning_rate']
        )
    elif model_name == "ResNet_custom_layers":
        return ResNet_custom_layers(
            layers=extra_args['layers'],
            ceil_mode=extra_args['ceil_mode'],
            num_classes=extra_args['num_classes'],
            image_channels=extra_args['image_channels'],
            padding_layer_sizes=extra_args['padding_layer_sizes'],
            learning_rate=extra_args['learning_rate']
        )

def get_nn_model_class(model_name: str):
    if model_name not in globals():
        raise ValueError(f"Model class {model_name} not found, or input is not valid.\nPlease use one of the following: {list(globals().keys())}") # TODO: Check if this is correct
    # Return Class object, not instance
    model = globals()[model_name]

    # For some imports we have functions, not classes, thus we need to call the models in order to get the class object
    if callable(model) and model_name in ["ResNet50", "ResNet_custom_layers", "ResNet101", "ResNet152", "testNet"]:
        try:
            model = model()
        except TypeError:
            num_classes = 2
            image_channels = 1
            padding_layer_sizes = (2, 2, 4, 3, 7, 7)
            ceil_mode = True
            model = model(num_classes=num_classes, image_channels=image_channels, padding_layer_sizes=padding_layer_sizes, ceil_mode=ceil_mode)
        return model.__class__
    
    return model

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

def extract_arguments(text):

    if isinstance(text, list):
        # Join the list into a single string
        text = " ".join(text)

    # Regular expression to match key=value pairs
    pattern = r"(\w+)=([0-9e\-.]+|\w+|\[[0-9, ]+\])"
    matches = re.findall(pattern, text)
    
    # Convert matches to a dictionary
    arguments = {key: value for key, value in matches}
    
    return arguments

def convert_types(arguments):
    for key, value in arguments.items():
        if value.startswith('[') and value.endswith(']'):
            # Convert string list to actual list
            try:
                arguments[key] = list(map(int, value[1:-1].split(','))) 
                # Convert to list of integers, assuming all values are integers, otherwise print error
            except ValueError:
                print(f"Error converting {key} to list of integers, with values: {value}")
        elif 'e' in value or '.' in value:
            # Convert to float
            try:
                arguments[key] = float(value)
            except ValueError:
                pass
                    # Keep it as a string if conversion fails, exaple 'linear', has e in it
        else:
            # Convert to int if it's an integer
            try:
                arguments[key] = int(value)
            except ValueError:
                pass  # Keep it as a string if conversion fails, example 'linear', as it is not an integer
    return arguments

def define_callbacks(args, callback_names: list):

    callbacks = []
    for callback_name in callback_names if callback_names is not None else []:
        if callback_name == "early_stopping":
            if not hasattr(args, 'monitor_stop'):
                args.monitor_stop = "val_loss"
            callbacks.append(pl.callbacks.EarlyStopping(monitor=args.monitor_stop))
        # elif callback_name == "model_checkpoint":
        #     callbacks.append(pl.callbacks.ModelCheckpoint(monitor='val_loss'))
        elif callback_name == "lr_monitor":
            callbacks.append(pl.callbacks.LearningRateMonitor())
        elif callback_name == "BatchSizeFinder":
            callbacks.append(FineTuneBatchSizeFinder())

        elif callback_name == "LearningRateFinder":
            print("Adding LearningRateFinder callback")

            if args.lrf_args is not None:
                try:
                    args.lrf_args = convert_types(extract_arguments(args.lrf_args))
                    min_lr = args.lrf_args.get('min_lr', 1e-6)
                    max_lr = args.lrf_args.get('max_lr', 0.1)
                    num_training_steps = args.lrf_args.get('num_training_steps', 100)
                    mode = args.lrf_args.get('mode', 'cos')
                    milestones = args.lrf_args.get('milestones', [0, 1])


                except (ValueError, IndexError) as e:
                    print(f"Error parsing lrf_args: {e}.")
                    min_lr = 1e-6
                    max_lr = 1e-1
                    num_training_steps = 100
                    mode = 'cos'
                    milestones = [0, 1]
            else:
                # If no lrf_args, use default LRF parameters
                print("LNo LRF arguments provided.")
                min_lr = 1e-6
                max_lr = 1e-1
                num_training_steps = 100
                mode = 'cos'
                milestones = [0, 1]
            
            print(f"Using LRF parameters: "
                    f"min_lr={min_lr}, "
                    f"max_lr={max_lr}, "
                    f"num_training_steps={num_training_steps}, "
                    f"mode={mode}, "
                    f"milestones={milestones}")

            callbacks.append(FineTuneLearningRateFinder(min_lr=min_lr, max_lr=max_lr, num_training_steps=num_training_steps, mode=mode, milestones=milestones))


        elif callback_name == "StochasticWeightAveraging":
            print("Adding StochasticWeightAveraging callback")

            # Check if checkpointing is enabled
            if not args.enable_checkpointing:
                print("Warning: StochasticWeightAveraging requires model checkpointing to be enabled. Disabling SWA.")
            else:
                try:
                    args.swa_args = convert_types(extract_arguments(args.swa_args))
                    swa_lrs = args.swa_args.get('swa_lrs', 1e-4)
                    swa_epoch_start = args.swa_args.get('swa_epoch_start', 10)
                    annealing_epochs = args.swa_args.get('annealing_epochs', 10)
                    annealing_strategy = args.swa_args.get('annealing_strategy', 'cos')
                except (ValueError, IndexError) as e:
                    print(f"Error parsing swa_args: {e}.")
                    swa_lrs = 1e-4
                    swa_epoch_start = 10
                    annealing_epochs = 10
                    annealing_strategy = 'cos'
                
                # Log the chosen SWA parameters
                print(f"Using SWA parameters: "
                      f"swa_lrs={swa_lrs}, "
                      f"swa_epoch_start={swa_epoch_start}, "
                      f"annealing_epochs={annealing_epochs}, "
                      f"annealing_strategy={annealing_strategy}")
                
                # Append the SWA callback with the parsed (or default) parameters
                callbacks.append(StochasticWeightAveraging(
                    swa_lrs=swa_lrs, 
                    swa_epoch_start=swa_epoch_start, 
                    annealing_epochs=annealing_epochs, 
                    annealing_strategy=annealing_strategy
                ))

    if args.enable_checkpointing:
        checkpoint_callback = create_checkpoint_callback(args)
        callbacks.append(checkpoint_callback)

    return callbacks

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones=0, min_lr:float=1e-6, max_lr: float= 0.1, num_training_steps: int = 100, mode: str ='cos', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_training_steps = num_training_steps
        self.mode = mode

        # Convert int to list if only one milestone is provided
        if isinstance(self.milestones, int):
            self.milestones = [self.milestones]

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

def create_checkpoint_callback(args):
    args.filename = replace_filename(args)

    # Dirpath -> check if exists, if not, create it
    create_dir_if_not_exists(args.dirpath)

    return [
    ModelCheckpoint(
        save_top_k=args.save_top_k,
        save_last=True, # Always saves last.ckpt
        save_on_train_epoch_end=False, # Save at end of validation
        monitor=args.monitor,
        mode=args.mode,
        dirpath=args.dirpath,
        filename=args.filename # Default is "model_name_data_module_name-{epoch:02d}-{val_loss:.2f}"
        ),
    ModelCheckpoint(
        save_top_k=1,
        save_on_train_epoch_end=False,
        monitor='f1_score_val',
        mode=args.mode,
        dirpath=args.dirpath,
        filename='best-f1_score-{epoch:02d}-{f1_score_val:.2f}'
        )
    ]

def replace_filename(args):
    filename = args.filename
    filename = filename.replace("model_name", args.model_class)
    filename = filename.replace("data_module_name", args.data_module)
    filename = filename.replace("epoch", "{epoch:02d}")
    filename = filename.replace("val_loss", "{val_loss:.2f}")

    filename = clean_filename(filename)
    
    return filename

def clean_filename(filename):
    # Set of special characters to be replaced with an underscore
    #special_chars = {'/', '\\', ':', '*', '?', '"', '<', '>', '|', "'", '!', '@', '#', '$', '%', '^', '&', '(', ')', '+', '=', '{', '}', '[', ']', ';', ',', '.', '`', '~', ' '}
    special_chars = {"'", '"', '/', '\\', ' '}
    # Use regex to replace all special characters in the set with an underscore
    return re.sub(r'[{}]'.format(re.escape(''.join(special_chars))), '_', filename)


class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in [0, 1]:
            self.scale_batch_size(trainer, pl_module)

def define_trainer(args, callbacks=None):
    trainer_kwargs = {
        'profiler': args.profiler,
        'enable_progress_bar': args.enable_progress_bar,
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
        'strategy': args.strategy,
        'gradient_clip_val': args.gradient_clip_val,
        'callbacks': callbacks,
        'deterministic': False # TODO DISABLE LATER
    }
    print(f"Trainer kwargs: {trainer_kwargs}")
    return L.Trainer(**trainer_kwargs)

def load_data_module(args):
    if args.data_module == "BaseDataModule":
        if not hasattr(args, 'batch_size') or (hasattr(args, 'batch_size') and args.batch_size is None):
            print("No batch size provided, using default batch size of 32")
            data_module = BaseDataModule(data_dir="./data", batch_size=32)
        else:
            data_module = BaseDataModule(data_dir="./data", batch_size=args.batch_size)
    
    elif args.data_module == "CustomDataModule":
        if not hasattr(args, 'setup_file') or hasattr(args, 'setup_file') and args.setup_file is None: #TODO: This should later be able to be superseeded by providing individual directories for all files as well
            try:
                setup_file = '/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json'
            except FileNotFoundError: # TODO remove this logic later
                pass
        else:
            setup_file = args.setup_file

        if args.batch_size is not None:
            try:
                args.batch_size = int(args.batch_size)
            except ValueError:
                print(f"Error converting batch_size to integer: {args.batch_size}, using default value of None")
                args.batch_size = None
        if args.num_workers is not None:
            try:
                args.num_workers = int(args.num_workers)
            except ValueError:
                print(f"Error converting num_workers to integer: {args.num_workers}, using default value of None")
                args.num_workers = None

        data_module = CustomDataModule(setup_file=setup_file, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise ValueError(f"Unknown data module: {args.data_module}")
    
    data_module.setup() # TODO: Add stage argument
    return data_module

def train_nn_model(args):
    data_module = load_data_module(args)
    #print(f"train loader is type {type(data_module.train_dataloader())}")
    loss_fn = create_loss_fn_with_weights(data_module.train_dataloader(), loss_fn=args.loss_fn, weight=args.loss_weight)
    extra_args = get_extra_args(args, loss_fn=loss_fn)
    model = get_nn_model(args.model_class, extra_args)

    callbacks = define_callbacks(args, args.callbacks)
    trainer = define_trainer(args, callbacks)

    print("Training with the following configuration:", args)

    # Profiling memory
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         trainer.fit(model, datamodule=data_module)
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
    # print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))
    # print(prof.key_averages().table(sort_by='self_cuda_memory_usage', row_limit=30))

    # Same with other logger:
    # Start recording memory snapshot history
    start_record_memory_history()
    trainer.fit(model, datamodule=data_module)

   # Create the memory snapshot file
    export_memory_snapshot()

   # Stop recording memory snapshot history
    stop_record_memory_history()

#     with torch.profiler.profile(
#        activities=[
#            torch.profiler.ProfilerActivity.CPU,
#            torch.profiler.ProfilerActivity.CUDA,
#        ],
#        schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
#        record_shapes=True,
#        profile_memory=True,
#        on_trace_ready=trace_handler,
#        with_stack=True # Gives some error
#    ) as prof:
#         trainer.fit(model, datamodule=data_module)


###### MEM USAGE ######
logging.basicConfig(
   format="%(levelname)s:%(asctime)s %(message)s",
   level=logging.INFO,
   datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000

def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )

def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not recording memory history")
       return

   logger.info("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)

def export_memory_snapshot() -> None:
   if not torch.cuda.is_available():
       logger.info("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"

   try:
       logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       logger.error(f"Failed to capture memory snapshot {e}")
       return
######################

#################### MEMORY PROFILING ####################
# logging.basicConfig(
#    format="%(levelname)s:%(asctime)s %(message)s",
#    level=logging.INFO,
#    datefmt="%Y-%m-%d %H:%M:%S",
# )
# logger: logging.Logger = logging.getLogger(__name__)
# logger.setLevel(level=logging.INFO)

# TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# def trace_handler(prof: torch.profiler.profile):
#    # Prefix for file names.
#    host_name = socket.gethostname()
#    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
#    file_prefix = f"{host_name}_{timestamp}"

#    # Construct the trace file.
#    prof.export_chrome_trace(f"{file_prefix}.json.gz")

#    # Construct the memory timeline file.
#    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
#################### MEMORY PROFILING ####################

def predict_nn_model(args):
    model_class = get_nn_model_class(args.model_class)

    # Define possible paths and file names to attempt loading
    model_paths = [
        f"{args.model_file}",
        f"{args.model_file}.ckpt",
        os.path.join(args.model_dir, f"{args.model_file}"),
        os.path.join(args.model_dir, f"{args.model_file}.ckpt")
    ]
    
    model = None

    # Attempt to load the model from each path
    for model_path in model_paths:
        try:
            print(f"Attempting to load model from {model_path}")
            model = model_class.load_from_checkpoint(model_path)
            print(f"Successfully loaded model from {model_path}")
            break
        except FileNotFoundError:
            print(f"Model file not found at {model_path}, trying next option...")
    
    # If none of the paths worked, raise an error
    if model is None:
        raise FileNotFoundError(f"Model file not found. Tried the following paths: {model_paths}")

    data_module = load_data_module(args)
    trainer = define_trainer(args)

    model.eval()
    model.freeze()

    if args.stage == "predict":
        print("Predicting - this is not implemented yet for CustomDataModule!")
        predictions = trainer.predict(model, datamodule=data_module) # TODO: Not implemented yet for custom datamodule.
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
    nn_common_parser.add_argument("--profiler", choices=[None,"simple", "advanced"], default=None, help="Enable PyTorch Lightning profiler, choices are simple, advanced")
    nn_common_parser.add_argument("--enable_progress_bar", action="store_true", help="Enables the progress bar")
    nn_common_parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs for training")
    nn_common_parser.add_argument("--default_root_dir", type=str, default="./logs", help="Default root directory for logs")
    nn_common_parser.add_argument("--devices", default='auto', help="Devices to use for training, expects int, default is auto")
    nn_common_parser.add_argument("--accelerator", type=str, default="auto", choices=['cpu', 'gpu', 'tpu'], help="Accelerator to use for training")
    nn_common_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradient batches")
    nn_common_parser.add_argument("--fast_dev_run", action="store_true", help="Run a fast development run")
    nn_common_parser.add_argument("--limit_train_batches", type=float, default=1.0, help="Limit train batches")
    nn_common_parser.add_argument("--limit_val_batches", type=float, default=1.0, help="Limit validation batches")
    nn_common_parser.add_argument("--limit_test_batches", type=float, default=1.0, help="Limit test batches")
    nn_common_parser.add_argument("--limit_predict_batches", type=float, default=1.0, help="Limit predict batches")
    nn_common_parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log every n steps")
    nn_common_parser.add_argument("--callbacks", nargs='+', help="Callbacks to use: (early_stopping, model_checkpoint, lr_monitor)")
    nn_common_parser.add_argument("--data_module", type=str, default="BaseDataModule", help="Data module to use")
    nn_common_parser.add_argument("--setup_file", type=str, help="Setup file for CustomDataModule")
    nn_common_parser.add_argument("--model_class", type=str, required=True, choices=['ResNet50', 'ResNet101', 'ResNet152', 'ResNet_custom_layers', 'BaseNNModel', 'BaseNNModel2', 'testNet'], help="Model class to use")
    # nn_common_parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (ResNet)")
    # nn_common_parser.add_argument("--image_channels", type=int, default=1, help="Image channels (ResNet)")
    # nn_common_parser.add_argument("--padding_layer_sizes", type=tuple, default=(2, 2, 4, 3, 7, 7), help="Padding layers for ResNet")
    # nn_common_parser.add_argument("--ceil_mode", action="store_false", help="Ceil mode for ResNet on the maxpool layer, default is True")
    # nn_common_parser.add_argument("--layers", nargs='+', type=int, help="Number of layers for custom models or ResNet_custom_layers")
    nn_common_parser.add_argument("--sync_batchnorm", action="store_true", help="Synchronize batch normalization layers")
    nn_common_parser.add_argument("--enable_checkpointing", action="store_true", help="Enable model checkpointing")
    nn_common_parser.add_argument("--save_top_k", type=int, default=1, help="Save top k models")
    nn_common_parser.add_argument("--monitor", type=str, default="val_loss", help="Monitor metric for model checkpointing")
    nn_common_parser.add_argument("--monitor_stop", type=str, default="val_loss", help="Monitor metric for early stopping")
    nn_common_parser.add_argument("--mode", type=str, default="min", help="Mode for model checkpointing ('min' or 'max')")
    nn_common_parser.add_argument("--dirpath", type=str, default="./models/ckpt", help="Directory to save models using checkpointing")
    nn_common_parser.add_argument("--filename", type=str, default="model_name_data_module_name_epoch_val_loss", help="Filename for saved models, best to not change!")
    nn_common_parser.add_argument("--strategy", type=str, default='auto', help="Training strategy (ddp, ddp_spawn, deepspeed, etc.)")
    nn_common_parser.add_argument("--loss_fn", type=str, default="cross_entropy", help="Loss function to use (cross_entropy, bce, mse)")
    nn_common_parser.add_argument("--loss_weight", type=str, choices=["balanced", None], default=None, help="Class weight for classification, default is None")
    nn_common_parser.add_argument("--num_workers", default=None, help="Number of workers for dataloader")
    nn_common_parser.add_argument("--batch_size", default=None, help="Batch size for dataloader")
    nn_common_parser.add_argument("--gradient_clip_val", type=float, default=None, help="Gradient clipping value")
    nn_common_parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")

    nn_common_parser.add_argument("--lrf_args", nargs='+', default=None, help="LearningRateFinder arguments. Implemented: min_lr, max_lr, num_training_steps, mode, milestones")
    nn_common_parser.add_argument("--swa_args", nargs='+', default=None, help="StochasticWeightAveraging arguments. Implemented: swa_lrs, swa_epoch_start, annealing_epochs, annealing_strategy")

    # NN Train parser
    nn_train_parser = nn_subparsers.add_parser("train", parents=[nn_common_parser], help="Train a neural network model")
    nn_train_parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    # nn_train_parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")

    nn_train_parser.add_argument("--layers", nargs='+', type=int, help="Number of layers for custom models or ResNet_custom_layers")
    nn_train_parser.add_argument("--num_classes", type=int, default=2, help="Number of classes to predict (ResNet)")
    nn_train_parser.add_argument("--image_channels", type=int, default=1, help="Image channels (ResNet)")
    nn_train_parser.add_argument("--padding_layer_sizes", type=tuple, default=(2, 2, 4, 3, 7, 7), help="Padding layers for ResNet")
    nn_train_parser.add_argument("--ceil_mode", action="store_false", help="Ceil mode for ResNet on the maxpool layer, default is True")

    # NN Predict parser
    nn_predict_parser = nn_subparsers.add_parser("predict", parents=[nn_common_parser], help="Predict using a neural network model")
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

def get_extra_args(args, loss_fn):
    extra_args = {
                'ceil_mode': args.ceil_mode,
                'num_classes': args.num_classes,
                'image_channels': args.image_channels,
                'padding_layer_sizes': args.padding_layer_sizes,
                'layers': args.layers,
                'padding_layer_sizes': args.padding_layer_sizes,
                'loss_fn': loss_fn,
                'learning_rate': args.learning_rate
                }
    return extra_args

def main():
    args = parse_arguments()
    
    # NN Branch
    if args.model_type == "nn":
        if args.command == "train":
            if args.model_class in ['ResNet_custom_layers', 'BaseNNModel2']:
                args = process_layers_argument(args)
            train_nn_model(args)
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

# nn train --model_class ResNet_custom_layers --enable_progress_bar --callbacks BatchSizeFinder LearningRateFinder StochasticWeightAveraging --loss_weight balanced --enable_checkpointing --save_top_k 2 --default_root_dir /Users/agreic/Documents/GitHub/cell-classification/src/nucleus_3d_classification/models/ --dirpath /Users/agreic/Documents/GitHub/cell-classification/src/nucleus_3d_classification/models/ --max_epochs 5 --swa_args annealing_epochs=2 swa_lrs=1e-3 swa_epoch_start=2  --data_module CustomDataModule --layers 1 0 0 0