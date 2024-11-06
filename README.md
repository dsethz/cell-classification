# cell-classification

[![License BSD-3](https://img.shields.io/pypi/l/cell-classification.svg?color=green)](https://github.com/agreic/cell-classification/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/cell-classification.svg?color=green)](https://pypi.org/project/cell-classification)
[![Python Version](https://img.shields.io/pypi/pyversions/cell-classification.svg?color=green)](https://python.org)
[![tests](https://github.com/agreic/cell-classification/workflows/tests/badge.svg)](https://github.com/agreic/cell-classification/actions)
[![codecov](https://codecov.io/gh/agreic/cell-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/agreic/cell-classification)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/cell-classification)](https://napari-hub.org/plugins/cell-classification)

A cell classifier using information limited to images of DAPI stained nuclei.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

# Flexible Model Training and Prediction Interface

## Overview

This project provides a flexible interface for training and predicting with various model types: neural networks (`nn`), logistic regression (`logreg`), and random forests (`rf`). The interface uses `argparse` for handling a wide array of customizable options, allowing tailored configurations for both training and prediction.

## Installation

This project was built with **Python 3.12.2** and may not support other Python versions. To install:

1. Clone the repository and navigate into it:
    ```bash
    git clone <repository-url>
    cd cell-classification
    ```

2. Install `cell-classification` via [pip]:
    ```bash
    pip install .
    ```

## Project Structure

Below is an overview of the project structure, outlining the main directories and their purpose:

```
cell-classification/
├── Analysis/                   # Analysis files: plotting, validation, and test scoring
├── src/
│   └── nucleus_3d_classification/
│       ├── preprocess/         # Scripts for dataset generation and ground-truth creation
│       ├── baseline/           # Script to fit baseline models on 2D as well as 3D extracted features, with setup files
│       ├── utils/              # Dataset preparation, transformations, and loss functions
│       ├── models/             # Model architectures (ResNet support)
│       └── main.py             # Main entry point for model training
└── setup_json/                 # Example JSON configurations for NN training, including model_runner setup
```

### Folder Details

1. **Analysis Folder**  
   Contains all files for data analysis, including scripts for plotting and evaluating model scores on validation/test data.

2. **Source Code**  
   Located in `src/nucleus_3d_classification/`, this folder contains the core codebase:
   - **`preprocess/`**: Scripts to generate and curate the ground-truth dataset. For example:
       - `napari_curation_and_labelling.py`: A Napari plug-in tool used to label and curate nuclei segmentation masks.
       - Additional scripts for feature extraction and crop generation.
   - **`utils/`**: Contains essential scripts for preparing the dataset for NN training and testing, including:
       - `datamodule.py`: Dataset setup.
       - `transforms.py`: Custom transformations.
       - Loss function configurations.
   - **`models/`**: Defines the available model architectures. Currently, only ResNet models are supported, with all configurations in `ResNet.py`.

3. **Setup JSON**  
   The `setup_json` directory provides examples of how JSON files should be formatted for the `--data` argument during NN training. These examples demonstrate the data structure required by the models. The `model_runner_example.py` script contains a sample configuration file illustrating the setup for the `model_runner --params` argument, as used during model training.

### Basic Usage

Run the script by specifying the model type (`--model_type`) and command (`--command`), along with the necessary parameters for each specific model.

The main entry point for training models is `main.py`. To view usage options, run:

```bash
python main.py --h
```

To view more options for a specific model type and command, for example a neural network and train, run:

```bash
python main.py --model_type nn --command train --h
```

#### Examples

**Training a Neural Network:**

```bash
python script.py --model_type nn --command train --data /path/to/cd41_setup.json --data_module CustomDataModule --model_class ResNet50 --max_epochs 20
```

**Testing a trained Neural Network:**

```bash
python main.py --model_type nn --command predict --data /path/to/Sca_setup.json --model_class ResNet50 --batch_size 4 --enable_progress_bar --data_module CustomDataModule --model_file /path/to/Sca1_best-f1_score-epoch=87-val_f1=0.34.ckpt --stage test --save_dir /cluster/project/schroeder/AG/CD41/results/predictions/sca1/
```


## Below is kept as generated by CookieCutter.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"cell-classification" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
