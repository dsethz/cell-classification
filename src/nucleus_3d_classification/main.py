######################################################################################################################
# This script coordinates training and testing of cell segmentation with Pytorch Lightning                           #
# Author:               Aurimas Greicius, Daniel Schirmacher                                                         #
#                       Cell Systems Dynamics Group, D-BSSE, ETH Zurich                                              #
# Python Version:       3.12.2                                                                                       #
# PyTorch Version:      2.3.1                                                                                        #
# PyTorch Lightning Version: 2.3.1                                                                                   #
######################################################################################################################

import argparse
import os
from datetime import datetime

import pytorch_lightning as pl
import torch

from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI()

if __name__ == '__main__':
    cli_main()