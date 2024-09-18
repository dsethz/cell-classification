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

from models.ResNet import ResNet, Block

def cli_main():
    cli = LightningCLI()

if __name__ == '__main__':
    # cli_main()
    model = ResNet(block=Block, layers=[1,1,1,1], num_classes=2, image_channels=1, padding_layer_sizes=(2,2,4,3,20,19))
        #print(model)

    tensor = torch.randn(2, 1, 34, 164, 174) # Batch, Channel, Depth, Height, Width
    output = model(tensor)
    print(output.shape)
    #print(output)