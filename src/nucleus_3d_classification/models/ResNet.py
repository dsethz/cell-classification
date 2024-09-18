import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F

class Block(nn.Module):
    '''
    This class is a single block of the ResNet model. It consists of 3 convolutional layers and can have an identity downsample.
    '''
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(Block, self).__init__()
        self.expansion = 4

        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn0 = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 =nn.BatchNorm3d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.identity_downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels*self.expansion)
                )
        else:
            self.identity_downsample = None
    
    def forward(self, x):
        identity=x

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(L.LightningModule):
    def __init__(self, block, layers, num_classes, 
                image_channels=1, ceil_mode=False,
                zero_init_resudual: bool = False, # TODO: Check if we can implement this
                padding_layer_sizes=None):
        super(ResNet, self).__init__()
        '''
        This class is the ResNet model. It consists of an initial convolutional layer, 4 residual blocks and a final fully connected layer.
        The default parameters are the same as in the Pytorch implementation of ResNet at https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py,
        checked at 2024-09-18, using the stride for downsampling at the second 3x3x3 convolution, additionally, the model is adapted to work with 3D data.
        The model can be modified to have a different number of layers in each block, by changing the layers parameter, as well as allowing the ceil_mode
        parameter to be set to True or False for the max pooling layer (for odd inputs). An extra padding layer can be added after the max pooling layer
        to ensure that the data is the correct size for the first residual block.
        '''

        # Variables

        self.initial_out_channels = 64
        self.in_channels = self.initial_out_channels
        self.padding_layer_sizes = padding_layer_sizes

        # INITIAL LAYERS

        self.conv1 = nn.Conv3d(in_channels=image_channels, out_channels=self.initial_out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.initial_out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=ceil_mode) # TODO: Check if want ceil mode true

        # RESIDUAL BLOCKS

        self.conv2_x = self.make_layers(block=block, num_blocks=layers[0], out_channels=64, stride=1)
        self.conv3_x = self.make_layers(block=block, num_blocks=layers[1], out_channels=128, stride=2)
        self.conv4_x = self.make_layers(block=block, num_blocks=layers[2], out_channels=256, stride=2)
        self.conv5_x = self.make_layers(block=block, num_blocks=layers[3], out_channels=512, stride=2)

        # FINAL LAYERS

        # Avg pool
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        # FC
        self.fc = nn.Linear(self.in_channels, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.max_pool(x)

    #     x = self.conv2_x(x)
    #     x = self.conv3_x(x)
    #     x = self.conv4_x(x)
    #     x = self.conv5_x(x)

    #     x = self.avg_pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(f"After Conv1: {x.shape}")
        x = self.max_pool(x)
        print(f"After MaxPool: {x.shape}")

        # PADDING
        if self.padding_layer_sizes is not None:
            x = F.pad(x, self.padding_layer_sizes)  # TODO: Check if I want this, as this is not in the original code
            print(f'After padding: {x.shape}')

        x = self.conv2_x(x)
        print(f"After Conv2_x: {x.shape}")
        x = self.conv3_x(x)
        print(f"After Conv3_x: {x.shape}")
        x = self.conv4_x(x)
        print(f"After Conv4_x: {x.shape}")
        x = self.conv5_x(x)
        print(f"After Conv5_x: {x.shape}")

        x = self.avg_pool(x)
        print(f"After AvgPool: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"After Reshape: {x.shape}")
        x = self.fc(x)
        print(f"After FC: {x.shape}")

        return x

    def make_layers(self, block, num_blocks, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*4

        for i in range(num_blocks-1): # -1 cause above we create 1 already
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)
    
    def print_model(self):
        print(self)

def ResNet50(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None):
    '''
    This function creates a ResNet50 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,4,6,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes)

def ResNet101(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None):
    '''
    This function creates a ResNet101 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,4,23,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes)

def ResNet152(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None):
    '''
    This function creates a ResNet152 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,8,36,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes)

######################################################################################################################

# TODO : REMOVE BELOW, JUST FOR TESTING

def main():
    model = ResNet(block=Block, layers=[1,1,1,1], num_classes=2, image_channels=1, padding_layer_sizes=(2,2,4,3,20,19))
    #print(model)

    tensor = torch.randn(2, 1, 34, 164, 174) # Batch, Channel, Depth, Height, Width
    output = model(tensor)
    print(output.shape)
    #print(output)

# Before avg pool torch.Size([2, 1024, 5, 10, 11])
# After avg pool torch.Size([2, 1024, 1, 1, 1])
# Reshaping it to torch.Size([2, 1024])
# Using a linear layer to get torch.Size([2, 2])

# // Input shape: torch.Size([2, 1, 34, 164, 174])
# After Conv1: torch.Size([2, 64, 17, 82, 87])
# After MaxPool: torch.Size([2, 64, 9, 42, 44])
# After padding: torch.Size([2, 64, 48, 48, 48])
# After Conv2_x: torch.Size([2, 256, 48, 48, 48])
# After Conv3_x: torch.Size([2, 512, 24, 24, 24])
# After Conv4_x: torch.Size([2, 1024, 12, 12, 12])
# After Conv5_x: torch.Size([2, 2048, 6, 6, 6])
# After AvgPool: torch.Size([2, 2048, 1, 1, 1])
# After Reshape: torch.Size([2, 2048])
# After FC: torch.Size([2, 2])
# torch.Size([2, 2])

if __name__ == '__main__':
    main()