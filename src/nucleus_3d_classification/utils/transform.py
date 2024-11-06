'''
This module contains custom PyTorch transformations for 3D image data.
These transformations can be used to augment the data during training.

The following transformations are implemented:
- pad: Pads the input tensor to the specified target size.
- scale: Scales the input image intensities to the specified range.
- normalize: Normalizes the input image intensities using the specified mean and standard deviation.
- rotate_invert: Applies random rotations and inversions to the input 3D image.
'''

import torch.nn.functional as F
import torch
from torchvision.transforms import v2
import random

class pad(torch.nn.Module):
    def __init__(self, target_size):
        """
        Initializes the Pad module.

        Args:
            target_size (tuple): The desired target size for padding in the format (depth, height, width).
        """
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        """
        Pads the input tensor to the specified target size.

        Args:
            x (torch.Tensor): Input tensor of shape (D, H, W) where:
                - D: Depth
                - H: Height
                - W: Width

        Returns:
            torch.Tensor: Padded tensor.
        """
        # Get the current shape of the input tensor
        depth, height, width = x.shape  # Assuming input shape is (D, H, W)
        target_depth, target_height, target_width = self.target_size
        
        # Calculate padding sizes for depth, height, and width
        pad_depth = max(0, target_depth - depth)
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)
        
        # Padding configuration: (left, right, top, bottom, front, back)
        # Eg X Y Z

        # Random padding on the left or right side if theres an odd number of pixels to pad
        pad_width = random.choice([[pad_width // 2, pad_width - pad_width // 2], [pad_width - pad_width // 2, pad_width // 2]])

        # Random padding on the top or bottom side if theres an odd number of pixels to pad
        pad_height = random.choice([[pad_height // 2, pad_height - pad_height // 2], [pad_height - pad_height // 2, pad_height // 2]])

        # Random padding on the front or back side if theres an odd number of pixels to pad
        pad_depth = random.choice([[pad_depth // 2, pad_depth - pad_depth // 2], [pad_depth - pad_depth // 2, pad_depth // 2]])

        padding = (
            pad_width[0], pad_width[1],  # Width padding
            pad_height[0], pad_height[1],  # Height padding
            pad_depth[0], pad_depth[1] # Depth padding
        )

        # padding = ( ## OLD WAY
        #     pad_width // 2, pad_width - pad_width // 2,  # Width padding
        #     pad_height // 2, pad_height - pad_height // 2,  # Height padding
        #     pad_depth // 2, pad_depth - pad_depth // 2  # Depth padding
        # )
        
        # Apply padding using F.pad
        # Note: F.pad expects input of shape (D, H, W)
        padded_tensor = F.pad(x, padding, mode='constant', value=0)  # Use constant padding with value 0
        #print(x.shape, padded_tensor.shape)
        return padded_tensor


class scale(torch.nn.Module):
    def __init__(self, min_val, max_val):
        """
        Initializes the Scale module.

        Args:
            min_val (float): Minimum value for scaling.
            max_val (float): Maximum value for scaling.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        '''
        Scales the input image intensities to the specified range.
        '''
        img = (img - img.min()) / (img.max() - img.min())
        img = img * (self.max_val - self.min_val) + self.min_val
        return img
    
# Custom transformation to normalize the image intensities

class normalize(torch.nn.Module):
    def __init__(self, mean, std):
        """
        Initializes the Normalize module.

        Args:
            mean (float): Mean value for normalization.
            std (float): Standard deviation value for normalization.
        """
        super().__init__()
        self.mean = mean
        self.std = std
    
    def forward(self, img):
        img = (img - self.mean) / self.std
        return img
class rotate_invert(torch.nn.Module): 
    def __init__(self, rotate_prob=0, invert_z_prob=0.5, invert_x_prob=0.5, invert_y_prob=0.5):
        """
        Args:
            rotate_prob (float): Probability of applying rotation along the z-axis.
            invert_prob (float): Probability of applying inversion along the z-axis.
        """
        super().__init__()
        self.rotate_prob = rotate_prob
        self.invert_z_prob = invert_z_prob
        self.invert_x_prob = invert_x_prob
        self.invert_y_prob = invert_y_prob

    def forward(self, image):
        """
        Apply the transformations to the 3D image.
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Transformed 3D image.
        """
        # Ensure the image has 3 dimensions (D, H, W)
        assert image.dim() == 3, "Input image must be 3D (D, H, W)"

        # Optionally rotate along the z-axis (depth)
        #if random.random() < self.rotate_prob:
        #    image = self.rotate_along_z(image)

        # Optionally invert along the z-axis (depth)
        if random.random() < self.invert_z_prob:
            image = self.invert_z(image)

        if random.random() < self.invert_x_prob:
            image = self.invert_x(image)

        if random.random() < self.invert_y_prob:
            image = self.invert_y(image)

        return image

    def rotate_along_z(self, image):
        """
        Rotate the image along the z-axis (depth).
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Rotated 3D image.
        """
        rotations = random.choice([1,2,3])  # Randomly rotate 90, 180, or 270 degrees
        # print('Image is rotated by', rotations*90, 'degrees')
        return torch.rot90(image, rotations, dims=[1, 2])

    def invert_z(self, image):
        """
        Invert the image along the z-axis (depth).
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Inverted 3D image.
        """
        
        #print('Image is inverted along the z-axis')
        return torch.flip(image, dims=[0])  # Invert along the depth (z-axis)
    
    def invert_x(self, image):
        """
        Invert the image along the x axis.
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Inverted 3D image (in x).
        """
        
        return torch.flip(image, dims=[2])
    
    def invert_y(self, image):
        """
        Invert the image along the y axis.
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Inverted 3D image (in y).
        """
        
        return torch.flip(image, dims=[1])
