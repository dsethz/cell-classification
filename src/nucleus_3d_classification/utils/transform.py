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
        padding = (
            pad_width // 2, pad_width - pad_width // 2,  # Width padding
            pad_height // 2, pad_height - pad_height // 2,  # Height padding
            pad_depth // 2, pad_depth - pad_depth // 2  # Depth padding
        )

        # Apply padding using F.pad
        # Note: F.pad expects input of shape (D, H, W)
        padded_tensor = F.pad(x, padding, mode='constant', value=0)  # Use constant padding with value 0
        #print(x.shape, padded_tensor.shape)
        return padded_tensor


class scale:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, img):
        img = (img - img.min()) / (img.max() - img.min())
        img = img * (self.max_val - self.min_val) + self.min_val
        return img
    
# Custom transformation to normalize the image intensities

class normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = (img - self.mean) / self.std
        return img
class ZDepthTransform: # TODO: Test this
    def __init__(self, rotate_prob=0.5, invert_prob=0.5):
        """
        Args:
            rotate_prob (float): Probability of applying rotation along the z-axis.
            invert_prob (float): Probability of applying inversion along the z-axis.
        """
        self.rotate_prob = rotate_prob
        self.invert_prob = invert_prob

    def __call__(self, image):
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
        if random.random() < self.rotate_prob:
            image = self.rotate_along_z(image)

        # Optionally invert along the z-axis (depth)
        if random.random() < self.invert_prob:
            image = self.invert_z(image)

        return image

    def rotate_along_z(self, image):
        """
        Rotate the image along the z-axis (depth).
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Rotated 3D image.
        """
        rotations = random.choice([0, 1, 2, 3])  # Randomly rotate 0, 90, 180, or 270 degrees
        return torch.rot90(image, rotations, dims=[1, 2])

    def invert_z(self, image):
        """
        Invert the image along the z-axis (depth).
        
        Args:
            image (Tensor): 3D image tensor of shape (D, H, W).
        
        Returns:
            Tensor: Inverted 3D image.
        """
        return torch.flip(image, dims=[0])  # Invert along the depth (z-axis)

