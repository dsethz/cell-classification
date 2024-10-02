import torch.nn.functional as F
import torch
import random


class pad: # TODO: check if this should be a nn module and whatnot
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, img):
        depth, height, width = img.shape
        target_depth, target_height, target_width = self.target_size
        
        pad_depth = max(0, target_depth - depth)
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)
        
        # Padding in depth, height, width directions
        padding = (
                pad_width // 2, pad_width - pad_width // 2,
                pad_height // 2, pad_height - pad_height // 2,
                pad_depth // 2, pad_depth - pad_depth // 2
                )
        img = F.pad(img, padding)

        # TODO: Resize if necessary, not implemented yet
        # img = F.resize(img, self.target)

        return img

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

