# Description: This file contains the padding class that is used to pad the input image to the target size

import torch.nn.functional as F

class pad:
    def __init__(self, target_size):
        self.target_size = target_size
    
    def __call__(self, img):
        depth, height, width = img.shape
        target_depth, target_height, target_width = self.target_size
        
        # Padding if necessary
        pad_depth = (target_depth - depth) if depth < target_depth else 0
        pad_height = (target_height - height) if height < target_height else 0
        pad_width = (target_width - width) if width < target_width else 0
        
        # Padding in depth, height, width directions
        padding = (
            pad_width // 2, pad_width - pad_width // 2, 
            pad_height // 2, pad_height - pad_height // 2, 
            pad_depth // 2, pad_depth - pad_depth // 2
        )

        # TODO: Resize if necessary, not implemented yet
        # img = F.resize(img, self.target)

        img = F.pad(img, padding)
        return img