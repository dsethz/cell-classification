# NOT USED
# Description: Function to show a slice of an image
import matplotlib.pyplot as plt
import numpy as np


def show_slice(image, label = None, slice_index=None, ax=None, show=True):
    image = image if isinstance(image, np.ndarray) else image.cpu().detach().numpy()

    if slice_index is None:
        # If a slice index is not provided, take the middle slice
        slice_index = int(round(image.shape[0] / 2, 0))
    
    if ax is None:
        # Get the current Axes instance if not provided
        ax = plt.gca()

    image = image[slice_index]  # Get the slice of the image at the specified index

    ax.imshow(image, cmap='gray')
    if label is not None:
        ax.set_title(label)

    plt.tight_layout()
    ax.axis('off')
    if show:
        plt.show()
