# Description: Test the CustomDataModule class and the CustomDataset class.

from utils.datamodule import CustomDataset, CustomDataModule
from utils.transform import rotate_invert
import matplotlib.pyplot as plt
from utils.show_slice import show_slice
import torch

def test():
    data_module = CustomDataModule(setup_file='/Users/agreic/Desktop/Project/Data/Raw/Training/setup.json', batch_size=2)
    data_module.setup()

    # Get and print training dataloader length
    train_dataloader = data_module.train_dataloader()
    print(f"Training dataloader length (num batches): {len(train_dataloader)}")
    
    # Get and print validation dataloader length
    val_dataloader = data_module.val_dataloader()
    print(f"Validation dataloader length (num batches): {len(val_dataloader)}")
    
    # Get and print test dataloader length (if applicable)
    test_dataloader = data_module.test_dataloader()
    print(f"Test dataloader length (num batches): {len(test_dataloader)}")


    ztrans = rotate_invert(0,0,1)
    batch, labels = next(iter(train_dataloader))
    print(batch.shape)
    
    img = batch[0].squeeze(0)
    print(img.shape)
    # show_slice(img, label='Original Image', show=True)

    # # Show the first slice of the first image in the batch
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # show_slice(img, label='Original Image', show=False, ax=ax[0])
    # show_slice(ztrans(img), label='Transformed Image', show=False, ax=ax[1])
    # plt.show()

    # # Print shapes
    # # print('Original image shape:', img.shape)
    # print('Transformed image shape:', ztrans(img).shape)

    # Testing torch.flip
    # Create a tensor, shape Z, H, W:
    # x = torch.tensor([
    #     [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    #     [[10, 11, 12], [13, 14, 15], [16, 17, 18]], 
    #     [[19, 20, 21], [22, 23, 24], [25, 26, 27]]       
    # ])


    # print(x.shape)

    # print('Original tensor:')
    # print(x)
    # print('Flipped tensor:')
    # print(torch.flip(x, dims=[0]))

if __name__ == '__main__':
    test()