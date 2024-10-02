from utils.datamodule import CustomDataset, CustomDataModule

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

    # Accesing the first batch of the training dataloader
    for batch in train_dataloader:
        print(batch[0].shape)
        break



if __name__ == '__main__':
    test()