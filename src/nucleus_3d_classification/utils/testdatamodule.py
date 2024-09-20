import torch
import pytorch_lightning as pl
import lightning as L
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Optional

class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # These will be set in the setup() method
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # Generate random synthetic data: 1000 samples, 10 features
        num_samples = 1000
        num_features = 10
        num_classes = 2

        # Create random tensors for features (X) and labels (y)
        X = torch.randn(num_samples, num_features)
        y = torch.randint(0, num_classes, (num_samples,))

        # Wrap data into a TensorDataset
        dataset = TensorDataset(X, y)

        # Split into training, validation, and test datasets
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)  # Example: using test dataset for prediction
