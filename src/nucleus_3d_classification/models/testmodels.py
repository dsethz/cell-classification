# This will hold models that are used for testing purposes
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import pytorch_lightning as pl


class testBlock(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 10)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class BaseNNModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class BaseNNModel2(L.LightningModule):
    def __init__(self, Block=testBlock, layers = 3):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)
        self.Block = Block
        self.layer1 = self._make_layers(layers)

    def _make_layers(self,layers):
        blocks = []
        for _ in range(1, layers):
            blocks.append(self.Block())
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.fc(x)
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss)
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch # As this is a dummy model, we are using data which has labels, thus we need to extract them
        y_hat = self(x)
        # Format the output to be a single prediction value for each sample
        y_hat = torch.argmax(y_hat, dim=1)
        print(f"Predictions: {y_hat}, True labels: {y}")
        return y_hat
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    

    ######
# def get_nn_model(model_name: str, extra_args: dict = None):

#     if extra_args is None:
#         extra_args = {}

#     if model_name == "BaseNNModel":
#         model = BaseNNModel()
#         return model
#     elif model_name == "BaseNNModel2":
#         if 'Block' not in extra_args or 'layers' not in extra_args:
#             raise ValueError("BaseNNModel2 requires 'Block' and 'layers' in extra_args.")

#         model = BaseNNModel2(Block=extra_args['Block'], layers=extra_args['layers'])

#         return model

def get_BaseNNModel():
    return BaseNNModel

def get_BaseNNModel2(layers):
    return BaseNNModel2(layers=layers)

if __name__ == '__main__':
        extra_args = {'Block': testBlock, 'layers': 3}
        model = get_nn_model('BaseNNModel2', extra_args=extra_args)
        print(isinstance(model, L.LightningModule))

        model = testBlock()
        print(isinstance(model, L.LightningModule))