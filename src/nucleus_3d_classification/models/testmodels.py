import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import pytorch_lightning as pl
# from utils.eval.evals import MyAccuracy


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
    def __init__(self, Block=testBlock, layers=3):
        super().__init__()
        self.save_hyperparameters()

        #self.accuracy=MyAccuracy()

        self.fc = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)
        self.Block = Block
        self.layer1 = self._make_layers(layers)

        num_classes = 2
        self.num_classes = num_classes
        self.results = {}

    def _make_layers(self, layers):
        blocks = []
        for _ in range(1, layers):
            blocks.append(self.Block())
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.fc(x)
        x = self.layer1(x)
        x = self.fc2(x)
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
    
        # Apply softmax to get probabilities
        y_pred_proba = torch.softmax(y_hat, dim=1)
        
        # Get predicted class
        y_pred_class = torch.argmax(y_hat, dim=1)

        print(y_pred_class, y)

        print(y_pred_proba, y)

        # precision = self.precision(y_pred_class, y)


        # values = {
        #     "precision": precision,
        #     "recall": recall,
        #     }
        

        # self.log_dict(values, on_epoch=True, on_step=False, sync_dist=True) #, reduce_fx=torch.mean)
    
        return test_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)  # Shape: [batch_size, num_classes]
        # # Get predicted class (for accuracy calculation)
        y_pred_class = torch.argmax(y_hat, dim=1)
        
        # Return predictions
        return y_pred_class
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
    

# Function to return BaseNNModel
def get_BaseNNModel():
    return BaseNNModel()

# Function to return BaseNNModel2
def get_BaseNNModel2(layers):
    return BaseNNModel2(layers=layers)

if __name__ == '__main__':
    model = BaseNNModel2(layers=3)
    print(isinstance(model, L.LightningModule))
