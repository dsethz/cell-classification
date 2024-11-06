'''
The below code was useed to test the functionality of the lightning module.
It should not be used for any other purpose.
'''

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

        # Initialize TP, FP, TN, FN counters
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

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

        # Flatten tensors for comparison if they have extra dimensions
        y = y.view(-1)
        predicted_classes = y_pred_class.view(-1)

        # Calculate TP, FP, TN, FN
        self.tp += torch.sum((y == 1) & (predicted_classes == 1)).item()
        self.fp += torch.sum((y == 0) & (predicted_classes == 1)).item()
        self.tn += torch.sum((y == 0) & (predicted_classes == 0)).item()
        self.fn += torch.sum((y == 1) & (predicted_classes == 0)).item()
                # Optional: print for debugging purposes
        print(f"Predicted: {predicted_classes}, True: {y}")
        return test_loss
    
    def on_test_epoch_end(self):
        # Calculate accuracy, precision, recall, etc.
        total = self.tp + self.fp + self.tn + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Log or print your metrics
        self.log('test_accuracy', accuracy, on_epoch=True, sync_dist=True)
        self.log('test_precision', precision, on_epoch=True, sync_dist=True)
        self.log('test_recall', recall, on_epoch=True, sync_dist=True)
        self.log('test_f1_score', f1_score, on_epoch=True, sync_dist=True)

        # Print for debugging
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

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
