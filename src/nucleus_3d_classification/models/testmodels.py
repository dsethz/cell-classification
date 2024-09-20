# This will hold models that are used for testing purposes

class testBlock(nn.Module):
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
    def __init__(self, Block, layers = 3):
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
        x = batch
        return self(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)