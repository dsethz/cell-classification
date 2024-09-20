# import Lightning as L
# import torch.nn as nn
# import torch.nn.functional as F

# # define the LightningModule
# class lightning_model(L.LightningModule):
#     def __init__(self, model, save_hyperparameters=True, learning_rate=1e-3, loss_fn=F.cross_entropy):
#         super().__init__()

#         self.model = model
#         self.learning_rate = learning_rate
#         self.loss_fn = loss_fn

#         if save_hyperparameters:
#             self.save_hyperparameters()


#     def forward(self, x):
#         x = self.model(x)
#         return x
    
#     def training_step(self, batch, batch_idx):
#         loss_fn = self.loss_fn
#         x, y = batch
#         y_hat = self.model(x)
#         loss = loss_fn(y_hat, y)
#         self.log('training_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         loss_fn = self.loss_fn
#         x, y = batch
#         y_hat = self.model(x)
#         val_loss = loss_fn(y_hat, y)

        
#         # TODO: Fix this
#         # y_pred = y_hat.argmax(dim=1)


#         # # Calculate accuracy, precision, recall
#         # acc = (x.argmax(dim=1) == y).float().mean()
#         # precision = (x.argmax(dim=1) == y).float().mean()
#         # recall = (x.argmax(dim=1) == y).float().mean()
        
#         # values = {"val_loss": val_loss, "val_accuracy": acc, "val_precision": precision, "val_recall": recall}

#         # # Log the validation loss
#         # self.log_dict(values, on_epoch=True,
#         #         prog_bar=True, sync_dist=True) # Sync dist is used for distributed training

#         return val_loss
    
#     def test_step(self, batch, batch_idx):
#         loss_fn = self.loss_fn

#         x, y = batch
#         y_hat = self.model(x)
#         Test_step_loss = loss_fn(y_hat, y)

#         # TODO: Fix this
#         # # Calculate accuracy, precision, recall
#         # test_acc = (x.argmax(dim=1) == y).float().mean()
#         # test_precision = (x.argmax(dim=1) == y).float().mean()
#         # test_recall = (x.argmax(dim=1) == y).float().mean()
        
#         # values = {"test_loss": Test_step_loss, "test_accuracy": test_acc, "test_precision": test_precision, "test_recall": test_recall}

#         # # Log the validation loss
#         # self.log_dict(values,
#         #         prog_bar=True, sync_dist=True) # Sync dist is used for distributed training

#         return Test_step_loss
    
#     def predict_step(self, x):
#         # TODO: Check this out
#         x = self.model(x)
#         return x
    
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)
#         # Adam.self.parameters() is used to optimize the model, we can input these parameters: 
#         # lr, betas, eps, weight_decay, amsgrad, etc.