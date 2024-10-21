import torch.nn as nn
import torch
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

class Block(nn.Module):
    '''
    This class is a single block of the ResNet model. It consists of 3 convolutional layers and can have an identity downsample.
    '''
    def __init__(self, in_channels, out_channels, stride=1, bias=False, leaky=True):
        super(Block, self).__init__()
        self.expansion = 4

        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn0 = nn.BatchNorm3d(out_channels)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn2 =nn.BatchNorm3d(out_channels*self.expansion)
        if leaky == True:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif leaky == False:
            self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or in_channels != out_channels*self.expansion:
            self.identity_downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels*self.expansion)
                )
        else:
            self.identity_downsample = None
    
    def forward(self, x):
        identity=x

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(L.LightningModule):
    def __init__(self, layers, num_classes, block=Block,
                image_channels=1, ceil_mode=False,
                zero_init_resudual: bool = False, # TODO: Check if we can implement this
                padding_layer_sizes=None, learning_rate=1e-5,
                loss_fn = 'cross_entropy', scheduler_type=None, step_size=10, gamma=0.9, leaky=True):
        super(ResNet, self).__init__()
        '''
        This class is the ResNet model. It consists of an initial convolutional layer, 4 residual blocks and a final fully connected layer.
        The default parameters are the same as in the Pytorch implementation of ResNet at https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py,
        checked at 2024-09-18, using the stride for downsampling at the second 3x3x3 convolution, additionally, the model is adapted to work with 3D data.
        The model can be modified to have a different number of layers in each block, by changing the layers parameter, as well as allowing the ceil_mode
        parameter to be set to True or False for the max pooling layer (for odd inputs). An extra padding layer can be added after the max pooling layer
        to ensure that the data is the correct size for the first residual block.
        '''

        # Variables
        self.initial_out_channels = 64
        self.in_channels = self.initial_out_channels
        self.padding_layer_sizes = padding_layer_sizes
        self.learning_rate = learning_rate
        self.ceil_mode = ceil_mode
        
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma

        # Initialize TP, FP, TN, FN counters
        self.tp = self.fp = self.tn = self.fn = 0

        # Save Hyperparameters
        self.save_hyperparameters()

        # Loss fn handling
        match loss_fn:
            case 'cross_entropy':
                self.loss_fn = F.cross_entropy
            case 'mse':
                self.loss_fn = F.mse_loss
            case _:
                try:
                    self.loss_fn = loss_fn
                    print(f"Using custom loss function: {loss_fn}")
                except:
                    return ValueError(f"Loss function {loss_fn} not supported for this model!")

        # INITIAL LAYERS

        if leaky == True:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif leaky == False:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels=image_channels, out_channels=self.initial_out_channels, kernel_size=7, stride=(1,2,2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.initial_out_channels)
        self.max_pool = nn.MaxPool3d(kernel_size=3, stride=(1,2,2), padding=1, ceil_mode=ceil_mode) # TODO: Check if want ceil mode true

        # RESIDUAL BLOCKS

        self.conv2_x = self.make_layers(block=block, num_blocks=layers[0], out_channels=64, stride=1)
        self.conv3_x = self.make_layers(block=block, num_blocks=layers[1], out_channels=128, stride=2)
        self.conv4_x = self.make_layers(block=block, num_blocks=layers[2], out_channels=256, stride=2)
        self.conv5_x = self.make_layers(block=block, num_blocks=layers[3], out_channels=512, stride=2)

        # FINAL LAYERS

        # Avg pool
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        # FC
        self.fc = nn.Linear(self.in_channels, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_metrics(self, y_hat, y):
        pred = torch.argmax(y_hat, dim=1)
        self.tp += torch.sum((pred == 1) & (y == 1)).item()
        self.fp += torch.sum((pred == 1) & (y == 0)).item()
        self.tn += torch.sum((pred == 0) & (y == 0)).item()
        self.fn += torch.sum((pred == 0) & (y == 1)).item()

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print(f"After Conv1: {x.shape}")
        x = self.max_pool(x)
        # print(f"After MaxPool: {x.shape}")

        # PADDING
        if self.padding_layer_sizes is not None:
            x = F.pad(x, self.padding_layer_sizes)  # TODO: change this to be adaptive
            # print(f'After padding: {x.shape}')

        # x = pad_to_shape(x, (2, 64, 48, 48, 48))

        x = self.conv2_x(x)
        # print(f"After Conv2_x: {x.shape}")
        x = self.conv3_x(x)
        # print(f"After Conv3_x: {x.shape}")
        x = self.conv4_x(x)
        # print(f"After Conv4_x: {x.shape}")
        x = self.conv5_x(x)
        # print(f"After Conv5_x: {x.shape}")

        x = self.avg_pool(x)
        # print(f"After AvgPool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After Reshape: {x.shape}")
        x = self.fc(x)
        # print(f"After FC: {x.shape}")

        return x

    def make_layers(self, block, num_blocks, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*4

        for i in range(num_blocks-1): # -1 cause above we create 1 already
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)
    
    def print_model(self):
        print(self)
    
    def training_step(self, batch, batch_idx):
        loss_fn = self.loss_fn
        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)

        # Update metric counters
        self.update_metrics(y_hat, y)

        values = {"training_loss": loss}
        self.log_dict(values, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_fn = self.loss_fn
        x, y = batch
        # print(f"True labels: {y}")
        y_hat = self.forward(x)
        val_loss = loss_fn(y_hat, y)

        # Update metric counters
        self.update_metrics(y_hat, y)

        values = {"val_loss": val_loss}
        self.log_dict(values, on_epoch=True, prog_bar=True, sync_dist=True) # Sync dist is used for distributed training

        return val_loss

    def on_train_epoch_end(self):
        self.log_metrics('train')
        self.reset_metrics()

    def on_validation_epoch_end(self):
        self.log_metrics('val')
        self.reset_metrics()

    def on_test_epoch_end(self):
        self.log_metrics('test')
        self.reset_metrics()

    def log_metrics(self, prefix):
        total = self.tp + self.tn + self.fp + self.fn
        accuracy = (self.tp + self.tn) / total if total > 0 else 0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        self.log(f'{prefix}_tp', self.tp, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_tn', self.tn, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_fp', self.fp, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_fn', self.fn, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_accuracy', accuracy, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_precision', precision, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_recall', recall, on_epoch=True, sync_dist=True)
        self.log(f'{prefix}_f1', f1, on_epoch=True, sync_dist=True)

        print(f"{prefix.capitalize()} Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        print(f"{prefix.capitalize()} Counts - TP: {self.tp}, TN: {self.tn}, FP: {self.fp}, FN: {self.fn}")

    def reset_metrics(self):
        self.tp = self.fp = self.tn = self.fn = 0

    def test_step(self, batch, batch_idx):
        loss_fn = self.loss_fn

        x, y = batch
        y_hat = self.forward(x)
        Test_step_loss = loss_fn(y_hat, y)

        # Update metric counters
        self.update_metrics(y_hat, y)

        self.log('test_loss', Test_step_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        return Test_step_loss

    def predict_step(self, x):
        # TODO: Check this out
        x = self.forward(x)
        return x
    
    def configure_optimizers(self): # TODO: Implement scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.scheduler_type is None:
            return optimizer
        else:
            # Initialize the learning rate scheduler based on the specified type
            if self.scheduler_type == 'StepLR':
                scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            elif self.scheduler_type == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
            elif self.scheduler_type == 'ExponentialLR':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            else:
                print(f"Unknown scheduler type: {self.scheduler_type}")
                print(f"Using default optimizer: Adam with lr={self.learning_rate}")
                return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step', depending on the scheduler
                'frequency': 1,       # How often to call the scheduler (every epoch or every step)
            },
        }

    

def ResNet_custom_layers(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None, layers=[1,1,1,1], loss_fn='cross_entropy', learning_rate=1e-5):
    '''
    This function creates a ResNet model with the specified number of classes and image channels.
    The layers parameter can be used to specify the number of layers in each block.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=layers, num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes, loss_fn=loss_fn, learning_rate=learning_rate)

def ResNet50(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None, loss_fn='cross_entropy', learning_rate=1e-5):
    '''
    This function creates a ResNet50 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,4,6,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes, loss_fn=loss_fn, learning_rate=learning_rate)

def ResNet101(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None, loss_fn='cross_entropy', learning_rate=1e-5):
    '''
    This function creates a ResNet101 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,4,23,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes, loss_fn=loss_fn, learning_rate=learning_rate)

def ResNet152(num_classes, image_channels=1, ceil_mode=False, zero_init_residual=False, padding_layer_sizes=None, loss_fn='cross_entropy', learning_rate=1e-5):
    '''
    This function creates a ResNet152 model with the specified number of classes and image channels.
    The ceil_mode parameter can be set to True or False for the max pooling layer (for odd inputs),
    and the padding_layer_sizes parameter can be set to a tuple of 6 integers to specify the padding before the first residual block.
    '''
    return ResNet(block=Block, layers=[3,8,36,3], num_classes=num_classes, image_channels=image_channels, ceil_mode=ceil_mode, zero_init_resudual=zero_init_residual, padding_layer_sizes=padding_layer_sizes, loss_fn=loss_fn, learning_rate=learning_rate)

class testNet(L.LightningModule):
    def __init__(self, layers=0, num_classes=2, block=Block,
                image_channels=1, ceil_mode=False,
                zero_init_resudual: bool = False, # TODO: Check if we can implement this
                padding_layer_sizes=None, learning_rate=1e-5,
                loss_fn = 'cross_entropy', scheduler_type=None, step_size=10, gamma=0.9, leaky=True):
        super().__init__()
        '''
        This class is the ResNet model. It consists of an initial convolutional layer, 4 residual blocks and a final fully connected layer.
        The default parameters are the same as in the Pytorch implementation of ResNet at https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py,
        checked at 2024-09-18, using the stride for downsampling at the second 3x3x3 convolution, additionally, the model is adapted to work with 3D data,
        using 3D convolutions instead of 2D and a leakyReLU is default instead of the normal ReLU.
        The model can be modified to have a different number of layers in each block, by changing the layers parameter, as well as allowing the ceil_mode
        parameter to be set to True or False for the max pooling layer (for odd inputs). An extra padding layer can be added after the max pooling layer
        to ensure that the data is the correct size for the first residual block.
        '''

        # Variables
        self.initial_out_channels = 64
        self.in_channels = self.initial_out_channels
        self.padding_layer_sizes = padding_layer_sizes
        self.learning_rate = learning_rate
        self.ceil_mode = ceil_mode
        
        self.scheduler_type = scheduler_type
        self.step_size = step_size
        self.gamma = gamma

        # Initialize TP, FP, TN, FN counters
        self.tp = self.fp = self.tn = self.fn = 0

        self.val_tp=0
        self.val_fp=0
        self.val_tn=0
        self.val_fn=0

        # Save Hyperparameters
        self.save_hyperparameters()

        # Loss fn handling
        match loss_fn:
            case 'cross_entropy':
                self.loss_fn = F.cross_entropy
            case 'mse':
                self.loss_fn = F.mse_loss
            case _:
                try:
                    self.loss_fn = loss_fn
                    print(f"Using custom loss function: {loss_fn}")
                except:
                    return ValueError(f"Loss function {loss_fn} not supported for this model!")

        # INITIAL LAYERS

        if leaky == True:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif leaky == False:
            self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_channels=image_channels, out_channels=self.initial_out_channels, kernel_size=10, stride=(1,2,2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.initial_out_channels)
        #self.relu = nn.ReLU(inplace=True)

        # FINAL LAYERS

        # Avg pool
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        # FC
        self.fc = nn.Linear(self.initial_out_channels, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_metrics(self, y_hat, y):
        pred = torch.argmax(y_hat, dim=1)
        self.tp += torch.sum((pred == 1) & (y == 1)).item()
        self.fp += torch.sum((pred == 1) & (y == 0)).item()
        self.tn += torch.sum((pred == 0) & (y == 0)).item()
        self.fn += torch.sum((pred == 0) & (y == 1)).item()

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        # Resize to 10x10x10 for testing:
        #print(f"Before Resize: {x.shape}")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.avg_pool(x)
        #print(f"After AvgPool: {x.shape}")
        x = x.view(x.size(0), -1)
        #print(f"After Reshape: {x.shape}")
        x = self.fc(x)
        #print(f"After FC: {x.shape}")

        return x

    def make_layers(self, block, num_blocks, out_channels, stride):

        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels*4

        for i in range(num_blocks-1): # -1 cause above we create 1 already
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)
    
    def print_model(self):
        print(self)
    
    def training_step(self, batch, batch_idx):
        loss_fn = self.loss_fn
        x, y = batch
        y_hat = self.forward(x)
        loss = loss_fn(y_hat, y)
        self.log('training_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss_fn = self.loss_fn
        x, y = batch
        y_hat = self.forward(x)
        val_loss = loss_fn(y_hat, y)

        y_pred_class = torch.argmax(y_hat, dim=1)        
        # Flatten tensors for comparison if they have extra dimensions
        y = y.view(-1)
        predicted_classes = y_pred_class.view(-1)

        # Calculate TP, FP, TN, FN
        self.val_tp += torch.sum((y == 1) & (predicted_classes == 1)).item()
        self.val_fp += torch.sum((y == 0) & (predicted_classes == 1)).item()
        self.val_tn += torch.sum((y == 0) & (predicted_classes == 0)).item()
        self.val_fn += torch.sum((y == 1) & (predicted_classes == 0)).item()
                # Optional: print for debugging purposes
        #print(f"Predicted: {predicted_classes}, True: {y}")
        #self.log('val_loss', val_loss,sync_dist=True)

        # # Calculate accuracy, precision, recall
        #     # convert to probabilities
        # y_pred_proba = torch.softmax(y_hat, dim=1)
        # acc = self.accuracy(y_hat, y)

        values = {"val_loss": val_loss}
        self.log_dict(values, on_epoch=True, prog_bar=True, sync_dist=True) # Sync dist is used for distributed training

        return val_loss

    def on_validation_epoch_end(self):

        # Calculate accuracy, precision, recall, etc.
        total_val = self.val_tp + self.val_fp + self.val_tn + self.val_fn
        accuracy_val = (self.val_tp + self.val_tn) / total_val if total_val > 0 else 0
        precision_val = self.val_tp / (self.val_tp + self.val_fp) if (self.val_tp + self.val_fp) > 0 else 0
        recall_val = self.val_tp / (self.val_tp + self.val_fn) if (self.val_tp + self.val_fn) > 0 else 0
        f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

        # Log or print your metrics
        self.log('val_accuracy', accuracy_val, on_epoch=True, sync_dist=True)
        self.log('precision_val', precision_val, on_epoch=True, sync_dist=True)
        self.log('recall_val', recall_val, on_epoch=True, sync_dist=True)
        self.log('f1_score_val', f1_score_val, on_epoch=True, sync_dist=True)

        self.log('tp_val', self.val_tp, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('fp_val', self.val_fp, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('tn_val', self.val_tn, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('fn_val', self.val_fn, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('total_val', total_val, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)

        # Print for debugging
        print(f"Validation Accuracy: {accuracy_val}, Validation Precision: {precision_val}, Validation Recall: {recall_val}, Validation F1 Score: {f1_score_val}")
    
    def test_step(self, batch, batch_idx):
        loss_fn = self.loss_fn

        x, y = batch
        y_hat = self.forward(x)
        Test_step_loss = loss_fn(y_hat, y)

        # Apply softmax to get probabilities
        #y_pred_proba = torch.softmax(y_hat, dim=1)
        
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
        #print(f"Predicted: {predicted_classes}, True: {y}")

        return Test_step_loss
    
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

        self.log('tp', self.tp, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('fp', self.fp, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('tn', self.tn, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('fn', self.fn, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)
        self.log('total', total, on_epoch=True, sync_dist=True, reduce_fx=torch.sum)

        # Print for debugging
        print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")

    
    def predict_step(self, x):
        # TODO: Check this out
        x = self.forward(x)
        return x
    
    def configure_optimizers(self): # TODO: Implement scheduler
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        if self.scheduler_type is None:
            return optimizer
        else:
            # Initialize the learning rate scheduler based on the specified type
            if self.scheduler_type == 'StepLR':
                scheduler = lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
            elif self.scheduler_type == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
            elif self.scheduler_type == 'ExponentialLR':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            else:
                print(f"Unknown scheduler type: {self.scheduler_type}")
                print(f"Using default optimizer: Adam with lr={self.learning_rate}")
                return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # or 'step', depending on the scheduler
                'frequency': 1,       # How often to call the scheduler (every epoch or every step)
            },
        }

######################################################################################################################

# TODO : REMOVE BELOW, JUST FOR TESTING

# from torchvision.transforms import pad_to_shape

def main():
    model = ResNet(block=Block, layers=[2,2,1,1], num_classes=2, image_channels=1, 
                padding_layer_sizes=(2,2,4,3,7,7)) # TODO: Make this padding adaptive and change it to 48 48 48, but stride 1 for z before
                # pad_to_shape=(2, 64, 48, 48, 48))
    #print(model)

    tensor = torch.randn(2, 1, 35, 331, 216) # Batch, Channel, Depth, Height, Width
    # torch.randn(2,1,35,331,179)
    output = model(tensor)
    print(output.shape)
    #print(output)

# Before avg pool torch.Size([2, 1024, 5, 10, 11])
# After avg pool torch.Size([2, 1024, 1, 1, 1])
# Reshaping it to torch.Size([2, 1024])
# Using a linear layer to get torch.Size([2, 2])

# // Input shape: torch.Size([2, 1, 34, 164, 174])
# After Conv1: torch.Size([2, 64, 17, 82, 87])
# After MaxPool: torch.Size([2, 64, 9, 42, 44])
# After padding: torch.Size([2, 64, 48, 48, 48])
# After Conv2_x: torch.Size([2, 256, 48, 48, 48])
# After Conv3_x: torch.Size([2, 512, 24, 24, 24])
# After Conv4_x: torch.Size([2, 1024, 12, 12, 12])
# After Conv5_x: torch.Size([2, 2048, 6, 6, 6])
# After AvgPool: torch.Size([2, 2048, 1, 1, 1])
# After Reshape: torch.Size([2, 2048])
# After FC: torch.Size([2, 2])
# torch.Size([2, 2])

if __name__ == '__main__':
    main()