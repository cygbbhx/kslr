import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models
from model.i3d import InceptionI3d
from model.p3d import P3D199
import torch

class MnistModel(BaseModel):
    def __init__(self, num_classes=60):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ResNet18(BaseModel):
    def __init__(self, num_classes=60):
        super().__init__()
        self.resnet18_pretrained = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18_pretrained.fc.in_features
        self.resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.resnet18_pretrained(x)
        return F.log_softmax(x, dim=1)

class I3D(BaseModel):
    def __init__(self, num_classes=60):
        super().__init__()
        self.i3d = InceptionI3d(pretrained=True, pretrained_path='model/pretrained/rgb_imagenet.pt', num_frames=32)
        self.i3d.replace_logits(num_classes=num_classes)

    def forward(self, x):
        x = self.i3d(x)
        return F.log_softmax(x, dim=1)

class P3D(BaseModel):
    def __init__(self, num_classes=60):
        super().__init__()
        self.model = P3D199(pretrained=False,num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

class Conv1d_LSTM(nn.Module):
    def __init__(self,in_channels=128, num_classes=60):
        super(Conv1d_LSTM2, self).__init__()
        # Adjusted in_channels to 137 to match dataset output
        self.conv1d_1 = nn.Conv1d(in_channels=in_channels, 
                                  out_channels=16, 
                                  kernel_size=5, 
                                  stride=1, 
                                  padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=16, 
                                  out_channels=32, 
                                  kernel_size=5, 
                                  stride=1, 
                                  padding=1)
        
        self.lstm = nn.LSTM(input_size=32, 
                            hidden_size=50, 
                            num_layers=1, 
                            bias=True, 
                            bidirectional=False, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)

        self.dense1 = nn.Linear(50, 32)
        # Adjusted out_features to num_classes
        self.dense2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # No need to reshape x as it already matches the expected shape
        # Proceed with 1D convolutions
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        # Transpose to match LSTM input requirements
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]

        x = self.dropout(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return F.log_softmax(x, dim=1)


class LSTMmodel(nn.Module):
    def __init__(self,in_channels=128, num_classes=60):
        super(LSTMmodel, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, 
                            hidden_size=50, 
                            num_layers=1, 
                            bias=True, 
                            bidirectional=False, 
                            batch_first=True)
        
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(50, 32)
        self.dense2 = nn.Linear(32, num_classes)

    def forward(self, x):
        lstm_out, (hidden, _) = self.lstm(x)

        # Optionally, you can use the output of the last time step from LSTM for further processing
        # For instance, if you're interested in the last time step's output:
        # x = lstm_out[:, -1, :]

        # If you want to use the last hidden state (which is common in many applications):
        x = hidden[-1]

        # Pass through dropout layer
        x = self.dropout(x)

        # Pass through fully connected layers
        x = self.dense1(x)
        x = self.dense2(x)
        return F.log_softmax(x, dim=1)
