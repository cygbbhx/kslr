import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models
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