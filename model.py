import torch.nn as nn
from torchvision import models

class BlurClassifier(nn.Module):
    def __init__(self, dropout_rate):
        super(BlurClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

# Commit message: "Define model architecture for blur classification"