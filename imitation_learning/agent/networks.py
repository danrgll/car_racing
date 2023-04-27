import torch.nn as nn
import torch
import torch.nn.functional as F

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=5):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(30976, 1024),
            nn.Linear(1024, n_classes))

    def forward(self, x):
        x = self.model.forward(x)
        return x

