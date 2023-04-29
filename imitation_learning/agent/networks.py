import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary import summary

"""
Imitation learning network
"""

class CNN(nn.Module):

    def __init__(self, history_length=1, n_classes=5):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=history_length, out_channels=24, kernel_size=5),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16384, 1164),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1164, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(100, n_classes))

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def summary(self, input_size):
        summary(self.model, input_size, device='cuda' if torch.cuda.is_available() else 'cpu')


"""
model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(5184, 1024),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, n_classes))
"""