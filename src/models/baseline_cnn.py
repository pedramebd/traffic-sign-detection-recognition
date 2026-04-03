"""
Baseline CNN
=============
Custom 3-block convolutional neural network for traffic sign classification.
Achieved 98.23% test accuracy on GTSRB — outperforming transfer learning models.

Architecture:
    Block 1: 3→32 channels, Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout
    Block 2: 32→64 channels, Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout
    Block 3: 64→128 channels, Conv-BN-ReLU-Conv-BN-ReLU-MaxPool-Dropout
    Classifier: FC(128*6*6, 512)-ReLU-Dropout-FC(512, 43)
"""

import torch.nn as nn


class BaselineCNN(nn.Module):
    """Custom CNN for GTSRB traffic sign classification.

    Args:
        num_classes: number of output classes (default: 43 for GTSRB)
    """

    def __init__(self, num_classes=43):
        super(BaselineCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
