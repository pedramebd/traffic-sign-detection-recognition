"""
ResNet-18 Transfer Learning
============================
Fine-tuned ResNet-18 for traffic sign classification.
Uses differential learning rates — lower for pretrained layers, higher for new classifier.
Achieved 97.51% test accuracy on GTSRB (v2 with more unfrozen layers).
"""

import torch.nn as nn
from torchvision import models


class ResNetTransfer(nn.Module):
    """ResNet-18 with modified classifier for GTSRB.

    Only the first conv layer and batch norm are frozen.
    All residual blocks are trainable with a lower learning rate.

    Args:
        num_classes: number of output classes (default: 43 for GTSRB)
    """

    def __init__(self, num_classes=43):
        super(ResNetTransfer, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Freeze only the very first conv + bn layer
        for name, param in self.resnet.named_parameters():
            if ('layer1' not in name and 'layer2' not in name and
                    'layer3' not in name and 'layer4' not in name and
                    'fc' not in name):
                param.requires_grad = False

        # Replace classifier head
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_param_groups(self, lr_pretrained=0.0001, lr_new=0.001):
        """Return parameter groups for differential learning rates."""
        pretrained_params = []
        new_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'fc' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        return [
            {'params': pretrained_params, 'lr': lr_pretrained},
            {'params': new_params, 'lr': lr_new}
        ]
