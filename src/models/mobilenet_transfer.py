"""
MobileNetV2 Transfer Learning
===============================
Fine-tuned MobileNetV2 for traffic sign classification.
Designed for deployment efficiency with fewer parameters.
Achieved 95.96% test accuracy on GTSRB (v2 with more unfrozen layers).
"""

import torch.nn as nn
from torchvision import models


class MobileNetTransfer(nn.Module):
    """MobileNetV2 with modified classifier for GTSRB.

    Only the first ~20 parameters are frozen.
    Later layers are trainable with a lower learning rate.

    Args:
        num_classes: number of output classes (default: 43 for GTSRB)
    """

    def __init__(self, num_classes=43):
        super(MobileNetTransfer, self).__init__()

        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze only early layers
        for i, (name, param) in enumerate(self.mobilenet.features.named_parameters()):
            if i < 20:
                param.requires_grad = False

        # Replace classifier head
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)

    def get_param_groups(self, lr_pretrained=0.0001, lr_new=0.001):
        """Return parameter groups for differential learning rates."""
        pretrained_params = []
        new_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        return [
            {'params': pretrained_params, 'lr': lr_pretrained},
            {'params': new_params, 'lr': lr_new}
        ]
