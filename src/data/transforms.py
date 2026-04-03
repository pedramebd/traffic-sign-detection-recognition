"""
Data Transforms
================
Training and evaluation transforms for GTSRB classification.
Training includes augmentation; validation/test does not.
"""

from torchvision import transforms

IMG_SIZE = 48

# ImageNet normalisation (used because transfer learning models expect it)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(img_size=IMG_SIZE):
    """Training transform with data augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transform(img_size=IMG_SIZE):
    """Validation/test transform without augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inverse_normalize():
    """Inverse normalisation for visualising images."""
    return transforms.Normalize(
        mean=[-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
        std=[1.0 / s for s in IMAGENET_STD]
    )
