"""
CropGuard AI — shared preprocessing (training/serving parity).
================================================================================
The inference transform MUST match the evaluation transform used in training,
so local inference / the API reproduce the validated accuracy. Imported by both
infer.py and the FastAPI service (app.py) so preprocessing can never drift
between train and serve.

The LeViT eval transform (from train_levit.py) is:
    Resize((img_size, img_size))  ->  ToTensor  ->  Normalize(mean, std)

The exact mean/std/img_size are read from the model checkpoint and passed in, so
serving always matches the model that was trained. No leaf segmentation — the
field-training pipeline does not segment (segmentation reintroduced a background
shortcut and blacked out diseased tissue).
"""

from torchvision import transforms

# ImageNet defaults — used when a checkpoint doesn't specify its own.
IMG_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_eval_transform(mean=NORM_MEAN, std=NORM_STD, img_size=IMG_SIZE):
    """Square-resize eval transform for the model's own normalization / input size.
    The LeViT backend reads mean/std/img_size from its checkpoint and passes them
    here, so serving matches train_levit.py's eval transform exactly. With no args
    it reproduces the ImageNet-224 default."""
    return transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])
