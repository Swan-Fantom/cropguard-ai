"""
CropGuard AI — shared preprocessing (training/serving parity).
================================================================================
Matches the VALIDATION transform from your Kaggle training exactly, so local
inference / the API produce the same results as your validated accuracy. This
module is imported by infer.py now and will be reused by the FastAPI service
later so preprocessing can never drift between train and serve.

Your Kaggle val_transform was:
    Resize((224, 224))  ->  ToTensor  ->  Normalize(ImageNet mean/std)

No leaf segmentation — your real pipeline does not segment.
"""

from torchvision import transforms

IMG_SIZE = 224

# ImageNet normalization — the exact mean/std used in your training transforms.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def build_val_transform():
    """Validation/inference transform — matches training's val_transform exactly."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


def build_eval_transform(mean=NORM_MEAN, std=NORM_STD, img_size=IMG_SIZE):
    """Same square-resize eval transform, but for an arbitrary backbone's own
    normalization / input size. The LeViT backend reads mean/std/img_size from its
    checkpoint and passes them here, so serving matches train_levit.py's eval_tf
    exactly. With no args it reproduces build_val_transform() (ImageNet, 224)."""
    return transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        transforms.Normalize(list(mean), list(std)),
    ])
