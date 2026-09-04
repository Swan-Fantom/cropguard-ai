"""
CropGuard AI — Standalone single-image inference (CLI).
================================================================================
A thin command-line wrapper around model.py so you can sanity-check the trained
model from a terminal. The FastAPI service (app.py) uses the exact same model.py
core, so CLI and API always agree.

The served model is the field-trained timm LeViT (pure timm, no repo clone
needed); the class names come from the checkpoint itself.

Usage:
    python infer.py --image leaf.jpg
    python infer.py --image leaf.jpg --checkpoint levit_cropguard.pth --topk 3

Install:
    pip install -r requirements.txt
"""

import argparse

import torch
from PIL import Image

from model import CropGuardModel


def main():
    parser = argparse.ArgumentParser(description="CropGuard LeViT single-image inference")
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--checkpoint", default="levit_cropguard.pth",
                        help="Path to the trained LeViT .pth checkpoint "
                             "(default: levit_cropguard.pth)")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    model = CropGuardModel.load_levit(args.checkpoint, device)
    print(f"[info] {model.num_classes} classes: {model.class_names}")

    image = Image.open(args.image).convert("RGB")
    results = model.predict(image, args.topk)

    print("\nPrediction")
    print("-" * 52)
    for rank, (name, p) in enumerate(results, 1):
        bar = "#" * int(round(p * 30))
        print(f"  {rank}. {name:<28} {p * 100:5.1f}%  {bar}")

    top_name, top_p = results[0]
    if top_p < 0.50:
        print("\n[note] Low confidence (<50%). Try a clearer, well-lit photo of a "
              "single leaf filling the frame.")


if __name__ == "__main__":
    main()
