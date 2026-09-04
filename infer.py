"""
CropGuard AI — Standalone single-image inference (CLI).
================================================================================
STEP 1 of the CropGuard build. A thin command-line wrapper around model.py so
you can sanity-check the trained model from a terminal. The FastAPI service
(app.py) uses the exact same model.py core, so CLI and API always agree.

The default backend is the field-trained LeViT (pure timm, no repo clone needed).
The optional Swin backend needs the Microsoft repo cloned into this folder:
    git clone https://github.com/microsoft/Swin-Transformer.git   # swin backend only

Usage (LeViT — the current model; classes come from the checkpoint):
    python infer.py --image leaf.jpg

Usage (old Swin model):
    python infer.py --backend swin --image leaf.jpg --checkpoint swin_cropguard.pth \
        --classes class_names.json --swin-repo Swin-Transformer

Install:
    pip install -r requirements.txt
"""

import argparse

import torch
from PIL import Image

from model import CropGuardModel


def main():
    parser = argparse.ArgumentParser(description="CropGuard Swin single-image inference")
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--backend", choices=("levit", "swin"), default="levit",
                        help="Which trained model to load (default: levit)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained .pth weights (default: levit_cropguard.pth for "
                             "levit, swin_cropguard.pth for swin)")
    parser.add_argument("--classes", default="class_names.json",
                        help="class_names.json (swin only; levit reads classes from its checkpoint)")
    parser.add_argument("--swin-repo", default="Swin-Transformer",
                        help="Path to the cloned microsoft/Swin-Transformer repo (swin only)")
    parser.add_argument("--topk", type=int, default=3, help="How many top predictions to show")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}  backend: {args.backend}")

    checkpoint = args.checkpoint or (
        "levit_cropguard.pth" if args.backend == "levit" else "swin_cropguard.pth")
    if args.backend == "levit":
        model = CropGuardModel.load_levit(checkpoint, device)
    else:
        model = CropGuardModel.load(checkpoint, args.classes, args.swin_repo, device)
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
