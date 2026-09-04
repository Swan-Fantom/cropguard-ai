"""
CropGuard AI — reusable model core (load once, predict many).
================================================================================
Single source of truth for building the trained model(s) and running inference.
Both the CLI (infer.py) and the FastAPI service (app.py) import from here, so the
architecture, checkpoint-loading, and prediction logic can never drift apart.

Two backends are supported:
  - LeViT (the current served model): the field-trained timm model from
    train_levit.py, loaded from a self-describing dict checkpoint via build_levit()
    / CropGuardModel.load_levit(). No class_names.json or repo clone needed — the
    class order and normalization are read from the checkpoint itself.
  - Swin-Tiny: the original from-scratch Microsoft model, via build_model() / .load().

Your Swin model is the OFFICIAL Microsoft Swin-Transformer (NOT timm), trained
FROM SCRATCH as a Swin-Tiny on the 4-class corn dataset. Inference therefore:
  (a) builds the model from the SAME repo's SwinTransformer class with identical
      architecture params (SWIN_KWARGS below), and
  (b) preprocesses images exactly like your val_transform (see preprocessing.py).

One-time setup (clone the same model code you trained with, into this folder):
    git clone https://github.com/microsoft/Swin-Transformer.git
"""

import json
import os
import sys

import torch
from PIL import Image

from preprocessing import build_val_transform, build_eval_transform

# --- Architecture used in training (Swin-Tiny). MUST match your notebook. -----
# From your training call: embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
# window_size=7, drop_path_rate=0.2, ape=False, patch_norm=True.
# (Your notebook passed `embed_dims=[...]` which the repo ignores via **kwargs,
#  leaving the default embed_dim=96 — i.e. a standard Swin-Tiny.)
SWIN_KWARGS = dict(
    img_size=224,
    patch_size=4,
    in_chans=3,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_path_rate=0.2,
    ape=False,
    patch_norm=True,
)


def load_class_names(path: str) -> list:
    """Ordered list of class names; index i must equal the integer label the
    model was trained on. Exported from train_data.classes — see GETTING_STARTED.md."""
    with open(path) as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(
            "class_names.json must be a JSON list of strings, ordered by "
            "training label index, e.g. [\"Corn_Blight\", \"Corn_Common_Rust\", "
            "\"Corn_Gray_Leaf_Spot\", \"Corn_Healthy\"]"
        )
    return names


def pretty_label(raw: str) -> str:
    """Turn a raw PlantVillage class name into a human-friendly label for the UI.

    'Corn_(maize)___Common_rust_'      -> 'Common rust'
    'Corn_(maize)___Northern_Leaf_Blight' -> 'Northern Leaf Blight'
    'Corn_(maize)___healthy'           -> 'Healthy'
    """
    s = raw
    if "___" in s:                       # drop the leading crop/genus prefix
        s = s.split("___", 1)[1]
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())              # collapse repeated whitespace
    return (s[:1].upper() + s[1:]) if s else raw


def clean_state_dict(checkpoint) -> dict:
    """Make loading 'just work' across the common ways people save checkpoints."""
    state_dict = checkpoint
    # Unwrap {'model_state_dict': ...} / {'state_dict': ...} / {'model': ...} saves.
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                state_dict = checkpoint[key]
                break
    # Strip a 'module.' prefix left behind by nn.DataParallel / DDP training.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def import_swin(swin_repo: str):
    """Import SwinTransformer from the cloned Microsoft repo.

    We add <repo>/models to sys.path and import the module file directly, which
    avoids the package's __init__.py (it pulls in MoE/SimMIM variants and extra
    dependencies we don't need). swin_transformer.py only needs torch + timm.
    """
    models_dir = os.path.join(swin_repo, "models")
    if not os.path.isfile(os.path.join(models_dir, "swin_transformer.py")):
        raise FileNotFoundError(
            f"Could not find models/swin_transformer.py under '{swin_repo}'.\n"
            "Clone the repo into this folder first:\n"
            "    git clone https://github.com/microsoft/Swin-Transformer.git"
        )
    if models_dir not in sys.path:
        sys.path.insert(0, models_dir)
    from swin_transformer import SwinTransformer  # noqa: E402
    return SwinTransformer


def build_model(checkpoint_path: str, num_classes: int, device: torch.device,
                swin_repo: str, verbose: bool = True):
    """Construct the Swin-Tiny and load the trained weights onto `device`."""
    SwinTransformer = import_swin(swin_repo)
    model = SwinTransformer(num_classes=num_classes, **SWIN_KWARGS)

    raw = torch.load(checkpoint_path, map_location="cpu")
    state_dict = clean_state_dict(raw)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        if missing:
            print(f"[warn] {len(missing)} missing key(s), e.g. {missing[:4]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected key(s), e.g. {unexpected[:4]}")
        if not missing and not unexpected:
            print("[info] weights loaded cleanly.")

    model.eval().to(device)
    return model


def build_levit(checkpoint_path: str, device: torch.device, verbose: bool = True):
    """Build the field-trained timm LeViT and load its weights from the dict
    checkpoint written by train_levit.py.

    That checkpoint is a dict:
        {state_dict, classes, model_name, num_classes, mean, std, img_size, ...}
    so, unlike the Swin path, we need neither a class_names.json nor a cloned repo:
    the class order (which MUST equal the training label indices) and the exact
    normalization are read straight from the file the model was saved with. That
    also sidesteps the recurring 'class order' gotcha — it's baked into the file.

    Returns (model, class_names, mean, std, img_size).
    """
    import timm  # lazy: only needed for the LeViT backend

    # weights_only=False: this is your own trusted checkpoint, and it stores plain
    # Python objects (the class list, mean/std) alongside the tensors. Fall back
    # for older torch that doesn't have the kwarg.
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(
            f"'{checkpoint_path}' is not a LeViT checkpoint from train_levit.py "
            "(expected a dict with a 'state_dict' key)."
        )
    class_names = ckpt.get("classes")
    if not class_names:
        raise ValueError(
            "LeViT checkpoint is missing the 'classes' list — re-run train_levit.py, "
            "which saves the class order needed to map outputs to disease names."
        )
    model_name = ckpt.get("model_name", "levit_256")
    num_classes = int(ckpt.get("num_classes", len(class_names)))
    mean = tuple(ckpt.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(ckpt.get("std", (0.229, 0.224, 0.225)))
    img_size = int(ckpt.get("img_size", 224))

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if verbose:
        if missing:
            print(f"[warn] {len(missing)} missing key(s), e.g. {missing[:4]}")
        if unexpected:
            print(f"[warn] {len(unexpected)} unexpected key(s), e.g. {unexpected[:4]}")
        if not missing and not unexpected:
            print(f"[info] LeViT ({model_name}) weights loaded cleanly.")
    model.eval().to(device)
    return model, list(class_names), mean, std, img_size


class CropGuardModel:
    """Loaded model + its transform + class names. Load once, call .predict() many.

    Typical use:
        m = CropGuardModel.load("swin_cropguard.pth", "class_names.json")
        results = m.predict(pil_image, topk=3)   # [(class_name, probability), ...]
    """

    def __init__(self, model, transform, class_names, device):
        self.model = model
        self.transform = transform
        self.class_names = class_names
        self.device = device

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @classmethod
    def load(cls, checkpoint_path: str, classes_path: str,
             swin_repo: str = "Swin-Transformer", device=None, verbose: bool = True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = load_class_names(classes_path)
        model = build_model(checkpoint_path, len(class_names), device, swin_repo, verbose=verbose)
        transform = build_val_transform()
        return cls(model, transform, class_names, device)

    @classmethod
    def load_levit(cls, checkpoint_path: str, device=None, verbose: bool = True):
        """Load the field-trained LeViT. Class order + normalization come from the
        checkpoint itself (no class_names.json, no Swin repo needed)."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, class_names, mean, std, img_size = build_levit(checkpoint_path, device, verbose=verbose)
        transform = build_eval_transform(mean, std, img_size)
        return cls(model, transform, class_names, device)

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 3):
        """Return the top-k [(class_name, probability), ...] for a PIL image."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)   # (1, 3, H, W)
        out = self.model(x)
        if isinstance(out, (tuple, list)):   # some timm variants (distilled) return (cls, dist)
            out = out[0]
        probs = torch.softmax(out, dim=1)[0]
        k = min(topk, self.num_classes)
        top_probs, top_idx = probs.topk(k)
        return [(self.class_names[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]
