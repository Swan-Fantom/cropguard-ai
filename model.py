"""
CropGuard AI — reusable model core (load once, predict many).
================================================================================
Single source of truth for building the trained model and running inference.
Both the CLI (infer.py) and the FastAPI service (app.py) import from here, so the
architecture, checkpoint-loading, and prediction logic can never drift apart.

The served model is a field-trained timm **LeViT** (see train_levit.py), loaded
from a self-describing dict checkpoint via build_levit() / CropGuardModel.load_levit().
No class_names.json or repo clone is needed — the class order (which MUST equal the
training label indices) and the exact normalization are read from the checkpoint
itself, which also sidesteps the classic "class order" gotcha.
"""

from PIL import Image
import torch

from preprocessing import build_eval_transform


def pretty_label(raw: str) -> str:
    """Turn a raw class folder name into a human-friendly label for the UI.

    'Common_Rust'           -> 'Common Rust'
    'Northern_Leaf_Blight'  -> 'Northern Leaf Blight'
    'Healthy'               -> 'Healthy'
    Also tolerates PlantVillage-style names with a genus prefix:
    'Corn_(maize)___Common_rust_' -> 'Common rust'
    """
    s = raw
    if "___" in s:                       # drop any leading crop/genus prefix
        s = s.split("___", 1)[1]
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())              # collapse repeated whitespace
    return (s[:1].upper() + s[1:]) if s else raw


def build_levit(checkpoint_path: str, device: torch.device, verbose: bool = True):
    """Build the field-trained timm LeViT and load its weights from the dict
    checkpoint written by train_levit.py.

    That checkpoint is a dict:
        {state_dict, classes, model_name, num_classes, mean, std, img_size, ...}
    so we need neither a class_names.json nor a cloned repo: the class order and
    the exact normalization are read straight from the file the model was saved
    with. That bakes the correct class order into the artifact itself.

    Returns (model, class_names, mean, std, img_size).
    """
    import timm  # lazy import: only needed to build the model

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
        m = CropGuardModel.load_levit("levit_cropguard.pth")
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
    def load_levit(cls, checkpoint_path: str, device=None, verbose: bool = True):
        """Load the field-trained LeViT. Class order + normalization come from the
        checkpoint itself (no class_names.json needed)."""
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
