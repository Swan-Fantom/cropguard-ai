"""
CropGuard AI — Explainability (Grad-CAM for Swin), from scratch (Step 3).
================================================================================
Turns a prediction into a *reason*: a heatmap over the leaf showing which regions
pushed the model toward its answer. This is the feature that makes CropGuard more
than "just a classifier" — you can visually check the model is looking at the
actual lesions, not the background.

WHY GRAD-CAM (not attention rollout) for Swin:
  Swin uses *windowed* attention, so raw attention maps are local and awkward to
  stitch into one full-image explanation. Grad-CAM sidesteps that: it asks "how
  much does each feature-map location, weighted by its gradient w.r.t. the chosen
  class, support that class?" It works on any conv/transformer feature map.

HOW IT WORKS HERE (the from-scratch recipe):
  1. Hook a Swin stage's last block (its `norm1`) to grab that layer's output
     activations on the forward pass.
  2. Do a forward pass, pick the target class (top-1 by default), and backprop
     that class score so we also capture the gradients at that same layer.
  3. Swin tokens are a flattened grid, so reshape (1, L, C) back to (side, side, C).
  4. Global-average-pool the gradients over the grid -> one weight per channel.
  5. Weighted sum of the activation channels -> ReLU -> normalize to [0,1].
  6. Upsample the small map to the image size, colorize (jet), and alpha-blend it
     over the original photo. Return it as a base64 PNG data URL.

RESOLUTION / the `stage` knob:
  Swin-Tiny's stages get smaller and more semantic as you go deeper:
    stage 0: 56x56, stage 1: 28x28, stage 2 (penultimate): 14x14, stage 3 (last): 7x7.
  Reading Grad-CAM off the LAST stage (7x7) is the most "semantic" but very coarse
  — you get a soft blob, not lesion-tight detail. The PENULTIMATE stage (14x14)
  localizes finer at a small cost in semantics. "combined" averages both for a
  balance. Pick per-image with the `stage` argument ("last" | "penultimate" |
  "combined"); different photos localize best at different depths.

Only numpy + Pillow + torch are needed (no extra dependencies). Torch is imported
lazily inside the functions that use it, so the image helpers below can be unit-
tested without a torch install.
"""

import base64
import io

import numpy as np
from PIL import Image


VALID_STAGES = ("last", "penultimate", "combined")


# --- Locating the layer(s) to explain ----------------------------------------
def _is_swin(model):
    """True if this looks like the Microsoft Swin model (has stage.blocks.norm1)."""
    try:
        _ = model.layers[-1].blocks[-1].norm1
        return True
    except (AttributeError, IndexError, TypeError):
        return False


def _discover_token_layers(model, example_input):
    """Auto-locate a model's token-grid layers for Grad-CAM by watching one forward
    pass. Records every LEAF module whose output is a token tensor (1, N, C) with N
    a perfect square (i.e. a reshapeable H×W grid), then keeps, for each distinct
    grid size, the DEEPEST such module (last in forward order). Returns those
    representatives ordered shallow -> deep.

    This adapts to the installed timm version instead of hardcoding a module path
    (LeViT's internal names shift between versions). It's the same detection
    `levit_inspect.py` uses. Result is cached on the model so we pay the cost once.

    For LeViT-256 @ 224 the grids are 14×14 (196) -> 7×7 (49) -> 4×4 (16), so the
    deepest representative (4×4, most semantic) maps to "last" and 7×7 to
    "penultimate" — directly analogous to Swin's last/penultimate stages."""
    import torch

    cached = getattr(model, "_cropguard_token_layers", None)
    if cached:
        return cached

    records = []  # (order_index, module, side)
    handles = []
    order = {"i": 0}

    def make_hook(mod):
        def hook(m, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                n = out.shape[1]
                side = int(round(n ** 0.5))
                if side >= 2 and side * side == n:
                    records.append((order["i"], m, side))
                    order["i"] += 1
        return hook

    for name, mod in model.named_modules():
        if name == "" or any(True for _ in mod.children()):
            continue  # leaf modules only
        handles.append(mod.register_forward_hook(make_hook(mod)))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()

    if not records:
        raise RuntimeError(
            "Could not auto-locate any (1, N, C) square token-grid layers for this "
            "model. Run `python levit_inspect.py` and inspect levit_layers.txt, then "
            "wire get_target_layers() in explain.py manually."
        )

    # Keep the deepest (last-in-forward-order) module for each distinct grid size.
    reps = {}  # side -> (order_index, module); later writes overwrite -> keeps deepest
    for idx, mod, side in records:
        reps[side] = (idx, mod)
    ordered = [mod for _, mod in sorted(reps.values(), key=lambda t: t[0])]  # shallow -> deep

    model._cropguard_token_layers = ordered
    return ordered


def get_target_layers(model, stage="last", example_input=None):
    """Return the list of layers to hook for the requested resolution `stage`:
      - "last"        -> deepest token-grid stage   (coarsest, most semantic)
      - "penultimate" -> next stage up              (finer detail)
      - "combined"    -> both of the above (their CAMs get merged)
    Returns a list so the extractor can hook one or several layers uniformly.

    Swin uses its known stage layout directly. Any other backend (e.g. the
    field-trained LeViT) is auto-discovered from a dummy forward pass, which needs
    `example_input` (a preprocessed (1,3,H,W) tensor on the model's device)."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage '{stage}'; use one of {VALID_STAGES}.")

    if _is_swin(model):
        last = model.layers[-1].blocks[-1].norm1
        penult = model.layers[-2].blocks[-1].norm1
    else:
        if example_input is None:
            raise RuntimeError(
                "Auto-discovering Grad-CAM layers for a non-Swin model needs an "
                "example input tensor; call get_target_layers(model, stage, example_input=x)."
            )
        ordered = _discover_token_layers(model, example_input)  # shallow -> deep
        last = ordered[-1]
        penult = ordered[-2] if len(ordered) >= 2 else ordered[-1]

    if stage == "last":
        return [last]
    if stage == "penultimate":
        return [penult]
    return [penult, last]  # "combined"


# --- Grad-CAM extractor -------------------------------------------------------
class TokenGradCAM:
    """Registers hooks on one OR several layers to capture activations + gradients,
    then builds a class-activation map (averaging across layers when given more
    than one). Create one, call it, then call .remove()."""

    def __init__(self, model, target_layers):
        self.model = model
        if not isinstance(target_layers, (list, tuple)):
            target_layers = [target_layers]
        self.target_layers = list(target_layers)
        n = len(self.target_layers)
        self.activations = [None] * n
        self.gradients = [None] * n
        self._handles = [
            layer.register_forward_hook(self._make_forward_hook(i))
            for i, layer in enumerate(self.target_layers)
        ]

    def _make_forward_hook(self, i):
        def hook(module, inputs, output):
            # Save the forward activations for layer i, and register a hook to
            # catch the gradient flowing back into this exact tensor in backprop.
            self.activations[i] = output
            if output.requires_grad:
                output.register_hook(lambda grad, i=i: self._save_grad(i, grad))
        return hook

    def _save_grad(self, i, grad):
        self.gradients[i] = grad

    def remove(self):
        for h in self._handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        import torch  # lazy: keeps the image helpers importable without torch

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)                 # (1, num_classes), grad ON
        if isinstance(logits, (tuple, list)):             # distilled LeViT -> (cls, dist)
            logits = logits[0]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        cams = []
        for a, g in zip(self.activations, self.gradients):
            if a is None or g is None:
                raise RuntimeError(
                    "Grad-CAM did not capture activations/gradients. Make sure the "
                    "forward pass is NOT under torch.no_grad() and the target "
                    "layer(s) are on the path to the output."
                )
            cams.append(self._compute_cam(a, g))
        cam = _merge_cams(cams)
        return cam, class_idx, logits.detach()

    @staticmethod
    def _compute_cam(activations, gradients):
        """activations, gradients: torch tensors shaped (1, L, C) where L is a
        square number of tokens. Returns an (H, W) numpy CAM normalized to [0,1]."""
        a = activations.detach().cpu().numpy()[0]         # (L, C)
        g = gradients.detach().cpu().numpy()[0]           # (L, C)
        L, C = a.shape
        side = int(round(L ** 0.5))
        if side * side != L:
            raise RuntimeError(f"Token count {L} is not a square grid; cannot reshape for Grad-CAM.")
        a = a.reshape(side, side, C)                      # (H, W, C), tokens are row-major
        g = g.reshape(side, side, C)
        weights = g.mean(axis=(0, 1))                     # (C,)  global-average-pooled grads
        cam = (a * weights[None, None, :]).sum(axis=-1)   # (H, W)
        cam = np.maximum(cam, 0.0)                        # ReLU: keep only positive support
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)                         # normalize to [0, 1]
        return cam


def _merge_cams(cams):
    """Combine one or more [0,1] CAMs of possibly different grid sizes into a
    single normalized map. Coarser CAMs are bilinearly upsampled to the finest
    grid, then all are averaged and re-normalized."""
    if len(cams) == 1:
        return cams[0]
    target = max(c.shape[0] for c in cams)
    resized = []
    for c in cams:
        if c.shape[0] != target:
            im = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8), mode="L")
            c = np.asarray(im.resize((target, target), Image.BILINEAR)).astype(np.float32) / 255.0
        resized.append(c)
    cam = np.mean(resized, axis=0)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)
    return cam


# --- Turning a CAM into a viewable overlay (numpy + Pillow only) --------------
def _jet(cam01):
    """Map a HxW array in [0,1] to a HxWx3 'jet' colormap in [0,1]
    (dark blue -> cyan -> green -> yellow -> red)."""
    x = cam01
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def make_overlay(pil_image, cam, alpha=0.45):
    """Blend the CAM over the original image. Uses per-pixel alpha scaled by the
    CAM value, so cool (unimportant) regions stay close to the original photo and
    hot regions get colored — clearer than a flat tint for spotting lesions."""
    pil_image = pil_image.convert("RGB")
    w0, h0 = pil_image.size

    # Upsample the small CAM (e.g. 7x7 / 14x14) to full image size, smoothly.
    cam_small = Image.fromarray((np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")
    cam_up = np.asarray(cam_small.resize((w0, h0), Image.BILINEAR)).astype(np.float32) / 255.0

    heat = _jet(cam_up)                                   # (H, W, 3) in [0,1]
    base = np.asarray(pil_image).astype(np.float32) / 255.0
    a = (alpha * cam_up)[..., None]                       # importance-weighted alpha
    out = (1.0 - a) * base + a * heat
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGB")


def to_data_url(pil_image, fmt="PNG"):
    """Encode a PIL image as a data: URL so a browser can render it directly."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


# --- Public entry point -------------------------------------------------------
def explain_image(cg_model, pil_image, topk=3, alpha=0.45, class_idx=None, stage="last"):
    """Run prediction + Grad-CAM on a PIL image using a loaded CropGuardModel.

    stage: "last" (7x7, coarse/semantic), "penultimate" (14x14, finer), or
           "combined" (both, averaged). See module docstring.

    Returns a dict:
        prediction : raw top-1 class name (or the requested class)
        class_idx  : integer label index explained
        top        : [(class_name, probability), ...] length <= topk
        stage      : the stage actually used
        heatmap    : 'data:image/png;base64,...' overlay ready for an <img> tag
    """
    import torch  # lazy

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    x = cg_model.transform(pil_image).unsqueeze(0).to(cg_model.device)

    target_layers = get_target_layers(cg_model.model, stage, example_input=x)
    cam_extractor = TokenGradCAM(cg_model.model, target_layers)
    try:
        cam, class_idx, logits = cam_extractor(x, class_idx=class_idx)
    finally:
        cam_extractor.remove()                            # never leak hooks

    probs = torch.softmax(logits, dim=1)[0]
    k = min(topk, len(cg_model.class_names))
    top_probs, top_idx = probs.topk(k)
    top = [(cg_model.class_names[int(i)], float(p)) for p, i in zip(top_probs, top_idx)]

    overlay = make_overlay(pil_image, cam, alpha=alpha)
    return {
        "prediction": cg_model.class_names[class_idx],
        "class_idx": class_idx,
        "top": top,
        "stage": stage,
        "heatmap": to_data_url(overlay),
    }
