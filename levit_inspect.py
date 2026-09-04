"""
CropGuard AI — one-off: dump LeViT's token-producing layers for Grad-CAM wiring.
================================================================================
Run this ONCE when you're ready to add the heatmap (/explain) for the LeViT model:

    python levit_inspect.py

It builds the SAME architecture the served checkpoint uses (levit_256) and records,
for a dummy 224x224 input, every leaf-module output shaped (1, N, C) where N is a
perfect square — i.e. a token grid we can turn into a heatmap. Those are the
candidate Grad-CAM targets; the last few (deepest, smallest grid) are the useful
ones — the deepest maps to the coarse/semantic "last" stage and the next one up to
the finer "penultimate" stage.

It prints the list and also saves it to `levit_layers.txt` in this folder, so it
can be read back to wire get_target_layers() in explain.py correctly (rather than
guessing a module path that differs between timm versions).

Needs only torch + timm (already in requirements.txt). No checkpoint needed — the
module structure is identical whether or not the trained weights are loaded.
"""

import math

import timm
import torch

MODEL_NAME = "levit_256"
NUM_CLASSES = 4


def is_square(n: int) -> bool:
    r = int(round(math.sqrt(n)))
    return r * r == n


def main():
    model = timm.create_model(MODEL_NAME, num_classes=NUM_CLASSES).eval()

    rows = []          # (name, shape, is_square_grid)
    handles = []
    for name, mod in model.named_modules():
        if name == "" or any(True for _ in mod.children()):
            continue   # only leaf modules (no children)

        def make_hook(n):
            def hook(m, inp, out):
                if isinstance(out, torch.Tensor) and out.dim() == 3:
                    _, N, C = out.shape
                    rows.append((n, tuple(out.shape), is_square(N)))
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(torch.randn(1, 3, 224, 224))
    for h in handles:
        h.remove()

    lines = [f"model={MODEL_NAME}  num_classes={NUM_CLASSES}  timm={timm.__version__}", ""]

    lines.append("SQUARE-GRID token outputs (Grad-CAM candidates; last = deepest/most semantic):")
    for n, shape, sq in rows:
        if sq:
            side = int(round(shape[1] ** 0.5))
            lines.append(f"  {shape}   grid={side}x{side}   {n}")

    lines.append("")
    lines.append("TOP-LEVEL children (stage structure):")
    for name, _ in model.named_children():
        lines.append(f"  {name}")

    lines.append("")
    lines.append("ALL (1,N,C) leaf outputs (for reference):")
    for n, shape, sq in rows:
        lines.append(f"  {shape}  square={sq}  {n}")

    text = "\n".join(lines)
    print(text)
    with open("levit_layers.txt", "w") as f:
        f.write(text + "\n")
    print("\n[saved] levit_layers.txt  — share this / let me read it to wire the heatmap.")


if __name__ == "__main__":
    main()
