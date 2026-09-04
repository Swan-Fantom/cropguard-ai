"""
CropGuard AI — Kaggle export cell.
================================================================================
COPY THIS INTO A NEW CELL AT THE END OF YOUR SWIN KAGGLE NOTEBOOK and run it in
the SAME session where training just finished (so `model` and `class_names` are
still defined in memory — your notebook does not save a checkpoint during
training, so the trained weights only exist in the live `model` object).

It writes two files to /kaggle/working/ that you then download into this folder:
    swin_cropguard.pth   (trained Swin-Tiny weights)
    class_names.json     (disease labels in training-label order)
"""

import torch
import json
import os

os.makedirs("/kaggle/working", exist_ok=True)

# 1) Save the trained Swin weights (state_dict only — recommended).
torch.save(model.state_dict(), "/kaggle/working/swin_cropguard.pth")

# 2) Save class names in EXACT training-label order.
#    `class_names` is already defined in your notebook as train_data.classes,
#    which ImageFolder orders alphabetically by folder name (the label order).
with open("/kaggle/working/class_names.json", "w") as f:
    json.dump(class_names, f)

# 3) Sanity checks — read these before downloading.
print("Saved weights -> /kaggle/working/swin_cropguard.pth")
print("Classes (index 0..N):", class_names)
print("num_classes check:", len(class_names), "should equal", NUM_CLASSES)
