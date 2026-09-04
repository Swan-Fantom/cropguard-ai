# CropGuard AI — Step 1: Get your trained Swin model out of Kaggle and predicting locally

You trained a **Swin-Tiny** using the **official Microsoft Swin-Transformer repo** (cloned in your notebook), **from scratch** (no pretrained weights), on the 4-class corn dataset. The goal of this step is small but foundational: **prove the trained model works outside the notebook.** Once a plain script loads your weights and classifies one image, wrapping it in an API and a web app is straightforward.

> **Two things are already matched to your training** (I read your notebook):
> - **Preprocessing** (`preprocessing.py`): `Resize((224,224))` → `ToTensor` → `Normalize(ImageNet mean/std)`. No segmentation. This is your exact `val_transform`.
> - **Architecture** (`infer.py`): Swin-Tiny with `embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=7, drop_path_rate=0.2`.
>
> You do **not** need to edit either.

---

## What you'll produce in this step

- `swin_cropguard.pth` — your trained Swin-Tiny weights, downloaded from Kaggle
- `class_names.json` — the 4 disease classes in training-label order
- A cloned `Swin-Transformer/` folder (the same model code you trained with)
- A working run of `infer.py` that prints a prediction for a test leaf image

---

## 1. Export the weights + class names from Kaggle

Open your Swin notebook. The export code is saved here as **`kaggle_export.py`** — copy its contents into a **new cell at the end of the notebook** and run it.

> **Run it in the same session where training just finished** — your notebook doesn't save a checkpoint during training (the loop has "no model saving"), so the trained weights only exist in the live `model` object. If the kernel has restarted, re-run the notebook to retrain (or run until `model` is trained and `class_names` is defined) before exporting.

The cell (for reference):

```python
import torch, json, os
os.makedirs("/kaggle/working", exist_ok=True)

torch.save(model.state_dict(), "/kaggle/working/swin_cropguard.pth")

# class_names is already defined in your notebook (= train_data.classes)
with open("/kaggle/working/class_names.json", "w") as f:
    json.dump(class_names, f)

print("Classes (index 0..N):", class_names)
print("num_classes check:", len(class_names), "should equal", NUM_CLASSES)
```

**Check the printed class list** — those four names, in that order, are what every prediction maps to.

## 2. Download the two files

In the Kaggle notebook editor, the right-hand **Output / `/kaggle/working`** panel lists files with a download button — download `swin_cropguard.pth` and `class_names.json` into this folder (`C:\College Materials\CropGuard`).

> **Size:** Swin-Tiny weights are ~110 MB. `.gitignore` already keeps `*.pth` out of git.

## 3. Clone the same model code (one-time)

Because you trained with the Microsoft repo's `SwinTransformer` class, we load the weights into that same class. From inside this folder:

```bash
git clone https://github.com/microsoft/Swin-Transformer.git
```

This creates a `Swin-Transformer/` subfolder. `infer.py` reads only `Swin-Transformer/models/swin_transformer.py` from it (it's `.gitignore`d, so it won't clutter your repo).

## 4. Install deps and run

From this folder:

```bash
pip install -r requirements.txt

python infer.py --image test_leaf.jpg --checkpoint swin_cropguard.pth --classes class_names.json --swin-repo Swin-Transformer
```

Expected output — something like:

```
[info] device: cpu
[info] 4 classes: ['Corn_Blight', 'Corn_Common_Rust', 'Corn_Gray_Leaf_Spot', 'Corn_Healthy']
[info] weights loaded cleanly.

Prediction
----------------------------------------------------
  1. Corn_Common_Rust             97.3%  #############################
  2. Corn_Blight                   1.8%  #
  3. Corn_Gray_Leaf_Spot           0.6%
```

(Your exact class names will be whatever the export cell printed.)

---

## How to know it worked

Pick a **test/validation image whose label you know** and run it. The top prediction should match with high confidence, and roughly agree with the test/val accuracy you saw in Kaggle across a few images. If it does, Step 1 is done.

> Tip for your resume later: the metrics your notebook already computes (accuracy, macro precision/recall/F1, per-class scores, confusion matrix, ROC-AUC) are exactly the numbers that make a strong project bullet. Note down your final **test** and **unseen-val** accuracy.

## If something looks off

- **`FileNotFoundError` about `models/swin_transformer.py`** → you haven't cloned the repo (Step 3), or `--swin-repo` points to the wrong path.
- **Import error on `timm.models.layers`** → your installed `timm` is too new. Either `pip install "timm==0.9.16"`, or edit `Swin-Transformer/models/swin_transformer.py` line `from timm.models.layers import ...` to `from timm.layers import ...`.
- **Lots of missing/unexpected keys when loading** → architecture mismatch. The params in `infer.py`'s `SWIN_KWARGS` must match your training call; if you changed anything (depths, embed_dim, window size), tell me and I'll adjust.
- **Predictions confident but wrong / random** → class order. Re-check the list printed by the export cell.

---

## What comes next

1. ✅ **You are here:** standalone inference works.
2. **FastAPI service** — wrap this in a `POST /predict` endpoint returning `{disease, confidence, top_k}`, loading the model once at startup and reusing `preprocessing.py` so train/serve stay identical.
3. **XAI heatmap** — attention-based explainability for Swin (attention rollout / `pytorch-grad-cam` with a reshape transform) so each prediction returns a heatmap of the region the model focused on. Your standout feature.
4. **Web app** — React + Tailwind upload UI; Node/Express for auth + prediction history in MongoDB.
5. **Containerize + deploy** — Docker, then Azure (ties to your Azure cert) or a free tier.

When `infer.py` prints a correct prediction on a real leaf image, tell me and we'll build **Step 2 (the FastAPI service)** together.
