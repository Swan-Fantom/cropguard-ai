"""
CropGuard AI — LeViT fine-tuning (robustness track), corrected training script.
================================================================================
Field-first fine-tune of a pretrained timm LeViT for corn-leaf disease
classification. This is the rewrite of the earlier notebook, with every fix from
the code review:

  1. NO segment_leaf(). The old transform HSV-masked the leaf and painted the
     background (and every non-green pixel) BLACK. That (a) reintroduced the
     black-background shortcut / Grad-CAM edge artifact we diagnosed in Step 3,
     and (b) blacked out the disease lesions themselves (rust / blight / gray
     leaf spot are not green), erasing the very signal to classify. We train on
     the raw resized leaf instead.
  2. Normalization matches the pretrained weights. mean/std are read from the
     model's own timm config (ImageNet), never hardcoded [0.5]. The eval
     transform is a plain square Resize -> ToTensor -> Normalize, identical to
     the serving preprocessing.py, so training and inference agree.
  3. Real evaluation. A field-only TEST split is scored with a confusion matrix
     and per-class precision / recall / F1 (not just accuracy) — this is how we
     judge the field-only-vs-combined ablation and watch the thin Gray-Leaf-Spot
     class (only ~25 field test images).
  4. Class imbalance handled. The ~6:1 rust:GLS imbalance is countered with a
     class-weighted loss so GLS recall is not traded away for overall accuracy.
  5. Best checkpoint saved. Tracks best validation macro-F1 and saves THAT
     state_dict (+ class order, mean/std, model name) so the served model is the
     best epoch, not the last. copy / torch.save are actually used now.

Extras: discriminative LR (backbone lower than head, so a small dataset does not
wash out the pretrained features), warmup that starts at 1/warmup instead of 0,
early stopping, and saved training curves.

DATA LAYOUT (torchvision ImageFolder; alphabetical class order == label index):
    DATA_DIR/
        train/<class>/*.jpg
        val/<class>/*.jpg      # field only
        test/<class>/*.jpg     # field only

Val and test MUST stay field-only. If you run the ablation that adds laboratory
images, add them to TRAIN ONLY via EXTRA_TRAIN_DIR below — never to val/test.

RUN (Kaggle GPU):
    pip install timm scikit-learn
    CROPGUARD_DATA_DIR=/kaggle/input/<your-field-dataset> python train_levit.py

OUTPUTS (written to OUT_DIR):
    levit_cropguard.pth        # {state_dict, classes, mean, std, model_name, ...}
    levit_class_names.json     # class order, for the serving side
    levit_confusion_matrix.png
    levit_test_report.txt / .json
    levit_training_curves.png
"""

import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

import timm
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import matplotlib
matplotlib.use("Agg")            # headless: save figures, never try to display
import matplotlib.pyplot as plt


# ============================== CONFIG =======================================
# Every knob can be overridden with an env var so you don't edit the file on
# Kaggle. Defaults are tuned for a ~2.4k-image field fine-tune.
DATA_DIR        = os.getenv("CROPGUARD_DATA_DIR", "/kaggle/input/corn-field/dataset")  # has train/ val/ test/
EXTRA_TRAIN_DIR = os.getenv("CROPGUARD_EXTRA_TRAIN_DIR", "")   # OPTIONAL lab data, TRAIN ONLY. "" = off.
OUT_DIR         = os.getenv("CROPGUARD_OUT_DIR", ".")
MODEL_NAME      = os.getenv("CROPGUARD_MODEL", "levit_256")
NUM_CLASSES     = int(os.getenv("CROPGUARD_NUM_CLASSES", "4"))

EPOCHS          = int(os.getenv("CROPGUARD_EPOCHS", "100"))    # early stopping (PATIENCE) usually halts before this
BATCH_SIZE      = int(os.getenv("CROPGUARD_BATCH", "32"))
WARMUP_EPOCHS   = int(os.getenv("CROPGUARD_WARMUP", "3"))
HEAD_LR         = float(os.getenv("CROPGUARD_HEAD_LR", "3e-4"))
BACKBONE_LR     = float(os.getenv("CROPGUARD_BACKBONE_LR", "3e-5"))   # 10x lower: protect pretrained features
WEIGHT_DECAY    = float(os.getenv("CROPGUARD_WD", "0.05"))
LABEL_SMOOTH    = float(os.getenv("CROPGUARD_LABEL_SMOOTH", "0.1"))
PATIENCE        = int(os.getenv("CROPGUARD_PATIENCE", "12"))   # early stop on val macro-F1
NUM_WORKERS     = int(os.getenv("CROPGUARD_WORKERS", "2"))
SEED            = int(os.getenv("CROPGUARD_SEED", "42"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def forward_logits(model, x):
    """timm's LeViT can be the *distilled* variant, whose forward returns a
    (cls, dist) tuple in training mode. Unwrap to the classification logits so
    the loss and metrics work on any timm version."""
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def get_targets(ds):
    """Flat list of integer labels for an ImageFolder or a ConcatDataset of them."""
    if isinstance(ds, ConcatDataset):
        t = []
        for d in ds.datasets:
            t.extend(get_targets(d))
        return t
    return list(ds.targets)


# ============================== DATA =========================================
def build_transforms(model):
    """Read the pretrained model's own normalization so inputs match how the
    backbone was trained. The eval transform mirrors serving preprocessing.py
    exactly (square resize, no center-crop, so the whole leaf is kept)."""
    try:
        cfg = timm.data.resolve_data_config({}, model=model)
        mean, std = tuple(cfg["mean"]), tuple(cfg["std"])
        img = cfg["input_size"][-1]
    except Exception:
        mean, std, img = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224

    fill = tuple(int(255 * m) for m in mean)   # rotate-fill with mean color, not black
    train_tf = transforms.Compose([
        transforms.RandomRotation(20, fill=fill),
        transforms.RandomResizedCrop(img, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),   # helps span the lab -> field colour gap
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf, mean, std, img


def build_loaders(train_tf, eval_tf):
    train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_set   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)
    test_set  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=eval_tf)

    classes = train_set.classes
    # Guard the recurring gotcha: every split must share the same class->index map.
    for name, s in (("val", val_set), ("test", test_set)):
        if s.classes != classes:
            raise RuntimeError(
                f"{name} split classes {s.classes} != train {classes}. "
                "All splits need identical class subfolders in the same (alphabetical) order."
            )

    # OPTIONAL: add laboratory images to TRAIN ONLY (the ablation). Never val/test.
    if EXTRA_TRAIN_DIR:
        extra = datasets.ImageFolder(EXTRA_TRAIN_DIR, transform=train_tf)
        if extra.classes != classes:
            raise RuntimeError(f"EXTRA_TRAIN_DIR classes {extra.classes} != train {classes}.")
        train_set = ConcatDataset([train_set, extra])
        print(f"[data] added {len(extra)} extra TRAIN images from {EXTRA_TRAIN_DIR}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    return train_loader, val_loader, test_loader, train_set, classes


# ============================== TRAIN / EVAL =================================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = forward_logits(model, x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = forward_logits(model, x)
        loss_sum += criterion(out, y).item() * x.size(0)
        n += x.size(0)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    acc = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro")
    return loss_sum / n, acc, macro_f1, labels, preds


# ============================== PLOTS / REPORTS =============================
def save_confusion_matrix(cm, classes, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("LeViT — field test confusion matrix")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_curves(hist, path):
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["train_loss"], label="train")
    a1.plot(ep, hist["val_loss"], label="val")
    a1.set_title("Loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(ep, hist["train_acc"], label="train acc")
    a2.plot(ep, hist["val_acc"], label="val acc")
    a2.plot(ep, hist["val_f1"], label="val macro-F1")
    a2.set_title("Accuracy / macro-F1")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ============================== MAIN =========================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[env] device={device}  model={MODEL_NAME}  epochs={EPOCHS}")

    # --- model: pretrained backbone, fresh 4-class head ---
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(device)

    train_tf, eval_tf, mean, std, img = build_transforms(model)
    train_loader, val_loader, test_loader, train_set, classes = build_loaders(train_tf, eval_tf)
    if len(classes) != NUM_CLASSES:
        raise RuntimeError(f"Found {len(classes)} class folders but NUM_CLASSES={NUM_CLASSES}: {classes}")
    print(f"[data] classes (index order): {list(enumerate(classes))}")

    # --- class-weighted loss for the imbalance (weights from TRAIN only) ---
    targets = np.asarray(get_targets(train_set))
    counts = np.bincount(targets, minlength=NUM_CLASSES)
    print(f"[data] train per-class counts: {dict(zip(classes, counts.tolist()))}")
    class_weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))   # 'balanced' weights
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"[data] class weights: {dict(zip(classes, np.round(class_weights, 3).tolist()))}")
    train_criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=LABEL_SMOOTH)
    eval_criterion  = nn.CrossEntropyLoss()   # honest: unweighted, no smoothing

    # --- discriminative LR: head learns fast, backbone gently ---
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        (head_params if "head" in name else backbone_params).append(p)
    optimizer = optim.AdamW(
        [{"params": backbone_params, "lr": BACKBONE_LR},
         {"params": head_params,     "lr": HEAD_LR}],
        weight_decay=WEIGHT_DECAY,
    )

    # --- warmup (from 1/warmup) then cosine to ~0; one factor scales both groups ---
    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- training loop: best-val-macro-F1 checkpointing + early stopping ---
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "val_f1")}
    best_f1, best_state, bad = -1.0, None, 0
    ckpt_path = os.path.join(OUT_DIR, "levit_cropguard.pth")
    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, train_criterion)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, eval_criterion)
        scheduler.step()

        for k, v in zip(("train_loss", "val_loss", "train_acc", "val_acc", "val_f1"),
                        (tr_loss, val_loss, tr_acc, val_acc, val_f1)):
            hist[k].append(v)
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} macroF1 {val_f1:.4f}  "
              f"({time.time()-t0:.1f}s)")

        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"state_dict": best_state, "classes": classes, "model_name": MODEL_NAME,
                 "num_classes": NUM_CLASSES, "mean": mean, "std": std, "img_size": img,
                 "val_macro_f1": best_f1, "epoch": epoch + 1},
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stop: no val macro-F1 improvement in {PATIENCE} epochs.")
                break

    # --- final: score the BEST checkpoint on the field-only TEST split ---
    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, eval_criterion)
    print(f"\n=== FIELD TEST ===  acc {test_acc:.4f}  macro-F1 {test_f1:.4f}  (best val F1 {best_f1:.4f})")

    report_txt = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    print("\n" + report_txt)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # --- save artifacts ---
    with open(os.path.join(OUT_DIR, "levit_class_names.json"), "w") as f:
        json.dump(classes, f, indent=2)
    with open(os.path.join(OUT_DIR, "levit_test_report.txt"), "w") as f:
        f.write(f"Field test accuracy: {test_acc:.4f}\nField test macro-F1: {test_f1:.4f}\n\n")
        f.write(report_txt + "\n\nConfusion matrix (rows=true, cols=pred):\n" + str(cm) + "\n")
    with open(os.path.join(OUT_DIR, "levit_test_report.json"), "w") as f:
        json.dump(
            {"accuracy": test_acc, "macro_f1": test_f1, "best_val_macro_f1": best_f1,
             "classes": classes,
             "per_class": classification_report(y_true, y_pred, target_names=classes,
                                                 output_dict=True, zero_division=0),
             "confusion_matrix": cm.tolist()},
            f, indent=2,
        )
    save_confusion_matrix(cm, classes, os.path.join(OUT_DIR, "levit_confusion_matrix.png"))
    save_curves(hist, os.path.join(OUT_DIR, "levit_training_curves.png"))
    print(f"\nSaved to {OUT_DIR}: levit_cropguard.pth, levit_class_names.json, "
          f"levit_test_report.(txt|json), levit_confusion_matrix.png, levit_training_curves.png")


if __name__ == "__main__":
    main()
