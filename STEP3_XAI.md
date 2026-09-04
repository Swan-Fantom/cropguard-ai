# CropGuard AI — Step 3: The explainability heatmap (Grad-CAM)

Your API can now say *why*. Alongside the diagnosis, `POST /explain` returns a **heatmap** painted over the leaf, showing which regions pushed the model toward its answer. A trustworthy result looks at the actual lesions; a suspicious one lights up the background or a shadow. This is the feature that turns CropGuard from "a classifier" into "a tool you can interrogate" — and it's a strong thing to demo in an interview.

## What got added

| File | Role |
|------|------|
| `explain.py` | **New.** From-scratch Grad-CAM — hooks a token-grid layer, builds the class-activation map, colorizes it, and overlays it on your image. |
| `app.py` | **New endpoint** `POST /explain` (plus a small refactor: upload validation is now shared with `/predict`). |
| `explain_test.html` | **New.** A throwaway browser tool to *see* the heatmap. Not the real app — that's Step 4. |

No new dependencies — it's built on the torch, numpy, and Pillow you already have.

---

## How to run it

Same as before — just restart the server so it picks up the new endpoint:

```bash
uvicorn app:app --reload
```

## See the heatmap (easiest)

Open **`explain_test.html`** in your browser (double-click it). Pick a leaf image, choose a **Detail** level (see below), hit **Diagnose & explain**, and you'll see the original and the heatmap side by side, with the confidence bars underneath.

> If the page shows a "Request failed" error, the API isn't running — start `uvicorn app:app --reload` first. (The server already allows browser requests via CORS.)

## Tuning heatmap resolution (the "Detail" dropdown / `stage`)

A vision transformer builds token grids that get smaller and more abstract with depth. Grad-CAM can be read off different grids, trading detail for semantics. The layers are auto-discovered per model, so the grid sizes below are the ones found for the served LeViT-256 @ 224 (14×14 → 7×7 → 4×4):

| Setting | Grid | Character |
|---------|------|-----------|
| **Coarse** (`last`) | 4×4 | Most semantic, but a big soft blob. |
| **Finer** (`penultimate`) | 7×7 | Localizes tighter on lesions; slightly less abstract. |
| **Combined** (default) | 7×7 + 4×4 averaged | Balance of the two — usually the most readable. |

Different photos localize best at different depths, so the tester has a **Detail** dropdown to compare them on your own images. The API default is `combined` (set `CROPGUARD_CAM_STAGE` to change it, or pass a `stage` form field per request).

## Or test it from code

```python
import requests, base64

with open("Sample_1.jpg", "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/explain",
        files={"file": f},
        data={"stage": "combined"},   # "last" | "penultimate" | "combined"
    ).json()

print(r["label"], f'{r["confidence"]*100:.1f}%')

# The heatmap comes back as a data URL; strip the prefix and save the PNG:
png = r["heatmap"].split(",", 1)[1]
with open("heatmap.png", "wb") as out:
    out.write(base64.b64decode(png))
print("saved heatmap.png")
```

## Or from `/docs`

`POST /explain` also appears in the Swagger page at `http://127.0.0.1:8000/docs`. It works, but the response contains the heatmap as a very long base64 string — fine for confirming it returns, awkward for actually viewing. Use the HTML tester for that.

---

## What the response looks like

```json
{
  "filename": "Sample_1.jpg",
  "prediction": "Common_Rust",
  "label": "Common Rust",
  "confidence": 0.838,
  "is_confident": true,
  "threshold": 0.7,
  "stage": "combined",
  "heatmap": "data:image/png;base64,iVBORw0KGgo...",
  "top_k": [ ... ]
}
```

It's the same shape as `/predict`, with two extra fields: **`heatmap`** (a ready-to-display PNG `data:` URL) and **`stage`** (which resolution was used). In the web app later, the heatmap drops straight into an `<img src="...">`.

---

## How it actually works (so you can explain it)

Grad-CAM answers: *"Which parts of the last feature map, weighted by how much they influence the chosen class, support that prediction?"*

1. **Hook** a token-grid layer to grab its output activations on the forward pass. The layer is *auto-discovered* from a dummy forward pass (the deepest module whose output is a square token grid), so there's no hardcoded module path — it adapts to the served LeViT and to other timm versions.
2. **Forward + backward:** run the image, pick the top class, and backprop that class score to get gradients at that same layer.
3. **Reshape:** the tokens are a flattened square grid (e.g. LeViT's 7×7 = 49 tokens), so reshape `(1, L, C)` back to `(side, side, C)`.
4. **Weight & combine:** average the gradients over the grid to get one weight per channel, take the weighted sum of the activation channels, then **ReLU** (keep only positive evidence).
5. **Normalize → upsample → colorize** the small map up to the image, apply a jet colormap, and alpha-blend it over the photo.

A few honest caveats worth knowing:

- The map is **coarse** — it shows *regions*, not pixel-perfect outlines. That's normal for Grad-CAM on a transformer. The **Detail** setting above (Coarse → Finer, i.e. 4×4 → 7×7) tightens it up, but it will never be a crisp per-lesion mask.
- **Black/segmented backgrounds cause an edge artifact.** On lab-style images with a solid black background (the kind you get from segmentation), the heatmap tends to light up the *border* rather than the leaf. Reason: black pixels become the most extreme values after ImageNet normalization, which spikes the tokens over the background. On natural field photos — the kind this model was trained on — this largely goes away and the map moves onto the leaf, a useful reminder that the heatmap partly reflects the data, not just the model.
- **Why Grad-CAM and not attention rollout?** Stitching per-head, per-layer attention into one clean whole-leaf map is fiddly and architecture-specific. Grad-CAM sidesteps that — it works off the feature map and its gradients — and is the reliable choice here. (Attention rollout could be a later experiment.)
- The heatmap explains **where**, not **whether the model is right**. A confident, wrong prediction can still produce a tidy-looking heatmap — so treat it as a sanity check, not proof. Genuinely fixing wrong-but-confident calls on field photos is the model-robustness track, not this feature.

---

## What comes next

4. **Web app (Step 4)** — the real React + Tailwind interface: upload a photo, see the diagnosis and this heatmap, with a Node/Express layer for login and saved history. The `explain_test.html` you just used is a tiny preview of what that page does.
5. **Containerize + deploy** — Docker, then Azure.

When the heatmap looks sensible in the tester, tell me and we'll start **Step 4 — the web app**.
