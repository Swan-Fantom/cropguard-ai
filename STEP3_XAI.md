# CropGuard AI — Step 3: The explainability heatmap (Grad-CAM)

Your API can now say *why*. Alongside the diagnosis, `POST /explain` returns a **heatmap** painted over the leaf, showing which regions pushed the model toward its answer. A trustworthy result looks at the actual lesions; a suspicious one lights up the background or a shadow. This is the feature that turns CropGuard from "a classifier" into "a tool you can interrogate" — and it's a strong thing to demo in an interview.

## What got added

| File | Role |
|------|------|
| `explain.py` | **New.** From-scratch Grad-CAM for Swin — hooks the last stage, builds the class-activation map, colorizes it, and overlays it on your image. |
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

Swin builds features in stages that get smaller and more abstract with depth. Grad-CAM can be read off different stages, trading detail for semantics:

| Setting | Grid | Character |
|---------|------|-----------|
| **Coarse** | 7×7 | Most semantic, but a big soft blob. The original Step-3 default. |
| **Finer** | 14×14 | Localizes tighter on lesions; slightly less abstract. |
| **Combined** (default) | 14×14 + 7×7 averaged | Balance of the two — usually the most readable. |

Different photos localize best at different depths, so the tester has a **Detail** dropdown to compare them on your own images. The API default is `combined` (set `CROPGUARD_CAM_STAGE` to change it, or pass a `stage` form field per request).

## Or test it from code

```python
import requests, base64

with open("RS_Rust_1918.JPG", "rb") as f:
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
  "filename": "RS_Rust_1918.JPG",
  "prediction": "Corn_(maize)___Common_rust_",
  "label": "Common rust",
  "confidence": 1.0,
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

1. **Hook** the last Swin stage's final block (`model.layers[-1].blocks[-1].norm1`) to grab its output on the forward pass.
2. **Forward + backward:** run the image, pick the top class, and backprop that class score to get gradients at that same layer.
3. **Reshape:** Swin's last stage is a 7×7 grid flattened to 49 tokens, so reshape `(1, 49, C)` back to `(7, 7, C)`.
4. **Weight & combine:** average the gradients over the grid to get one weight per channel, take the weighted sum of the activation channels, then **ReLU** (keep only positive evidence).
5. **Normalize → upsample → colorize** the 7×7 map up to the image, apply a jet colormap, and alpha-blend it over the photo.

A few honest caveats worth knowing:

- The map is **coarse** — it shows *regions*, not pixel-perfect outlines. That's normal for Grad-CAM on a transformer. The **Detail** setting above (7×7 → 14×14) tightens it up, but it will never be a crisp per-lesion mask.
- **Black/segmented backgrounds cause an edge artifact.** On lab-style images with a solid black background (like the PlantVillage training data), the heatmap tends to light up the *border* rather than the leaf. Reason: black pixels become the most extreme values after ImageNet normalization, which spikes the tokens over the background. On natural field photos this largely goes away and the map moves onto the leaf — a useful reminder that the heatmap partly reflects the data, not just the model.
- **Why Grad-CAM and not attention rollout?** Swin's attention is computed inside shifted local windows, so raw attention doesn't stitch into a clean whole-leaf map. Grad-CAM avoids that and is the reliable choice here. (Attention rollout could be a later experiment.)
- The heatmap explains **where**, not **whether the model is right**. A confident, wrong prediction can still produce a tidy-looking heatmap — so treat it as a sanity check, not proof. Genuinely fixing wrong-but-confident calls on field photos is the model-robustness track, not this feature.

---

## What comes next

4. **Web app (Step 4)** — the real React + Tailwind interface: upload a photo, see the diagnosis and this heatmap, with a Node/Express layer for login and saved history. The `explain_test.html` you just used is a tiny preview of what that page does.
5. **Containerize + deploy** — Docker, then Azure.

When the heatmap looks sensible in the tester, tell me and we'll start **Step 4 — the web app**.
