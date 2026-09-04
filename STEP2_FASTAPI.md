# CropGuard AI — Step 2: The FastAPI prediction service

Step 1 proved the model works from a script. Step 2 turns it into a **web service**: a small API that accepts an uploaded leaf photo over HTTP and returns a JSON prediction. This is the piece your React front-end will call, and later the thing you put in a Docker container and deploy.

Nothing about the model changed — `app.py` loads the exact same Swin-Tiny through the shared `model.py`, using the same `preprocessing.py`. When you later train a more robust model, you drop in the new `.pth` and the API is unchanged.

---

## What got added

| File | Role |
|------|------|
| `model.py` | **New.** The reusable core — builds Swin-Tiny, loads weights, runs a prediction. One source of truth. |
| `app.py` | **New.** The FastAPI service (`/predict`, `/classes`, `/health`). |
| `infer.py` | **Refactored** to sit on top of `model.py`. Same CLI, same output as before. |
| `requirements.txt` | Added `fastapi`, `uvicorn`, `python-multipart`. |

The model loads **once at startup** and stays in memory — a request just runs a forward pass, so responses are fast.

---

## 1. Install the new dependencies

From inside `C:\College Materials\CropGuard`:

```bash
pip install -r requirements.txt
```

(Only `fastapi`, `uvicorn`, and `python-multipart` are new since Step 1.)

## 2. Start the server

```bash
uvicorn app:app --reload
```

You'll see it load the model, then:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

`--reload` auto-restarts the server when you edit the code — handy while developing. Drop it in production.

## 3. Test it — the easy way (no extra code)

Open **http://127.0.0.1:8000/docs** in your browser.

This is FastAPI's built-in Swagger UI. Expand **POST /predict** → **Try it out** → choose an image file → **Execute**. You'll see the JSON response right there. This interactive doc page is free — it's generated from the type hints in `app.py`.

## 4. Test it — from the command line

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@RS_Rust_1918.JPG"
```

## 5. Test it — from Python

```python
import requests

with open("RS_Rust_1918.JPG", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict", files={"file": f})
print(r.json())
```

---

## What a response looks like

```json
{
  "filename": "RS_Rust_1918.JPG",
  "prediction": "Corn_(maize)___Common_rust_",
  "label": "Common rust",
  "confidence": 1.0,
  "is_confident": true,
  "threshold": 0.7,
  "message": null,
  "top_k": [
    { "disease": "Corn_(maize)___Common_rust_", "label": "Common rust", "confidence": 1.0 },
    { "disease": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "label": "Cercospora leaf spot Gray leaf spot", "confidence": 0.0 },
    { "disease": "Corn_(maize)___healthy", "label": "Healthy", "confidence": 0.0 }
  ]
}
```

- **`prediction`** is the raw class name (matches your training labels exactly).
- **`label`** is a cleaned-up version for showing in the UI.
- **`confidence`** is the top softmax probability (0–1).
- **`top_k`** lets the front-end show the runners-up (and the margin between them).

## The `is_confident` gate

If the top confidence is below the threshold (default **0.70**), the response sets `is_confident: false` and fills in a helpful `message` — the intended UX is "best guess, but unsure — retake the photo" instead of a confident wrong answer.

Be honest with yourself about this for now: a model trained from scratch on lab-clean PlantVillage tends to be **over-confident** (you saw 100% on the rust image), so raw softmax is a weak uncertainty signal. This gate is deliberately a **scaffold**. It becomes genuinely useful once we (a) calibrate the model with temperature scaling and (b) add real field data — the robustness track we discussed. The plumbing is here now so we don't have to touch the API later.

## Other endpoints

- **`GET /health`** — is the service up, how many classes, which device, current threshold. Useful for Docker/Azure health checks later.
- **`GET /classes`** — the list of diseases the model knows (raw + friendly labels), so the front-end can render them without hard-coding.

## Configuration (optional)

Everything has a sensible default matching your files. Override with environment variables if needed:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CROPGUARD_CHECKPOINT` | `swin_cropguard.pth` | weights file |
| `CROPGUARD_CLASSES` | `class_names.json` | class list |
| `CROPGUARD_SWIN_REPO` | `Swin-Transformer` | cloned repo path |
| `CROPGUARD_CONF_THRESHOLD` | `0.70` | below this → `is_confident: false` |
| `CROPGUARD_MAX_UPLOAD_MB` | `10` | reject bigger uploads |
| `CROPGUARD_TOPK` | `3` | how many predictions to return |

---

## If something looks off

- **`503 Model is still loading`** → the first request arrived before the model finished loading. Wait a second and retry (mostly relevant on cold start).
- **`400 Could not read that file as an image`** → the upload wasn't a valid JPG/PNG, or the field name wasn't `file`.
- **`FileNotFoundError` about `models/swin_transformer.py`** on startup → the `Swin-Transformer/` repo isn't cloned in this folder (see Step 1), or `CROPGUARD_SWIN_REPO` points elsewhere.
- **Import error on `timm.models.layers`** → same timm-version gotcha as Step 1 (`pip install "timm==0.9.16"`).

---

## What comes next

3. **XAI heatmap** — the standout feature. Add attention-based explainability for Swin (attention rollout / `pytorch-grad-cam` with a reshape transform) so `/predict` can also return a heatmap of *where* the model looked. This is what makes CropGuard more than "another classifier."
4. **Web app** — React + Tailwind upload UI calling this API; Node/Express for auth + prediction history in MongoDB.
5. **Containerize + deploy** — Docker, then Azure (ties to your Azure cert).

When the API returns a good prediction on `/docs`, tell me and we'll build **Step 3 (the explainability heatmap)**.
