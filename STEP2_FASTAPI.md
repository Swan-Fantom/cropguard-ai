# CropGuard AI — Step 2: The FastAPI prediction service

Step 1 proved the model works from a script. Step 2 turns it into a **web service**: a small API that accepts an uploaded leaf photo over HTTP and returns a JSON prediction. This is the piece your React front-end will call, and later the thing you put in a Docker container and deploy.

Nothing about the model changed — `app.py` loads the trained LeViT through the shared `model.py`, using the same `preprocessing.py`. When you later train an even more robust model, you drop in the new `.pth` and the API is unchanged.

---

## What got added

| File | Role |
|------|------|
| `model.py` | **New.** The reusable core — builds the LeViT, loads weights, runs a prediction. One source of truth. |
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
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@Sample_1.jpg"
```

## 5. Test it — from Python

```python
import requests

with open("Sample_1.jpg", "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict", files={"file": f})
print(r.json())
```

---

## What a response looks like

```json
{
  "filename": "Sample_1.jpg",
  "prediction": "Common_Rust",
  "label": "Common Rust",
  "confidence": 0.838,
  "is_confident": true,
  "threshold": 0.7,
  "message": null,
  "top_k": [
    { "disease": "Common_Rust", "label": "Common Rust", "confidence": 0.838 },
    { "disease": "Gray_Leaf_Spot", "label": "Gray Leaf Spot", "confidence": 0.087 },
    { "disease": "Northern_Leaf_Blight", "label": "Northern Leaf Blight", "confidence": 0.061 }
  ]
}
```

- **`prediction`** is the raw class name (matches the training labels exactly).
- **`label`** is a cleaned-up version for showing in the UI.
- **`confidence`** is the top softmax probability (0–1).
- **`top_k`** lets the front-end show the runners-up (and the margin between them).

## The `is_confident` gate

If the top confidence is below the threshold (default **0.70**), the response sets `is_confident: false` and fills in a helpful `message` — the intended UX is "best guess, but unsure — retake the photo" instead of a confident wrong answer.

This gate is deliberately a **scaffold**. Raw softmax is only a rough uncertainty signal — the trained LeViT is well-behaved on field photos (e.g. it correctly drops a borderline rust image to ~70%), but the gate becomes genuinely reliable once we add **temperature-scaling calibration**. The plumbing is here now so we don't have to touch the API later.

## Other endpoints

- **`GET /health`** — is the service up, how many classes, which device, current threshold. Useful for Docker/Azure health checks later.
- **`GET /classes`** — the list of diseases the model knows (raw + friendly labels), so the front-end can render them without hard-coding.

## Configuration (optional)

Everything has a sensible default matching your files. Override with environment variables if needed:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CROPGUARD_CHECKPOINT` | `levit_cropguard.pth` | trained LeViT weights file |
| `CROPGUARD_CONF_THRESHOLD` | `0.70` | below this → `is_confident: false` |
| `CROPGUARD_MAX_UPLOAD_MB` | `10` | reject bigger uploads |
| `CROPGUARD_TOPK` | `3` | how many predictions to return |

---

## If something looks off

- **`503 Model is still loading`** → the first request arrived before the model finished loading. Wait a second and retry (mostly relevant on cold start).
- **`400 Could not read that file as an image`** → the upload wasn't a valid JPG/PNG, or the field name wasn't `file`.
- **`FileNotFoundError` about `levit_cropguard.pth`** on startup → the trained checkpoint isn't in this folder, or `CROPGUARD_CHECKPOINT` points elsewhere.

---

## What comes next

3. **XAI heatmap** — the standout feature. Grad-CAM explanations (implemented from scratch) so `/explain` returns a heatmap of *where* the model looked. This is what makes CropGuard more than "another classifier."
4. **Web app** — React + Tailwind upload UI calling this API; Node/Express for auth + prediction history in MongoDB.
5. **Containerize + deploy** — Docker, then Azure (ties to your Azure cert).

When the API returns a good prediction on `/docs`, tell me and we'll build **Step 3 (the explainability heatmap)**.
