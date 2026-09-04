"""
CropGuard AI — FastAPI ML microservice (Step 2).
================================================================================
Wraps the Step 1 model in a small web API so the React front-end (and anything
else) can get a prediction over HTTP.

    POST /predict   (multipart image upload) -> {prediction, confidence, top_k, ...}
    GET  /classes   -> the disease classes the model knows
    GET  /health    -> liveness + model status
    GET  /          -> service info

The model is loaded ONCE at startup (FastAPI lifespan) and reused for every
request — loading Swin per-request would be far too slow. Preprocessing and the
architecture come from the shared model.py / preprocessing.py, so the API can
never drift from your validated Kaggle results.

Run it:
    pip install -r requirements.txt
    uvicorn app:app --reload
Then open the interactive docs at  http://127.0.0.1:8000/docs

Config (optional env vars):
    CROPGUARD_BACKEND           default "levit"   ("levit" | "swin")
    CROPGUARD_CHECKPOINT        default "levit_cropguard.pth" (levit) / "swin_cropguard.pth" (swin)
    CROPGUARD_CLASSES           default "class_names.json"  (swin only; levit reads classes from its checkpoint)
    CROPGUARD_SWIN_REPO         default "Swin-Transformer"   (swin only)
    CROPGUARD_CONF_THRESHOLD    default "0.70"   (below this -> is_confident=false)
    CROPGUARD_MAX_UPLOAD_MB     default "10"
    CROPGUARD_TOPK              default "3"
    CROPGUARD_CAM_ALPHA         default "0.45"  (heatmap blend strength; swin /explain only)
    CROPGUARD_CAM_STAGE         default "combined"  ("last" | "penultimate" | "combined"; swin only)
"""

import io
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from model import CropGuardModel, pretty_label
from explain import explain_image, VALID_STAGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("cropguard")

# --- Configuration (env vars override) ----------------------------------------
# BACKEND picks which trained model to serve:
#   "levit" (default) -> the field-trained timm LeViT from train_levit.py
#   "swin"            -> the original from-scratch Microsoft Swin-Tiny
BACKEND = os.getenv("CROPGUARD_BACKEND", "levit").strip().lower()
_default_ckpt = "levit_cropguard.pth" if BACKEND == "levit" else "swin_cropguard.pth"
CHECKPOINT = os.getenv("CROPGUARD_CHECKPOINT", _default_ckpt)
CLASSES = os.getenv("CROPGUARD_CLASSES", "class_names.json")      # swin only; levit reads classes from its checkpoint
SWIN_REPO = os.getenv("CROPGUARD_SWIN_REPO", "Swin-Transformer")  # swin only
CONF_THRESHOLD = float(os.getenv("CROPGUARD_CONF_THRESHOLD", "0.70"))
MAX_UPLOAD_MB = float(os.getenv("CROPGUARD_MAX_UPLOAD_MB", "10"))
TOPK = int(os.getenv("CROPGUARD_TOPK", "3"))
CAM_ALPHA = float(os.getenv("CROPGUARD_CAM_ALPHA", "0.45"))
# Grad-CAM resolution: "last" (7x7, coarse/semantic), "penultimate" (14x14, finer),
# or "combined" (both averaged). Per-request override allowed on /explain.
CAM_STAGE = os.getenv("CROPGUARD_CAM_STAGE", "combined")

# Loaded model lives here after startup.
STATE: dict = {"model": None}

# Grad-CAM runs a backward pass and briefly mutates shared model state (hooks,
# gradients), so serialize /explain calls to keep concurrent requests correct.
_explain_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server boots; release it on shutdown."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading CropGuard model (backend=%s) on %s ...", BACKEND, device)
    if BACKEND == "levit":
        STATE["model"] = CropGuardModel.load_levit(CHECKPOINT, device)
    elif BACKEND == "swin":
        STATE["model"] = CropGuardModel.load(CHECKPOINT, CLASSES, SWIN_REPO, device)
    else:
        raise RuntimeError(f"Unknown CROPGUARD_BACKEND '{BACKEND}' — use 'levit' or 'swin'.")
    logger.info("Model ready: backend=%s, %d classes, threshold=%.2f",
                BACKEND, STATE["model"].num_classes, CONF_THRESHOLD)
    yield
    STATE["model"] = None
    logger.info("Model unloaded.")


app = FastAPI(
    title="CropGuard AI",
    description="Corn-leaf disease classifier — field-trained LeViT (Swin selectable via CROPGUARD_BACKEND).",
    version="0.3.0",
    lifespan=lifespan,
)

# Allow the future React dev server to call this API from the browser.
# Tighten allow_origins to your real front-end origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response schemas ---------------------------------------------------------
class ClassPrediction(BaseModel):
    disease: str          # raw class name (matches training labels)
    label: str            # human-friendly version for display
    confidence: float     # 0..1


class PredictResponse(BaseModel):
    filename: Optional[str]
    prediction: str                 # raw top-1 class name
    label: str                      # human-friendly top-1
    confidence: float               # top-1 probability, 0..1
    is_confident: bool              # confidence >= threshold
    threshold: float
    message: Optional[str] = None   # guidance shown when not confident
    top_k: List[ClassPrediction]


class ExplainResponse(BaseModel):
    filename: Optional[str]
    prediction: str
    label: str
    confidence: float
    is_confident: bool
    threshold: float
    stage: str                      # which Grad-CAM resolution stage was used
    heatmap: str                    # 'data:image/png;base64,...' overlay for an <img> tag
    top_k: List[ClassPrediction]


def _require_model() -> CropGuardModel:
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still loading; try again shortly.")
    return model


async def _load_upload(file: UploadFile) -> Image.Image:
    """Read + validate an uploaded image; raise HTTPException on bad input."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image too large (> {MAX_UPLOAD_MB:.0f} MB).")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()                      # force decode so truncated files fail here
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400,
                            detail="Could not read that file as an image. Upload a JPG or PNG.")


# --- Endpoints ----------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "CropGuard AI",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["GET /health", "GET /classes", "POST /predict", "POST /explain"],
    }


@app.get("/health")
def health():
    model = STATE["model"]
    return {
        "status": "ok" if model is not None else "loading",
        "backend": BACKEND,
        "num_classes": model.num_classes if model else 0,
        "device": str(model.device) if model else None,
        "confidence_threshold": CONF_THRESHOLD,
    }


@app.get("/classes")
def classes():
    model = _require_model()
    return {
        "classes": model.class_names,
        "labels": [pretty_label(c) for c in model.class_names],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    model = _require_model()

    # 1) Read + validate the upload.
    image = await _load_upload(file)

    # 2) Predict.
    results = model.predict(image, topk=TOPK)   # [(disease, prob), ...]
    top_disease, top_conf = results[0]
    is_confident = top_conf >= CONF_THRESHOLD

    # 3) Confidence gate. NOTE: this is a scaffold — a from-scratch model trained
    # on lab-conditioned PlantVillage tends to be over-confident, so raw softmax
    # is a weak uncertainty signal. It gets meaningful once we calibrate the model
    # (temperature scaling) and add real field data in the robustness track.
    message = None
    if not is_confident:
        message = (
            f"Low confidence ({top_conf * 100:.1f}%). This might be a leaf/condition "
            "the model wasn't trained on, or the photo is unclear — try a sharp, "
            "well-lit shot of a single leaf filling the frame."
        )

    return PredictResponse(
        filename=file.filename,
        prediction=top_disease,
        label=pretty_label(top_disease),
        confidence=round(top_conf, 4),
        is_confident=is_confident,
        threshold=CONF_THRESHOLD,
        message=message,
        top_k=[
            ClassPrediction(disease=d, label=pretty_label(d), confidence=round(p, 4))
            for d, p in results
        ],
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(file: UploadFile = File(...), stage: Optional[str] = Form(None)):
    """Like /predict, but also returns a Grad-CAM heatmap overlay showing which
    leaf regions drove the diagnosis. Heavier than /predict (adds a backward pass),
    so it's a separate endpoint the UI calls only when it wants the visual.

    Optional `stage` form field tunes heatmap resolution: "last" (7x7, coarse),
    "penultimate" (14x14, finer), or "combined" (both). Defaults to CROPGUARD_CAM_STAGE."""
    model = _require_model()

    # Grad-CAM works for both backends: the Swin target layers are known, and any
    # other backend (LeViT) has its token-grid layers auto-discovered in explain.py.
    image = await _load_upload(file)

    stage_used = (stage or CAM_STAGE).strip().lower()
    if stage_used not in VALID_STAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid stage '{stage_used}'. Choose one of: {', '.join(VALID_STAGES)}.",
        )

    # Grad-CAM mutates shared model state, so serialize concurrent calls.
    with _explain_lock:
        result = explain_image(model, image, topk=TOPK, alpha=CAM_ALPHA, stage=stage_used)

    top_disease, top_conf = result["top"][0]
    is_confident = top_conf >= CONF_THRESHOLD

    return ExplainResponse(
        filename=file.filename,
        prediction=top_disease,
        label=pretty_label(top_disease),
        confidence=round(top_conf, 4),
        is_confident=is_confident,
        threshold=CONF_THRESHOLD,
        stage=result["stage"],
        heatmap=result["heatmap"],
        top_k=[
            ClassPrediction(disease=d, label=pretty_label(d), confidence=round(p, 4))
            for d, p in result["top"]
        ],
    )
