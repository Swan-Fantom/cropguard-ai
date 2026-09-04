# CropGuard AI 🌽

**Explainable corn-leaf disease diagnosis — a full-stack ML web application.**

Upload a photo of a corn (maize) leaf and CropGuard classifies the disease, tells
you how confident it is, and shows a **Grad-CAM heatmap** of the exact regions
that drove its decision. Every diagnosis is saved to a per-user history.

CropGuard is built as a **three-tier system**: a React single-page app talks to a
Node/Express API (which owns authentication and history), which in turn calls a
Python FastAPI microservice that runs the deep-learning model.

> **Status:** Runs locally end-to-end (all tiers verified). Containerization and
> cloud deployment are the next planned step — see [Roadmap](#roadmap).

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [The model](#the-model)
- [Explainability (Grad-CAM)](#explainability-grad-cam)
- [Running it locally](#running-it-locally)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Screenshots](#screenshots)

---

## Features

- **Disease classification** across 4 classes: Common Rust, Gray Leaf Spot,
  Northern Leaf Blight, and Healthy.
- **Confidence gating** — predictions below a threshold are flagged as
  "low confidence / treat as a suggestion" rather than shown as a certain answer.
- **Visual explanations** — a Grad-CAM heatmap overlays the leaf, with a selectable
  detail level, so a user can see *why* the model decided what it did.
- **User accounts** — register / log in with JWT-based auth; passwords are
  bcrypt-hashed.
- **Scan history** — every diagnosis (including its heatmap) is stored per user in
  MongoDB and can be revisited or deleted.

## Architecture

```
  Browser — React + Tailwind SPA (web/)
      │  fetch /api/...            (Vite dev-proxies /api → Node)
      ▼
  Node / Express API (server/)     auth (JWT) · scan history
      │            │
      │            └── MongoDB      users + saved scans
      ▼
  FastAPI ML service (app.py)      LeViT classify + Grad-CAM
      ▼
  PyTorch model (levit_cropguard.pth)
```

**Why the extra Node tier instead of the browser calling FastAPI directly?**
It owns the things a stateless ML service shouldn't: user accounts, auth tokens,
and per-user history in a database. It also gives the browser a single origin to
talk to. This SPA → app-server + DB → ML-microservice split is a deliberate,
scalable design choice.

## Tech stack

| Tier | Technology |
|------|-----------|
| Frontend | React 18, React Router, Tailwind CSS, Vite |
| API server | Node.js, Express, JWT, bcrypt, Mongoose |
| Database | MongoDB |
| ML service | Python, FastAPI, Uvicorn |
| Model / CV | PyTorch, timm (LeViT), Grad-CAM (implemented from scratch) |

## The model

The served model is a **LeViT-256 vision transformer**, pretrained on ImageNet and
**fine-tuned on real field photographs** of corn leaves (the "Natural Environment"
dataset, 4 classes with train/val/test splits).

A key design decision: an earlier version used a Swin Transformer trained
*from scratch* on clean lab-style images, which was **brittle on real-world photos**
(it would confidently mislabel field images). Switching to a *pretrained* backbone
fine-tuned *field-first* fixed this — the lab→field domain gap was the root cause,
not the architecture.

**Held-out field test results** (338 images):

| Metric | Score |
|--------|-------|
| Accuracy | **98.8%** |
| Macro-F1 | **0.981** |

Per-class (precision / recall): Common Rust 1.00 / 0.97, Gray Leaf Spot 0.89 / 1.00,
Healthy 0.99 / 1.00, Northern Leaf Blight 1.00 / 1.00.

> **Honest caveats:** the Gray Leaf Spot test set is small (n=25), so its recall is
> noisy — precision (0.89) is the more informative number there. The remaining
> errors are all on the Common-Rust↔Gray-Leaf-Spot boundary. Confidence
> calibration (temperature scaling) is a planned improvement.

The training script (`train_levit.py`) handles class imbalance, uses discriminative
learning rates, early-stops on validation macro-F1, and saves a confusion matrix +
per-class report. The trained weights (`levit_cropguard.pth`, ~72 MB) are **not**
committed to the repo — see [Running it locally](#running-it-locally).

## Explainability (Grad-CAM)

CropGuard implements **Grad-CAM from scratch** (forward/backward hooks + a NumPy/PIL
overlay — no `pytorch-grad-cam` dependency). It reshapes the transformer's token
grid back into a spatial map, weights it by the gradients of the predicted class,
and blends a jet-colormap heatmap onto the original image.

The implementation **auto-discovers** the model's token-grid layers at runtime, so
it adapts to different transformer backbones and timm versions instead of
hardcoding module paths. Three detail levels are exposed (Coarse / Finer /
Combined) mapping to different depths of the network.

## Running it locally

You need **Python 3.10+**, **Node 18+**, and **MongoDB** running locally.

**0. Get the data and model weights** (not in the repo):
- Dataset: the corn "Natural Environment" leaf dataset (4 classes, train/val/test).
- Model weights: `levit_cropguard.pth` — produced by running `train_levit.py`, or
  place your own trained checkpoint at the repo root.

**1. ML service** (repo root):
```bash
pip install -r requirements.txt
uvicorn app:app --reload            # → http://127.0.0.1:8000
```

**2. Node API** (`server/`):
```bash
cd server
cp .env.example .env                # then set JWT_SECRET and MONGO_URI
npm install
npm run dev                         # → http://127.0.0.1:4000
```

**3. React app** (`web/`):
```bash
cd web
npm install
npm run dev                         # → http://localhost:5173
```

Open **http://localhost:5173**, sign up, and diagnose a leaf.

More detail per step lives in `STEP2_FASTAPI.md`, `STEP3_XAI.md`, and
`STEP4_WEBAPP.md`.

## Project layout

```
CropGuard/
├─ app.py                 FastAPI ML service (/predict, /explain)
├─ model.py               model build/load + predict core
├─ explain.py             from-scratch Grad-CAM (auto-discovers token layers)
├─ preprocessing.py       inference transforms (must match training)
├─ infer.py               CLI inference
├─ train_levit.py         LeViT fine-tuning script
├─ requirements.txt
├─ server/                Node/Express API (auth, history, ML proxy)
│  └─ src/
│     ├─ routes/          auth.js, scans.js
│     ├─ models/          User.js, Scan.js
│     └─ middleware/      auth.js, errors.js
└─ web/                   React + Vite + Tailwind SPA
   └─ src/
      ├─ pages/           Login, Register, Diagnose, History
      └─ components/
```

## Roadmap

- [x] **Step 1** — Local model inference
- [x] **Step 2** — FastAPI ML microservice
- [x] **Step 3** — Grad-CAM explainability
- [x] **Step 4** — Full-stack web app (React + Node/Express + MongoDB)
- [ ] **Step 5** — Containerize each tier (Docker + docker-compose) and deploy to
  the cloud (Azure). This step also adds production security hardening:
  httpOnly-cookie auth + CSRF protection, rate limiting, and CORS lockdown.
- [ ] Confidence calibration (temperature scaling) + a real "unsure" gate.

## Screenshots

> _Add screenshots of the Diagnose page (with heatmap) and History here._

---

Built as a portfolio project exploring the full lifecycle of an ML product:
model training, explainability, API design, and full-stack integration.
