# Step 4 — Full-stack web app (React + Node/Express + MongoDB)

This step turns the CropGuard ML service into a real product: a web app where a
user signs up, uploads a corn-leaf photo, gets a diagnosis with a Grad-CAM
heatmap, and keeps a history of past scans.

## Architecture

```
  Browser (React + Tailwind, web/)
      │  fetch /api/...           (Vite dev-proxies /api -> Node)
      ▼
  Node/Express API (server/)      auth (JWT), scan history
      │           │
      │           └── MongoDB      users + saved scans
      ▼
  FastAPI ML service (app.py)     LeViT classify + Grad-CAM  (Step 2/3)
      ▼
  PyTorch model (levit_cropguard.pth)
```

Why the extra Node tier instead of the browser calling FastAPI directly? It's the
layer that owns the things a stateless ML service shouldn't: user accounts, auth
tokens, and per-user history in a database. It also gives the browser a single
origin to talk to. This 3-tier split (SPA → app server + DB → ML microservice) is
the architecture story worth being able to whiteboard in an interview.

## What each tier does

- **`web/`** — React (Vite) + Tailwind SPA. Pages: Login, Register, Diagnose
  (upload + heatmap detail selector + results), History (list + detail with saved
  heatmap). JWT is stored client-side and sent as a `Bearer` token.
- **`server/`** — Express API.
  - `POST /api/auth/register`, `POST /api/auth/login` → `{ token, user }`
  - `GET  /api/auth/me`
  - `POST /api/scans` (multipart `file`, optional `explain=true` + `stage`) —
    forwards the image to the FastAPI service, saves the result, returns it.
  - `GET  /api/scans` (history), `GET /api/scans/:id` (with heatmap),
    `DELETE /api/scans/:id`
  - Passwords are bcrypt-hashed; routes are protected by JWT middleware.
- **ML service** — the existing `app.py` (`uvicorn app:app`), unchanged.

## Running it (three terminals)

You need **MongoDB** running locally (or a MongoDB Atlas URI). Install Community
Server, or run via Docker: `docker run -d -p 27017:27017 --name mongo mongo:7`.

**1) ML service (Python)** — from the repo root:

```
uvicorn app:app --reload            # http://127.0.0.1:8000
```

**2) Node API** — from `server/`:

```
cd server
cp .env.example .env                # then edit .env (set JWT_SECRET, MONGO_URI)
npm install
npm run dev                         # http://127.0.0.1:4000
```

**3) React app** — from `web/`:

```
cd web
npm install
npm run dev                         # http://localhost:5173
```

Open http://localhost:5173, sign up, and diagnose a leaf. The Vite dev server
proxies `/api/*` to the Node server, which proxies image uploads to FastAPI.

## Configuration

- `server/.env` (see `.env.example`): `PORT`, `MONGO_URI`, `JWT_SECRET`,
  `JWT_EXPIRES_IN`, `ML_SERVICE_URL`, `CORS_ORIGINS`, `MAX_UPLOAD_MB`.
- The ML service still honors its own env vars (`CROPGUARD_BACKEND` defaults to
  `levit`, `CROPGUARD_CONF_THRESHOLD`, etc. — see `app.py`).
- Heatmap detail (`stage`) options in the UI map to the backend-agnostic
  Grad-CAM stages: **Combined** (balanced), **Finer** (penultimate — the sharpest
  useful map for the LeViT model), **Coarse** (last — most semantic, blocky).

## Security notes (before any public deploy)

This is built with sane defaults, but a portfolio project that mentions security
scores points. Things to harden before exposing it publicly:

- **JWT_SECRET** — set a long random value in `.env`; the code warns if left at
  the dev default.
- **Token storage** — the JWT is in `localStorage` for simplicity. For production,
  move to an `httpOnly` cookie so page JavaScript can't read it (limits XSS token
  theft), and add CSRF protection.
- **CORS** — `CORS_ORIGINS` should list only your real frontend origin(s).
- **Rate limiting** — add a limiter on `/api/auth/*` and `/api/scans` (e.g.
  `express-rate-limit`) to blunt brute-force and abuse.
- **HTTPS** — terminate TLS at your host/reverse proxy.
- The ML service currently allows all CORS origins and has an open (unauthed)
  API; in this stack the browser never hits it directly (only the Node server
  does), so keep the ML service on a private network / not publicly exposed.

## Next (Step 5)

Containerize each tier (Dockerfiles + `docker-compose` for web + server + ML +
MongoDB) and deploy to Azure — which ties into the Azure Fundamentals cert.
