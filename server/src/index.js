/**
 * CropGuard AI — Node/Express API entry point (Step 4).
 * ============================================================================
 * The middle tier of the stack:
 *
 *   React (web/)  ->  THIS Node/Express API  ->  FastAPI ML service (../app.py)
 *                          |
 *                       MongoDB  (users + scan history)
 *
 * Responsibilities the Python service deliberately doesn't have:
 *   - user accounts + JWT auth
 *   - persisting each diagnosis so a user has a history
 *   - being the single origin the browser talks to
 *
 * Actual image classification / Grad-CAM still happens in the FastAPI service;
 * this API forwards the uploaded image there and stores the result.
 */
import cors from "cors";
import express from "express";

import { config, isInsecureJwtSecret } from "./config.js";
import { connectDB } from "./db.js";
import authRoutes from "./routes/auth.js";
import scanRoutes from "./routes/scans.js";
import { errorHandler, notFound } from "./middleware/errors.js";

const app = express();

// The frontend runs on a different origin (Vite dev server), so allow it.
app.use(
  cors({
    origin: config.corsOrigins,
    credentials: true,
  })
);
// JSON body parsing for auth routes (image uploads use multipart, handled by multer).
app.use(express.json({ limit: "1mb" }));

// Lightweight request log.
app.use((req, _res, next) => {
  console.log(`${new Date().toISOString()}  ${req.method} ${req.url}`);
  next();
});

app.get("/", (_req, res) => {
  res.json({
    service: "CropGuard API",
    version: "0.1.0",
    endpoints: [
      "POST /api/auth/register",
      "POST /api/auth/login",
      "GET  /api/auth/me",
      "POST /api/scans",
      "GET  /api/scans",
      "GET  /api/scans/:id",
      "DELETE /api/scans/:id",
      "GET  /api/health",
    ],
  });
});

app.get("/api/health", (_req, res) => {
  res.json({ status: "ok", mlService: config.mlServiceUrl });
});

app.use("/api/auth", authRoutes);
app.use("/api/scans", scanRoutes);

// 404 + centralized error handling (must be last).
app.use(notFound);
app.use(errorHandler);

async function start() {
  if (isInsecureJwtSecret) {
    console.warn(
      "[security] JWT_SECRET is using the insecure dev default. Set a strong " +
        "JWT_SECRET in .env before deploying anywhere real."
    );
  }
  try {
    await connectDB();
  } catch (err) {
    console.error("[db] failed to connect to MongoDB:", err.message);
    console.error(
      "[db] Is MongoDB running? Set MONGO_URI in .env (local mongod or Atlas)."
    );
    process.exit(1);
  }
  app.listen(config.port, () => {
    console.log(`[api] CropGuard API listening on http://127.0.0.1:${config.port}`);
    console.log(`[api] proxying ML requests to ${config.mlServiceUrl}`);
  });
}

start();
