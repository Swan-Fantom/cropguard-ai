/**
 * Scan routes — run a diagnosis and manage a user's history. All require auth.
 *
 *   POST   /api/scans            (multipart: file, [explain=true], [stage])  -> saved scan
 *   GET    /api/scans            -> [scan, ...]  (newest first, no heatmap payload)
 *   GET    /api/scans/:id        -> scan (with heatmap if present)
 *   DELETE /api/scans/:id        -> { ok: true }
 *
 * POST forwards the image to the FastAPI ML service (/predict, or /explain when
 * a heatmap is requested), stores the result against the user, and returns it.
 */
import { Router } from "express";
import mongoose from "mongoose";
import multer from "multer";

import { config } from "../config.js";
import { asyncHandler, HttpError } from "../middleware/errors.js";
import { requireAuth } from "../middleware/auth.js";
import { Scan } from "../models/Scan.js";
import { explain as mlExplain, predict as mlPredict } from "../mlClient.js";

const router = Router();

// Keep the upload in memory — we immediately forward it to FastAPI, never to disk.
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: config.maxUploadMb * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype && file.mimetype.startsWith("image/")) return cb(null, true);
    cb(new HttpError(400, "Upload must be an image (JPG or PNG)."));
  },
});

// All scan routes require a logged-in user.
router.use(requireAuth);

// Multer runs before the handler; surface its errors (e.g. file too large) cleanly.
function uploadSingle(req, res, next) {
  upload.single("file")(req, res, (err) => {
    if (!err) return next();
    if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
      return next(new HttpError(413, `Image too large (max ${config.maxUploadMb} MB).`));
    }
    next(err);
  });
}

router.post(
  "/",
  uploadSingle,
  asyncHandler(async (req, res) => {
    if (!req.file) throw new HttpError(400, "No image uploaded (field name must be 'file').");

    const wantHeatmap = String(req.body?.explain || "").toLowerCase() === "true";
    const stage = req.body?.stage || undefined;
    const { buffer, originalname, mimetype } = req.file;

    // Ask the ML service. /explain also returns the classification, so one call
    // covers both cases.
    const result = wantHeatmap
      ? await mlExplain(buffer, originalname, mimetype, stage)
      : await mlPredict(buffer, originalname, mimetype);

    const scan = await Scan.create({
      user: req.userId,
      filename: originalname || "",
      prediction: result.prediction,
      label: result.label,
      confidence: result.confidence,
      isConfident: !!result.is_confident,
      threshold: result.threshold ?? 0,
      topK: (result.top_k || []).map((t) => ({
        disease: t.disease,
        label: t.label,
        confidence: t.confidence,
      })),
      stage: wantHeatmap ? result.stage || stage || null : null,
      heatmap: wantHeatmap ? result.heatmap || null : null,
    });

    res.status(201).json({ scan: scan.toDetailJSON() });
  })
);

router.get(
  "/",
  asyncHandler(async (req, res) => {
    const scans = await Scan.find({ user: req.userId })
      .sort({ createdAt: -1 })
      .limit(100); // heatmap excluded by default (select:false)
    res.json({ scans: scans.map((s) => s.toListJSON()) });
  })
);

/** Validate an :id param is a real ObjectId before hitting the DB. */
function requireValidId(req, _res, next) {
  if (!mongoose.isValidObjectId(req.params.id)) {
    return next(new HttpError(400, "Invalid scan id."));
  }
  next();
}

router.get(
  "/:id",
  requireValidId,
  asyncHandler(async (req, res) => {
    // Explicitly pull the heatmap (normally deselected).
    const scan = await Scan.findOne({ _id: req.params.id, user: req.userId }).select("+heatmap");
    if (!scan) throw new HttpError(404, "Scan not found.");
    res.json({ scan: scan.toDetailJSON() });
  })
);

router.delete(
  "/:id",
  requireValidId,
  asyncHandler(async (req, res) => {
    const deleted = await Scan.findOneAndDelete({ _id: req.params.id, user: req.userId });
    if (!deleted) throw new HttpError(404, "Scan not found.");
    res.json({ ok: true });
  })
);

export default router;
