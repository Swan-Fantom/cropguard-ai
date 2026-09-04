/**
 * Scan model — one saved diagnosis belonging to a user.
 *
 * We store the ML result (prediction, confidence, top-k) plus, optionally, the
 * Grad-CAM heatmap so a past scan can be re-viewed with its explanation without
 * re-running the model. The heatmap is a base64 data URL and can be large, so
 * it's excluded from list queries via `select: false` and only pulled on demand.
 */
import mongoose from "mongoose";

const classPredictionSchema = new mongoose.Schema(
  {
    disease: String, // raw class name (matches training labels)
    label: String, // human-friendly
    confidence: Number, // 0..1
  },
  { _id: false }
);

const scanSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
      index: true,
    },
    filename: { type: String, default: "" },
    prediction: { type: String, required: true }, // raw top-1 class
    label: { type: String, required: true }, // human-friendly top-1
    confidence: { type: Number, required: true }, // 0..1
    isConfident: { type: Boolean, default: false },
    threshold: { type: Number, default: 0 },
    topK: { type: [classPredictionSchema], default: [] },

    // Explainability (present only when the scan was run with /explain).
    stage: { type: String, default: null }, // grad-cam stage used
    heatmap: { type: String, default: null, select: false }, // base64 data URL, large
  },
  { timestamps: true }
);

/** List-friendly shape (no heavy heatmap payload). */
scanSchema.methods.toListJSON = function toListJSON() {
  return {
    id: this._id.toString(),
    filename: this.filename,
    prediction: this.prediction,
    label: this.label,
    confidence: this.confidence,
    isConfident: this.isConfident,
    threshold: this.threshold,
    topK: this.topK,
    stage: this.stage,
    hasHeatmap: this.stage != null,
    createdAt: this.createdAt,
  };
};

/** Full shape including the heatmap (when it was loaded). */
scanSchema.methods.toDetailJSON = function toDetailJSON() {
  return { ...this.toListJSON(), heatmap: this.heatmap || null };
};

export const Scan = mongoose.model("Scan", scanSchema);
