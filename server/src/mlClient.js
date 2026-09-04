/**
 * Thin client for the Python FastAPI ML service.
 *
 * Forwards an uploaded image (as a Buffer) to the FastAPI /predict or /explain
 * endpoint using multipart/form-data, and returns the parsed JSON. Uses Node's
 * built-in fetch/FormData/Blob (Node 18+), so no HTTP-client dependency.
 */
import { config } from "./config.js";
import { HttpError } from "./middleware/errors.js";

/**
 * Send an image buffer to a FastAPI endpoint ("/predict" or "/explain").
 * @param {string} endpoint  e.g. "/predict"
 * @param {Buffer} buffer    raw image bytes
 * @param {string} filename  original filename (for the multipart part)
 * @param {string} mimetype  image mime type
 * @param {object} [fields]  extra form fields (e.g. { stage: "penultimate" })
 */
async function postImage(endpoint, buffer, filename, mimetype, fields = {}) {
  const form = new FormData();
  form.append("file", new Blob([buffer], { type: mimetype || "application/octet-stream" }), filename || "upload.jpg");
  for (const [k, v] of Object.entries(fields)) {
    if (v !== undefined && v !== null) form.append(k, String(v));
  }

  let resp;
  try {
    resp = await fetch(`${config.mlServiceUrl}${endpoint}`, {
      method: "POST",
      body: form,
    });
  } catch (err) {
    // Connection refused / DNS / timeout — the Python service is likely down.
    throw new HttpError(
      502,
      `Could not reach the ML service at ${config.mlServiceUrl}. Is the FastAPI ` +
        `server running (uvicorn app:app)? (${err.message})`
    );
  }

  const text = await resp.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }

  if (!resp.ok) {
    // Bubble up FastAPI's error detail with its status code.
    const detail = data?.detail || data?.error || `ML service error (${resp.status}).`;
    throw new HttpError(resp.status === 422 ? 400 : resp.status, detail);
  }
  return data;
}

export function predict(buffer, filename, mimetype) {
  return postImage("/predict", buffer, filename, mimetype);
}

export function explain(buffer, filename, mimetype, stage) {
  return postImage("/explain", buffer, filename, mimetype, { stage });
}
