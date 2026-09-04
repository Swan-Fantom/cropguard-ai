/**
 * Diagnose page — the core product screen.
 *  - pick an image
 *  - choose whether to also get a Grad-CAM heatmap, and at what detail
 *  - see the diagnosis, confidence, top-k, and (if requested) the "where the
 *    model looked" overlay
 * Each successful diagnosis is saved server-side to the user's history.
 */
import { useRef, useState } from "react";

import { createScan } from "../api.js";
import { STAGE_OPTIONS } from "../labels.js";
import { ConfidenceBadge, ErrorNote, Spinner, TopKList } from "../components/ui.jsx";

export default function DiagnosePage() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [explain, setExplain] = useState(true);
  const [stage, setStage] = useState("penultimate"); // best detail for the LeViT model
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const fileInputRef = useRef(null);

  function onPickFile(e) {
    const f = e.target.files?.[0] || null;
    setResult(null);
    setError("");
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(f ? URL.createObjectURL(f) : null);
  }

  async function onDiagnose() {
    if (!file) {
      setError("Choose a leaf photo first.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const { scan } = await createScan(file, { explain, stage });
      setResult(scan);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold">Diagnose a corn leaf</h1>
      <p className="mb-6 text-sm text-gray-500">
        Upload a clear photo of a single leaf. The model classifies the disease and can show a
        heatmap of the regions that drove its answer.
      </p>

      <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-center gap-4">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={onPickFile}
            className="block text-sm file:mr-3 file:rounded-md file:border-0 file:bg-brand-600 file:px-4 file:py-2 file:text-white hover:file:bg-brand-700"
          />

          <label className="flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={explain}
              onChange={(e) => setExplain(e.target.checked)}
              className="h-4 w-4 rounded border-gray-300 text-brand-600 focus:ring-brand-500"
            />
            Show heatmap
          </label>

          <label className="flex items-center gap-2 text-sm text-gray-700">
            Detail:
            <select
              value={stage}
              disabled={!explain}
              onChange={(e) => setStage(e.target.value)}
              className="rounded-md border border-gray-300 px-2 py-1 text-sm disabled:opacity-50"
            >
              {STAGE_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>

          <button
            onClick={onDiagnose}
            disabled={busy || !file}
            className="ml-auto rounded-md bg-brand-600 px-5 py-2 font-medium text-white hover:bg-brand-700 disabled:opacity-60"
          >
            {busy ? "Analyzing…" : "Diagnose"}
          </button>
        </div>

        <div className="mt-4">
          {busy && <Spinner label="Running the model…" />}
          <ErrorNote>{error}</ErrorNote>
        </div>

        {(previewUrl || result) && (
          <div className="mt-5 grid gap-5 sm:grid-cols-2">
            {previewUrl && (
              <figure>
                <img
                  src={previewUrl}
                  alt="Uploaded leaf"
                  className="w-full rounded-lg border border-gray-200 object-cover"
                />
                <figcaption className="mt-1 text-center text-xs text-gray-500">
                  Uploaded image
                </figcaption>
              </figure>
            )}
            {result?.heatmap && (
              <figure>
                <img
                  src={result.heatmap}
                  alt="Grad-CAM heatmap"
                  className="w-full rounded-lg border border-gray-200 object-cover"
                />
                <figcaption className="mt-1 text-center text-xs text-gray-500">
                  Where the model looked{result.stage ? ` (${result.stage})` : ""}
                </figcaption>
              </figure>
            )}
          </div>
        )}

        {result && (
          <div className="mt-6 border-t border-gray-100 pt-5">
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-semibold">
                Diagnosis:{" "}
                <span className="text-brand-700">{result.label}</span>
              </h2>
              <ConfidenceBadge isConfident={result.isConfident} />
              <span className="text-lg font-medium text-gray-700">
                {Math.round(result.confidence * 1000) / 10}%
              </span>
            </div>
            {!result.isConfident && (
              <p className="mt-2 text-sm text-amber-700">
                Confidence is below the {Math.round(result.threshold * 100)}% threshold — treat
                this as a suggestion. Try a sharper, well-lit photo of a single leaf filling the
                frame.
              </p>
            )}
            <TopKList topK={result.topK} />
          </div>
        )}
      </div>
    </div>
  );
}
