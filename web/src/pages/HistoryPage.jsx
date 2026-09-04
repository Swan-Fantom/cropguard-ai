/**
 * History page — the user's past diagnoses. Clicking a row loads its full detail
 * (including the stored heatmap, if any) into a side panel.
 */
import { useEffect, useState } from "react";

import { deleteScan, getScan, listScans } from "../api.js";
import { ConfidenceBadge, ErrorNote, Spinner, TopKList } from "../components/ui.jsx";

function formatDate(iso) {
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function HistoryPage() {
  const [scans, setScans] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    setLoading(true);
    setError("");
    try {
      const { scans } = await listScans();
      setScans(scans);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function openScan(id) {
    setDetailLoading(true);
    setError("");
    try {
      const { scan } = await getScan(id);
      setSelected(scan);
    } catch (err) {
      setError(err.message);
    } finally {
      setDetailLoading(false);
    }
  }

  async function onDelete(id, e) {
    e.stopPropagation();
    if (!confirm("Delete this scan?")) return;
    try {
      await deleteScan(id);
      setScans((prev) => prev.filter((s) => s.id !== id));
      if (selected?.id === id) setSelected(null);
    } catch (err) {
      setError(err.message);
    }
  }

  if (loading) return <Spinner label="Loading history…" />;

  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold">Your scan history</h1>
      <p className="mb-6 text-sm text-gray-500">{scans.length} saved diagnoses.</p>

      <ErrorNote>{error}</ErrorNote>

      {scans.length === 0 ? (
        <p className="mt-6 text-gray-500">
          No scans yet. Head to Diagnose to analyze your first leaf.
        </p>
      ) : (
        <div className="grid gap-6 md:grid-cols-2">
          {/* List */}
          <ul className="divide-y divide-gray-100 overflow-hidden rounded-xl border border-gray-200 bg-white">
            {scans.map((s) => (
              <li
                key={s.id}
                onClick={() => openScan(s.id)}
                className={`flex cursor-pointer items-center justify-between px-4 py-3 hover:bg-gray-50 ${
                  selected?.id === s.id ? "bg-brand-50" : ""
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-gray-800">{s.label}</span>
                    <ConfidenceBadge isConfident={s.isConfident} />
                  </div>
                  <div className="text-xs text-gray-500">
                    {Math.round(s.confidence * 1000) / 10}% · {formatDate(s.createdAt)}
                    {s.filename ? ` · ${s.filename}` : ""}
                  </div>
                </div>
                <button
                  onClick={(e) => onDelete(s.id, e)}
                  className="rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-600"
                  title="Delete"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>

          {/* Detail panel */}
          <div className="rounded-xl border border-gray-200 bg-white p-5">
            {detailLoading ? (
              <Spinner label="Loading scan…" />
            ) : selected ? (
              <div>
                <div className="flex items-center gap-3">
                  <h2 className="text-lg font-semibold text-brand-700">{selected.label}</h2>
                  <ConfidenceBadge isConfident={selected.isConfident} />
                  <span className="font-medium text-gray-700">
                    {Math.round(selected.confidence * 1000) / 10}%
                  </span>
                </div>
                <p className="mt-1 text-xs text-gray-500">{formatDate(selected.createdAt)}</p>
                {selected.heatmap && (
                  <figure className="mt-4">
                    <img
                      src={selected.heatmap}
                      alt="Grad-CAM heatmap"
                      className="w-full rounded-lg border border-gray-200"
                    />
                    <figcaption className="mt-1 text-center text-xs text-gray-500">
                      Where the model looked{selected.stage ? ` (${selected.stage})` : ""}
                    </figcaption>
                  </figure>
                )}
                <TopKList topK={selected.topK} />
              </div>
            ) : (
              <p className="text-sm text-gray-500">Select a scan to see its details.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
