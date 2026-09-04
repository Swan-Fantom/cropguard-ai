/**
 * Small shared UI pieces used across pages.
 */
import { pretty } from "../labels.js";

/** Colored badge for a confidence level. */
export function ConfidenceBadge({ isConfident }) {
  return isConfident ? (
    <span className="rounded-full bg-brand-100 px-2.5 py-0.5 text-xs font-medium text-brand-800">
      Confident
    </span>
  ) : (
    <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800">
      Low confidence
    </span>
  );
}

/** A horizontal probability bar (one per class in top-k). */
export function ProbBar({ label, confidence }) {
  const pct = Math.round(confidence * 1000) / 10;
  return (
    <div className="flex items-center gap-3 py-1">
      <div className="w-40 shrink-0 text-sm text-gray-700">{label}</div>
      <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-gray-100">
        <div
          className="absolute inset-y-0 left-0 rounded-full bg-brand-500"
          style={{ width: `${Math.max(pct, 1.5)}%` }}
        />
      </div>
      <div className="w-14 shrink-0 text-right text-sm tabular-nums text-gray-600">{pct}%</div>
    </div>
  );
}

/** Full top-k prediction list. */
export function TopKList({ topK }) {
  return (
    <div className="mt-2">
      {topK.map((t) => (
        <ProbBar key={t.disease} label={t.label || pretty(t.disease)} confidence={t.confidence} />
      ))}
    </div>
  );
}

export function Spinner({ label = "Working..." }) {
  return (
    <div className="flex items-center gap-2 text-sm text-gray-500">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-brand-500" />
      {label}
    </div>
  );
}

export function ErrorNote({ children }) {
  if (!children) return null;
  return (
    <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
      {children}
    </div>
  );
}
