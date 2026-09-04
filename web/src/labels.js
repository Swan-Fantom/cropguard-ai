/**
 * Client-side helpers for display. The server already sends a human-friendly
 * `label`, but this is a fallback if only a raw class name is available.
 */
export function pretty(raw) {
  if (!raw) return "";
  return String(raw)
    .replace(/Corn_\(maize\)___/g, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Detail-mode options for the Grad-CAM heatmap (backend-agnostic labels). */
export const STAGE_OPTIONS = [
  { value: "combined", label: "Combined (balanced)" },
  { value: "penultimate", label: "Finer (sharper detail)" },
  { value: "last", label: "Coarse (most semantic)" },
];
