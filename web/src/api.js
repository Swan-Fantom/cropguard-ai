/**
 * Tiny API client for the CropGuard Node server.
 * All calls go to /api (Vite proxies that to the Node server in dev).
 *
 * The JWT is kept in localStorage. That's fine for a dev/portfolio app and keeps
 * things simple; for production you'd move to an httpOnly cookie so JS can't read
 * the token (mitigates XSS token theft). See STEP4_WEBAPP.md.
 */
const TOKEN_KEY = "cropguard_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(token) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders() {
  const t = getToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function parse(resp) {
  const text = await resp.text();
  const data = text ? JSON.parse(text) : {};
  if (!resp.ok) {
    throw new Error(data.error || `Request failed (${resp.status}).`);
  }
  return data;
}

// --- Auth ---
export function register(email, password, name) {
  return fetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name }),
  }).then(parse);
}

export function login(email, password) {
  return fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  }).then(parse);
}

export function me() {
  return fetch("/api/auth/me", { headers: { ...authHeaders() } }).then(parse);
}

// --- Scans ---
/**
 * Upload an image for diagnosis.
 * @param {File} file
 * @param {object} opts { explain?: boolean, stage?: string }
 */
export function createScan(file, { explain = false, stage } = {}) {
  const form = new FormData();
  form.append("file", file);
  if (explain) {
    form.append("explain", "true");
    if (stage) form.append("stage", stage);
  }
  return fetch("/api/scans", {
    method: "POST",
    headers: { ...authHeaders() }, // don't set Content-Type; browser sets multipart boundary
    body: form,
  }).then(parse);
}

export function listScans() {
  return fetch("/api/scans", { headers: { ...authHeaders() } }).then(parse);
}

export function getScan(id) {
  return fetch(`/api/scans/${id}`, { headers: { ...authHeaders() } }).then(parse);
}

export function deleteScan(id) {
  return fetch(`/api/scans/${id}`, {
    method: "DELETE",
    headers: { ...authHeaders() },
  }).then(parse);
}
