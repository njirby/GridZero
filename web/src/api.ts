import { useStore } from "./store";
import type { ControlMode } from "./types";

// Operator token: taken once from the page URL (?token=), kept in memory and
// (best-effort) sessionStorage so a reload without the query still works.
let token: string | null = null;
export function getToken(): string {
  if (token != null) return token;
  let t = "";
  try { t = new URLSearchParams(window.location.search).get("token") ?? ""; } catch { /* no window */ }
  try {
    if (t) sessionStorage.setItem("operator_token", t);
    else t = sessionStorage.getItem("operator_token") ?? "";
  } catch { /* storage unavailable */ }
  token = t;
  return token;
}
export function __resetToken() { token = null; }

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

// Append ?token= to a URL (for EventSource, which cannot set headers).
export function withToken(url: string): string {
  const t = getToken();
  if (!t) return url;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(t)}`;
}

// POST JSON; resolves true only on a 2xx response. Failures (403 without a token,
// 5xx, network) are surfaced via the store's error banner.
async function post(path: string, body: unknown, what: string): Promise<boolean> {
  const { setError } = useStore.getState();
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      setError(res.status === 403
        ? `${what} rejected: operator token required (open the page with ?token=...)`
        : `${what} failed (HTTP ${res.status})`);
      return false;
    }
    setError(null);
    return true;
  } catch {
    setError(`${what} failed: backend unreachable`);
    return false;
  }
}

export function control(cmd: string, args: Record<string, unknown> = {}) {
  return post("/control", { cmd, args }, `Control "${cmd}"`);
}

// Adversarial action: same envelope as /sim/act, tagged source="opponent".
// The defender (agent) is never told it came from the human.
export function attack(action: Record<string, unknown>) {
  return post("/sim/attack", { action }, "Attack");
}

// Local mode only changes once the backend accepted the command.
export async function setMode(mode: ControlMode, send: () => Promise<boolean>) {
  if (await send()) useStore.getState().setMode(mode, mode !== "paused");
}

export const controls = {
  pause: () => control("pause"),
  resume: () => control("resume"),
  singleStep: () => control("single_step"),
  takeOver: () => control("take_over"),
  release: () => control("release"),
  reset: () => control("reset"),
  setModel: (model: string, variant?: string) =>
    control("set_model", { model, ...(variant && variant !== "default" ? { variant } : {}) }),
  instruction: (text: string) => control("instruction", { text }),
  manualAction: (action: Record<string, unknown>) => control("manual_action", { args: action }),
};

export interface ModelCard { id: string; name: string; variants: string[]; }

// Load the available models + reasoning variants from the backend.
export async function loadModels(): Promise<ModelCard[]> {
  try {
    const r = await fetch("/models", { headers: authHeaders() });
    return (await r.json()) as ModelCard[];
  } catch {
    return [];
  }
}

// Fetch the currently-selected model/variant (for the selector's initial state).
export async function loadCurrentModel(): Promise<{ model: string; variant: string } | null> {
  try {
    const r = await fetch("/model", { headers: authHeaders() });
    const d = await r.json();
    return { model: d.model, variant: d.variant || "default" };
  } catch {
    return null;
  }
}

// Resync after a detected gap (and on startup): adopt /state's sim, mode and running flag.
export async function resync() {
  try {
    const r = await fetch("/state", { headers: authHeaders() });
    if (!r.ok) throw new Error(String(r.status));
    const d = await r.json();
    useStore.getState().resync(d.sim ?? null, d.mode, d.running);
  } catch {
    useStore.getState().resync(null); // clear the flag; the next gap retries
  }
}
