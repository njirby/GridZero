import { useStore } from "./store";
import type { ControlMode } from "./types";

export async function control(cmd: string, args: Record<string, unknown> = {}) {
  try {
    await fetch("/control", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd, args }),
    });
  } catch {
    /* control is best-effort; the /event stream reflects the real state */
  }
}

// Adversarial action: same envelope as /sim/act, tagged source="opponent".
// The defender (agent) is never told it came from the human.
export async function attack(action: Record<string, unknown>) {
  try {
    await fetch("/sim/attack", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
  } catch {
    /* attack is best-effort; the /event stream reflects the real state */
  }
}

export function setMode(mode: ControlMode) {
  useStore.getState().setMode(mode, mode !== "paused");
}

export const controls = {
  pause: () => control("pause"),
  resume: () => control("resume"),
  singleStep: () => control("single_step"),
  takeOver: () => control("take_over"),
  release: () => control("release"),
  reset: () => control("reset"),
  resetWithModel: (model: string, variant?: string) =>
    control("reset", { model, ...(variant && variant !== "default" ? { variant } : {}) }),
  setModel: (model: string, variant?: string) =>
    control("set_model", { model, ...(variant && variant !== "default" ? { variant } : {}) }),
  instruction: (text: string) => control("instruction", { text }),
  manualAction: (action: Record<string, unknown>) => control("manual_action", { args: action }),
};

export interface ModelCard { id: string; name: string; variants: string[]; }

// Load the available models + reasoning variants from the backend.
export async function loadModels(): Promise<ModelCard[]> {
  try {
    const r = await fetch("/models");
    return (await r.json()) as ModelCard[];
  } catch {
    return [];
  }
}

// Fetch the currently-selected model/variant (for the selector's initial state).
export async function loadCurrentModel(): Promise<{ model: string; variant: string } | null> {
  try {
    const r = await fetch("/model");
    const d = await r.json();
    return { model: d.model, variant: d.variant || "default" };
  } catch {
    return null;
  }
}

// Resync after a detected gap: fetch /state and adopt its sim + last_seq.
export async function resync() {
  try {
    const r = await fetch("/state");
    const d = await r.json();
    useStore.getState().resync(d.sim, d.last_seq);
  } catch {
    /* retry on next gap */
  }
}
