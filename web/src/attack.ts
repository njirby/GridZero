import type { Attack } from "./types";

// Human-readable verb for an attack, derived from the raw action args.
export function attackVerb(a: Attack): string {
  const sls = a.args.set_line_status as Record<string, number> | undefined;
  if (sls) {
    const v = sls[a.line];
    if (v === -1) return "cut";
    if (v === 1) return "re-closed";
  }
  return "toggled";
}

// Short effect annotation, e.g. " -> new overload 1_4_4".
export function attackEffectText(a: Attack): string {
  if (!a.effect) return "";
  const bits: string[] = [];
  if (a.effect.disc_lines.length) bits.push(`tripped ${a.effect.disc_lines.join(", ")}`);
  if (a.effect.new_overloads.length) bits.push(`new overload ${a.effect.new_overloads.join(", ")}`);
  if (!bits.length) return "";
  return ` -> ${bits.join("; ")}`;
}
