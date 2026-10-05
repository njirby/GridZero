import { useEffect, useRef } from "react";
import type { GridState, GridMeta, GridTarget } from "../types";

export function band(rho: number): "green" | "amber" | "red" {
  if (rho >= 0.9) return "red";
  if (rho >= 0.7) return "amber";
  return "green";
}

const BAND_COLOR: Record<string, string> = {
  green: "#22c55e", amber: "#f59e0b", red: "#ef4444",
};

export function GridMap({ state, meta, attackMode = false, selectedTarget = null, onSelect }: {
  state: GridState; meta?: GridMeta | null; attackMode?: boolean;
  selectedTarget?: GridTarget | null; onSelect?: (target: GridTarget) => void;
}) {
  const prev = useRef<Record<string, string>>({});
  const subs = state.subs;
  const pos = new Map<string, { x: number; y: number }>();
  for (const s of subs) pos.set(s.name, { x: s.x, y: s.y });

  // viewBox from sub coords ±8%
  const xs = subs.map((s) => s.x), ys = subs.map((s) => s.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const padX = (maxX - minX || 1) * 0.08, padY = (maxY - minY || 1) * 0.08;
  const vb = `${minX - padX} ${minY - padY} ${maxX - minX + 2 * padX} ${maxY - minY + 2 * padY}`;

  // flash tracking
  useEffect(() => {
    for (const l of state.lines) {
      const sig = `${band(l.rho)}:${l.status}`;
      if (prev.current[l.name] && prev.current[l.name] !== sig) {
        const el = document.querySelector(`[data-line="${CSS.escape(l.name)}"]`);
        el?.classList.add("flash");
        setTimeout(() => el?.classList.remove("flash"), 300);
      }
      prev.current[l.name] = sig;
    }
  });

  const thermal = new Map((meta?.lines ?? []).map((l) => [l.name, l.thermal_limit]));
  const genNames = new Map<string, string[]>();
  for (const gen of state.gens) genNames.set(gen.sub, [...(genNames.get(gen.sub) ?? []), gen.name]);

  return (
    <div className="gridmap" data-testid="gridmap">
      <svg viewBox={vb} preserveAspectRatio="xMidYMid meet" width="100%" height="100%">
        {state.lines.map((l) => {
          const o = pos.get(l.or), e = pos.get(l.ex);
          if (!o || !e) return null;
          const b = band(l.rho);
          const down = l.status !== "up";
          const cls = `line line-${b} ${down ? "line-" + l.status : ""}`;
          return (
            <line
              key={l.id}
              data-line={l.name}
              data-name={l.name}
              data-status={l.status}
              data-band={b}
              data-rho={l.rho}
              data-selected={selectedTarget?.kind === "line" && selectedTarget.id === l.name ? "true" : undefined}
              className={`${cls}${attackMode ? " attack-target" : ""}`}
              x1={o.x} y1={o.y} x2={e.x} y2={e.y}
              stroke={down ? "#64748b" : BAND_COLOR[b]}
              strokeWidth={attackMode ? Math.max(10, 2 + 4 * Math.min(l.rho, 1.3)) : 2 + 4 * Math.min(l.rho, 1.3)}
              strokeDasharray={l.status === "down" ? "6 4" : l.status === "maintenance" ? "2 4" : undefined}
              opacity={l.status === "down" ? 0.5 : 1}
              onClick={() => attackMode && onSelect?.({ kind: "line", id: l.name })}
              role={attackMode ? "button" : undefined}
              tabIndex={attackMode ? 0 : undefined}
              aria-label={attackMode ? `Select line ${l.name}, ${l.status}, ${Math.round(l.rho * 100)} percent loaded` : undefined}
              onKeyDown={(event) => { if (attackMode && (event.key === "Enter" || event.key === " ")) onSelect?.({ kind: "line", id: l.name }); }}
            >
              <title>{l.name}: {(l.rho * 100).toFixed(1)}%{thermal.has(l.name) ? ` / ${thermal.get(l.name)} A` : ""} [{l.status}]</title>
            </line>
          );
        })}
        {subs.map((s) => {
          const r = 7 + Math.min(3, Math.abs(s.p) / 30);
          const role = s.type === "both" ? "GEN + LOAD" : s.type === "gen" ? "GENERATOR" : s.type === "load" ? "LOAD" : "TRANSIT";
          const mw = `${s.p > 0 ? "+" : ""}${s.p.toFixed(1)} MW`;
          const attachedGens = genNames.get(s.name) ?? [];
          return (
            <g key={s.id} data-sub={s.name} data-selected={selectedTarget?.kind === "sub" && selectedTarget.id === s.name ? "true" : undefined}
              className={attackMode ? "attack-target" : ""} onClick={() => attackMode && onSelect?.({ kind: "sub", id: s.name })}
              role={attackMode ? "button" : undefined} tabIndex={attackMode ? 0 : undefined}
              aria-label={attackMode ? `Select ${s.name}, ${role}` : undefined}
              onKeyDown={(event) => { if (attackMode && (event.key === "Enter" || event.key === " ")) onSelect?.({ kind: "sub", id: s.name }); }}>
              <rect x={s.x - r} y={s.y - r} width={r * 2} height={r * 2} rx="2.5"
                className={`sub node-station sub-${s.type}`} fill="#101a23" stroke="#d0dbe4" strokeWidth={2.5} />
              <text x={s.x} y={s.y + 3.5} className="node-core" textAnchor="middle">{s.name.replace("sub_", "")}</text>
              {s.type === "load" || s.type === "both" ? <g className="node-kind load-kind" transform={`translate(${s.x + r + 4},${s.y - 2})`}><circle r="6" /><text y="2.5" textAnchor="middle">L</text></g> : null}
              {(s.type === "gen" || s.type === "both") && <g className="node-kind gen-kind" transform={`translate(${s.x + r + 4},${s.y + 10})`}><path d="M 0 -7 L 7 5 L -7 5 Z" /><text y="3" textAnchor="middle">G</text></g>}
              <text x={s.x} y={s.y - r - 5} className="sublabel subname" textAnchor="middle">SUBSTATION</text>
              <text x={s.x} y={s.y + r + 12} className="sublabel subinfo" textAnchor="middle">{mw}</text>
              <title>{`${s.name} · ${role} · net injection ${mw}${attachedGens.length ? ` · generators: ${attachedGens.join(", ")}` : ""}`}</title>
            </g>
          );
        })}
        {state.gens.map((gen) => {
          const station = pos.get(gen.sub);
          if (!station) return null;
          const siblings = state.gens.filter((item) => item.sub === gen.sub);
          const index = siblings.findIndex((item) => item.name === gen.name);
          return <g key={gen.id} data-gen={gen.name} data-selected={selectedTarget?.kind === "gen" && selectedTarget.id === gen.name ? "true" : undefined}
            className={`gen-marker${attackMode ? " attack-target" : ""}`} transform={`translate(${station.x - 16},${station.y + (index - (siblings.length - 1) / 2) * 12})`}
            onClick={(event) => { if (!attackMode) return; event.stopPropagation(); onSelect?.({ kind: "gen", id: gen.name }); }}
            role={attackMode ? "button" : undefined} tabIndex={attackMode ? 0 : undefined}
            aria-label={attackMode ? `Select generator ${gen.name}, ${gen.p.toFixed(1)} megawatts` : undefined}
            onKeyDown={(event) => { if (attackMode && (event.key === "Enter" || event.key === " ")) { event.stopPropagation(); onSelect?.({ kind: "gen", id: gen.name }); } }}>
            <title>{`${gen.name} · ${gen.renewable ? "renewable" : "dispatchable"} · ${gen.p.toFixed(1)} MW at ${gen.sub}`}</title>
            <path className="gen-tag" d="M -5 -5 L 5 -5 L 8 0 L 5 5 L -5 5 L -8 0 Z" />
            <text y="2.5" textAnchor="middle">G</text>
          </g>;
        })}
      </svg>
      <div className="legend">
        <span><i style={{ background: BAND_COLOR.green }} /> &lt;70%</span>
        <span><i style={{ background: BAND_COLOR.amber }} /> 70–90%</span>
        <span><i style={{ background: BAND_COLOR.red }} /> ≥90%</span>
        <span><i className="dash" /> down</span>
        <span><b className="legend-sub">12</b> substation</span>
        <span><b className="legend-load">L</b> load</span>
        <span><b className="legend-gen">G</b> generator</span>
        <span>MW = net injection</span>
      </div>
    </div>
  );
}
