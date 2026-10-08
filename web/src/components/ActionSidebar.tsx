import { useState } from "react";
import { useStore } from "../store";

// Backends send summaries as plain strings or as dicts of applied effects.
export function fmtSummary(v: string | Record<string, unknown> | null | undefined): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  try { return JSON.stringify(v); } catch { return String(v); }
}

export function ActionSidebar() {
  const actionLog = useStore((s) => s.actionLog);
  const [open, setOpen] = useState(false);
  const partyLabel = (source: string) => source === "opponent" ? "ATTACKER" : source === "user" ? "HUMAN" : source.toUpperCase();

  return <aside className={`actions-sidebar${open ? " expanded" : ""}`}>
    <button className="actions-rail" aria-expanded={open} aria-label={open ? "Close action history" : "Open action history"} onClick={() => setOpen(!open)}>
      <span className="rail-icon">≡</span><span>GRID ACTIONS</span><b>{actionLog.length}</b>
    </button>
    {open && <div className="actions-drawer">
      <div className="section-heading"><span>RUNNING ACTION LOG</span><button aria-label="Close action history" onClick={() => setOpen(false)}>×</button></div>
      <div className="action-log" data-testid="action-log">
        {actionLog.length === 0 && <div className="action-log-empty">Actions from the agent, operator, and attacker will appear here.</div>}
        {[...actionLog].reverse().map((item) => <article className={`action-entry source-${item.source}`} key={`${item.seq}-${item.t}`}>
          <div className="action-entry-top"><span className={`party-badge party-${item.source}`}>{partyLabel(item.source)}</span><span className="action-step">STEP {item.t}</span><span className="action-reward">Δ {item.reward.toFixed(1)}</span></div>
          <div className="action-summary">{fmtSummary(item.summary)}</div>
          {(item.disc_lines.length > 0 || item.new_overloads.length > 0 || item.illegal || item.ambiguous) && <div className="action-effects">
            {item.disc_lines.length > 0 && <span className="effect-bad">Trip: {item.disc_lines.join(", ")}</span>}
            {item.new_overloads.length > 0 && <span className="effect-warn">Overload: {item.new_overloads.join(", ")}</span>}
            {(item.illegal || item.ambiguous) && <span className="effect-bad">Rejected</span>}
          </div>}
        </article>)}
      </div>
    </div>}
  </aside>;
}
