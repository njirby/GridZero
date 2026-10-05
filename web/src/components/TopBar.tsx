import type { GridState, ControlMode, ConnState } from "../types";
import { controls, setMode } from "../api";
import { ModelSelector } from "./ModelSelector";

export function TopBar({ sim, mode, sessionStatus, conn }: {
  sim: GridState | null; mode: ControlMode; sessionStatus: string | null; conn: ConnState;
}) {
  const live = mode === "agent" && sessionStatus === "busy";
  const paused = mode === "paused" || mode === "manual" || sessionStatus === "idle";
  const badge = conn !== "connected" ? "OFFLINE" : live ? "LIVE" : paused ? (mode === "manual" ? "MANUAL" : "PAUSED") : "…";
  return (
    <div className="topbar" data-testid="topbar">
      <div className="brand"><span className="brand-mark">G</span><span><strong>Grid Operator</strong><small>CASE 14 · CONTROL ROOM</small></span></div>
      <span className={`badge badge-${badge.toLowerCase()}`}><i />{badge}</span>
      <div className="tb-metrics">
        <span className="tb-stat"><small>STEP</small><strong>{sim?.t ?? 0}<em> / {sim?.max_t ?? "—"}</em></strong></span>
        <span className="tb-stat"><small>SIM TIME</small><strong>{sim?.sim_clock || "—"}</strong></span>
        <span className="tb-stat"><small>CUMULATIVE REWARD</small><strong>{sim?.cum_reward?.toFixed(1) ?? "—"}</strong></span>
        <span className="tb-stat"><small>PEAK LOADING</small><strong className={(sim?.max_rho ?? 0) >= .9 ? "stat-danger" : (sim?.max_rho ?? 0) >= .7 ? "stat-warn" : ""}>{((sim?.max_rho ?? 0) * 100).toFixed(0)}%</strong></span>
        <span className="tb-stat"><small>LINES DOWN</small><strong>{sim?.n_down ?? 0}</strong></span>
      </div>
      <ModelSelector />
      <div className="tb-controls">
        {mode === "agent"
          ? <button onClick={() => { controls.pause(); setMode("paused"); }}>Pause</button>
          : <button onClick={() => { controls.resume(); setMode("agent"); }}>Resume</button>}
        <button onClick={() => controls.singleStep()}>Step +1</button>
        {mode === "manual"
          ? <button onClick={() => { controls.release(); setMode("agent"); }}>Release</button>
          : <button onClick={() => { controls.takeOver(); setMode("manual"); }}>Take over</button>}
        <button className="danger" onClick={() => controls.reset()}>Reset</button>
      </div>
    </div>
  );
}
