import { useState } from "react";
import { useStore } from "./store";
import { GridMap } from "./components/GridMap";
import { Timeline } from "./components/Timeline";
import { AgentFeed } from "./components/AgentFeed";
import { TopBar } from "./components/TopBar";
import { AttackPanel } from "./components/AttackPanel";
import { ModelView } from "./components/ModelView";
import { ActionSidebar } from "./components/ActionSidebar";
import type { GridTarget } from "./types";

export function App() {
  const sim = useStore((s) => s.sim);
  const meta = useStore((s) => s.meta);
  const mode = useStore((s) => s.mode);
  const sessionStatus = useStore((s) => s.sessionStatus);
  const conn = useStore((s) => s.conn);
  const turns = useStore((s) => s.turns);
  const summary = useStore((s) => s.summary);
  const series = useStore((s) => s.series);
  const ticks = useStore((s) => s.ticks);
  const error = useStore((s) => s.error);
  const setError = useStore((s) => s.setError);
  const [view, setView] = useState<"map" | "model">("map");
  const [attackMode, setAttackMode] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState<GridTarget | null>(null);
  const setAttackModeAndView = (enabled: boolean) => {
    setAttackMode(enabled);
    if (enabled) setView("map");
  };

  return (
    <div className="app">
      {error && <div className="error-banner" role="alert">{error}<button onClick={() => setError(null)}>Dismiss</button></div>}
      <TopBar sim={sim} mode={mode} sessionStatus={sessionStatus} conn={conn} />
      <div className="main">
        <div className="left">
          <div className="viewtoggle">
            <button className={view === "map" ? "on" : ""} onClick={() => setView("map")}>Map</button>
            <button className={view === "model" ? "on" : ""} onClick={() => setView("model")}>Model's PNG</button>
          </div>
          <AttackPanel attackMode={attackMode} setAttackMode={setAttackModeAndView} target={selectedTarget} />
          {view === "map" && sim ? <GridMap state={sim} meta={meta} attackMode={attackMode}
            selectedTarget={selectedTarget} onSelect={setSelectedTarget} /> : <ModelView />}
        </div>
        <div className="right">
          <AgentFeed turns={turns} summary={summary} />
        </div>
      </div>
      <ActionSidebar />
      <div className="bottom">
        <Timeline series={series} ticks={ticks} />
      </div>
    </div>
  );
}
