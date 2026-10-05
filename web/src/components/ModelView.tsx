import { useState } from "react";
import { useStore } from "../store";

export function ModelView() {
  const sim = useStore((s) => s.sim);
  const [, bump] = useState(0);
  // The backend serves the render dir at /render/*; C3.png is the basename.
  const base = sim?.png ? sim.png.split("/").pop() : null;
  const src = base ? `/render/${base}?t=${sim?.t ?? 0}` : null;
  return (
    <div className="modelview" data-testid="modelview">
      {src
        ? <img src={src} alt="model's grid view" className="model-png" />
        : <div className="modelview-empty">No render yet — the agent creates one with `simctl render`.</div>}
      <button onClick={() => bump((n) => n + 1)}>Refresh</button>
    </div>
  );
}
