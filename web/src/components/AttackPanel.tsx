import { useMemo, useState } from "react";
import { attack } from "../api";
import { useStore } from "../store";
import type { GridTarget } from "../types";

interface PlannedAction {
  id: string;
  label: string;
  action: Record<string, unknown>;
}

export function AttackPanel({ attackMode = false, setAttackMode = () => {}, target = null }: {
  attackMode?: boolean; setAttackMode?: (enabled: boolean) => void; target?: GridTarget | null;
}) {
  const sim = useStore((s) => s.sim);
  const [draft, setDraft] = useState<PlannedAction[]>([]);
  const [curtailPct, setCurtailPct] = useState(50);
  const gens = sim?.gens ?? [];
  const lines = sim?.lines ?? [];
  const selected = useMemo(() => {
    if (!target) return null;
    if (target.kind === "line") return lines.find((line) => line.name === target.id) ?? null;
    if (target.kind === "gen") return gens.find((gen) => gen.name === target.id) ?? null;
    return sim?.subs.find((sub) => sub.name === target.id) ?? null;
  }, [target, lines, gens, sim?.subs]);

  const stage = (id: string, label: string, action: Record<string, unknown>) => {
    const touches = Object.entries(action).flatMap(([kind, value]) =>
      value && typeof value === "object" && !Array.isArray(value)
        ? Object.keys(value as Record<string, unknown>).map((entity) => `${kind}:${entity}`) : []);
    setDraft((current) => [...current.filter((item) => item.id !== id && !Object.entries(item.action).some(([kind, value]) =>
      value && typeof value === "object" && !Array.isArray(value) && Object.keys(value as Record<string, unknown>).some((entity) => touches.includes(`${kind}:${entity}`)))), { id, label, action }]);
  };

  const fire = async () => {
    const combined: Record<string, unknown> = {};
    for (const item of draft) {
      for (const [key, value] of Object.entries(item.action)) {
        if (value && typeof value === "object" && !Array.isArray(value)) {
          combined[key] = { ...(combined[key] as Record<string, unknown> | undefined), ...(value as Record<string, unknown>) };
        } else combined[key] = value;
      }
    }
    await attack(combined);
    setDraft([]);
    setAttackMode(false);
  };

  const subName = (name: string) => name.replace("sub_", "Sub ");
  const targetDescription = target?.kind === "line" && selected && "or" in selected
    ? `${selected.name} · ${subName(selected.or)} ↔ ${subName(selected.ex)}`
    : target?.kind === "gen" && selected && "sub" in selected
      ? `${selected.name} · ${subName(selected.sub)} · ${selected.p.toFixed(1)} MW`
      : target?.kind === "sub" && selected && "type" in selected
        ? `${selected.name.replace("sub_", "Sub ")} · ${selected.type === "both" ? "generation + load" : selected.type}`
        : null;

  if (!attackMode) return <div className="attack-launcher">
    <button onClick={() => setAttackMode(true)} data-testid="attack-mode-toggle">{draft.length ? `Review attack · ${draft.length} staged` : "Enter attack mode"}</button>
    {draft.length > 0 && <button className="danger" onClick={() => void fire()}>Fire attack</button>}
  </div>;

  return (
    <div className={`attack-panel${attackMode ? " attack-mode-active" : ""}`} data-testid="attack-panel">
      <div className="attack-heading">
        <div><span className="attack-title">ATTACK PLANNER</span><span className="attack-subtitle">Build a set of actions, then fire once</span></div>
        <button className={attackMode ? "attack-mode-button active" : "attack-mode-button"}
          onClick={() => setAttackMode(!attackMode)} data-testid="attack-mode-toggle">
          {attackMode ? "Exit attack mode" : "Enter attack mode"}
        </button>
      </div>

      <>
        <div className="attack-instruction">Click a line, substation, or generator on the map to configure an action.</div>
        <div className="attack-config">
          {!targetDescription && <div className="attack-empty">No component selected yet</div>}
          {targetDescription && <>
            <div className="selected-target"><span className={`target-kind target-${target?.kind}`}>{target?.kind}</span><strong>{targetDescription}</strong></div>
            {target?.kind === "line" && selected && "or" in selected && <div className="target-actions">
              <span className="target-reading">{Math.round(selected.rho * 100)}% loaded · {selected.status}</span>
              <button className="danger" disabled={selected.status !== "up"} onClick={() => stage(`line:${selected.name}`, `Trip ${selected.name}`, { set_line_status: { [selected.name]: -1 } })}>Stage trip</button>
              <button disabled={selected.status !== "down"} onClick={() => stage(`line:${selected.name}`, `Restore ${selected.name}`, { set_line_status: { [selected.name]: 1 } })}>Stage restore</button>
            </div>}
            {target?.kind === "gen" && selected && "sub" in selected && <div className="target-actions">
              <span className="target-reading">{selected.renewable ? "Renewable generator" : selected.redispatchable ? "Dispatchable generator" : "Generator"}</span>
              {selected.renewable ? <>
                <select aria-label="Curtailment level" value={curtailPct} onChange={(event) => setCurtailPct(Number(event.target.value))}>
                  <option value={50}>Cap at 50%</option><option value={0}>Cap at 0%</option>
                </select>
                <button className="danger" onClick={() => stage(`gen:${selected.name}`, `Curtail ${selected.name} to ${curtailPct}%`, { curtail: { [selected.name]: curtailPct / 100 } })}>Stage curtailment</button>
              </> : selected.redispatchable ? <button className="danger" onClick={() => stage(`gen:${selected.name}`, `Reduce ${selected.name} by 5 MW`, { redispatch: { [selected.name]: -5 } })}>Stage −5 MW redispatch</button>
                : <span className="target-reading">No direct attack action is available for this generator.</span>}
            </div>}
            {target?.kind === "sub" && selected && "type" in selected && (() => {
              const connectedLines = lines.filter((line) => line.or === selected.name || line.ex === selected.name);
              const attachedLines = connectedLines.filter((line) => line.status === "up");
              const downLines = connectedLines.filter((line) => line.status === "down");
              const attachedGens = gens.filter((gen) => gen.sub === selected.name);
              const renewable = attachedGens.filter((gen) => gen.renewable);
              const dispatchable = attachedGens.filter((gen) => gen.redispatchable);
              return <div className="target-actions target-actions-wrap">
                <span className="target-reading">{connectedLines.length} connected lines · {attachedGens.length} generators</span>
                <button className="danger" disabled={!attachedLines.length} onClick={() => stage(`sub-lines:${selected.name}`, `Trip ${attachedLines.length} active lines at ${selected.name.replace("sub_", "Sub ")}`, { set_line_status: Object.fromEntries(attachedLines.map((line) => [line.name, -1])) })}>Stage trip {attachedLines.length} active lines</button>
                {downLines.length > 0 && <button onClick={() => stage(`sub-restore:${selected.name}`, `Restore ${downLines.length} down lines at ${selected.name.replace("sub_", "Sub ")}`, { set_line_status: Object.fromEntries(downLines.map((line) => [line.name, 1])) })}>Stage restore down lines</button>}
                {renewable.length > 0 && <button className="danger" onClick={() => stage(`sub-renewable:${selected.name}`, `Curtail renewable generation at ${selected.name.replace("sub_", "Sub ")}`, { curtail: Object.fromEntries(renewable.map((gen) => [gen.name, 0])) })}>Stage curtail renewable generation</button>}
                {dispatchable.length > 0 && <button className="danger" onClick={() => stage(`sub-dispatch:${selected.name}`, `Reduce dispatchable generation by 5 MW at ${selected.name.replace("sub_", "Sub ")}`, { redispatch: Object.fromEntries(dispatchable.map((gen) => [gen.name, -5])) })}>Stage −5 MW per dispatchable generator</button>}
              </div>;
            })()}
          </>}
        </div>
      </>

      <div className="draft-head"><strong>Pending actions</strong><span>{draft.length}</span></div>
      {draft.length > 0 && <div className="draft-list" data-testid="attack-draft">
        {draft.map((item) => <div className="draft-item" key={item.id}><span>{item.label}</span><button aria-label={`Remove ${item.label}`} onClick={() => setDraft((current) => current.filter((entry) => entry.id !== item.id))}>×</button></div>)}
      </div>}
      <div className="attack-actions">
        <button className="danger fire-attack" disabled={!draft.length} onClick={() => void fire()}>Fire attack · {draft.length} {draft.length === 1 ? "action" : "actions"}</button>
        {draft.length > 0 && <button onClick={() => setDraft([])}>Clear</button>}
      </div>
      <div className="attack-note">All staged actions are sent together and advance the simulation by one step.</div>
    </div>
  );
}
