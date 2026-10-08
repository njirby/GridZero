import { useEffect, useRef } from "react";
import type { Turn, EpisodeSummary } from "../types";
import { OperatorBox } from "./OperatorBox";

function cmdSummary(tool: string, input: Record<string, unknown>): string {
  if (tool === "bash" && typeof input.command === "string") return input.command;
  if (typeof input.command === "string") return input.command;
  return JSON.stringify(input);
}

function ToolCard({ t }: { t: Turn["tools"][number] }) {
  const open = t.status === "error" || t.status === "pending" || (t.output?.length ?? 0) > 200;
  return (
    <div className={`toolcard tc-${t.status}`}>
      <div className="toolcard-head">
        <span className="tc-icon">{t.status === "pending" ? "…" : t.status === "error" ? "×" : "✓"}</span>
        <span className="tc-tool">{t.tool}</span>
        <code className="tc-cmd">{cmdSummary(t.tool, t.input)}</code>
        {t.durationMs != null && <span className="tc-dur">{t.durationMs}ms</span>}
      </div>
      {t.output != null && (
        <pre className={`tc-out ${open ? "" : "tc-out-clamped"}`}>{t.output}</pre>
      )}
    </div>
  );
}

function TurnRow({ turn }: { turn: Turn }) {
  const reasoningLive = turn.status === "active" && turn.reasoning.length > 0;
  return (
    <div className={`turn ${turn.status}`}>
      {turn.reasoning && (
        <div className={`reasoning ${reasoningLive ? "reasoning-live" : "reasoning-done"}`}>
          <span className="reasoning-label">{reasoningLive ? "thinking…" : "thought"}</span>
          <span className="reasoning-text">{turn.reasoning}</span>
        </div>
      )}
      {turn.tools.map((t) => <ToolCard key={t.partId} t={t} />)}
      {turn.text && <div className="turn-text">{turn.text}</div>}
      {turn.outcome && (
        <div className="outcome">
          <span className="oc-t">t={turn.outcome.t}</span>
          <span className="oc-reward">Δr {turn.outcome.reward?.toFixed(2)}</span>
          {turn.outcome.disc_lines.length > 0 && <span className="oc-disc">tripped: {turn.outcome.disc_lines.join(", ")}</span>}
          {turn.outcome.new_overloads.length > 0 && <span className="oc-over">overload: {turn.outcome.new_overloads.join(", ")}</span>}
          {(turn.outcome.illegal || turn.outcome.ambiguous) && <span className="oc-illegal">rejected: illegal/ambiguous</span>}
        </div>
      )}
      {turn.stats && (
        <div className="turn-stats">{turn.stats.tokens_in ?? "?"}→{turn.stats.tokens_out ?? "?"} tok · {((turn.stats.latency_ms ?? 0) / 1000).toFixed(1)}s</div>
      )}
    </div>
  );
}

export function AgentFeed({ turns, summary }: {
  turns: Turn[]; summary: EpisodeSummary | null;
}) {
  const feedRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true); // only follow the stream while the user is near the bottom
  const onScroll = () => {
    const el = feedRef.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };
  useEffect(() => {
    const el = feedRef.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  });

  return (
    <section className="chat-panel">
      <div className="section-heading"><span>AGENT CONVERSATION</span><span className="chat-count">{turns.length} turns</span></div>
        <div className="feed" data-testid="feed" ref={feedRef} onScroll={onScroll}>
      {turns.length === 0 && <div className="chat-empty">The agent’s reasoning and replies will appear here.</div>}
      {turns.map((t) => <TurnRow key={t.id} turn={t} />)}
      {summary && (
        <div className="summary" data-testid="summary">
          Episode ended at t={summary.t} · cum_reward {summary.cum_reward?.toFixed(1)} · {summary.cause}
        </div>
      )}
        </div>
        <OperatorBox />
    </section>
  );
}
