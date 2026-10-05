import { create } from "zustand";
import type {
  HarnessEvent, GridState, GridMeta, AgentDelta, ToolCall, ToolResult, TurnEnd,
  SessionStatus, UserAction, OpponentAction, SystemEvt, EpisodeSummary, StepOutcome,
  Turn, TickMark, SeriesPoint, OperatorRow, SystemRow, Attack, ConnState, ControlMode, ActionLogEntry,
} from "./types";

export interface State {
  conn: ConnState;
  meta: GridMeta | null;
  sim: GridState | null;
  mode: ControlMode;
  running: boolean;
  sessionStatus: "busy" | "idle" | "error" | null;
  lastSeq: number;
  needResync: boolean;
  turns: Turn[];
  pending: AgentDelta[];
  rafScheduled: boolean;
  ticks: TickMark[];
  series: SeriesPoint[];
  operators: OperatorRow[];
  systems: SystemRow[];
  attacks: Attack[];
  actionLog: ActionLogEntry[];
  summary: EpisodeSummary | null;
  model: string;
  variant: string;
  modelCards: { id: string; name: string; variants: string[] }[];
}

export function initial(): State {
  return {
    conn: "connecting", meta: null, sim: null,
    mode: "agent", running: true, sessionStatus: null,
    lastSeq: -1, needResync: false,
    turns: [], pending: [], rafScheduled: false,
    ticks: [], series: [], operators: [], systems: [], attacks: [], actionLog: [], summary: null,
    model: "AA-Dense-Blackwell", variant: "default", modelCards: [],
  };
}

function newTurn(id: string): Turn {
  return { id, status: "active", reasoning: "", text: "", tools: [] };
}

function upsertTurn(turns: Turn[], id: string): Turn[] {
  const i = turns.findIndex((t) => t.id === id);
  if (i >= 0) return turns;
  return [...turns, newTurn(id)];
}

// Target line names of an attack action (set_line_status keys / change_line_status entries).
function attackLines(action: Record<string, unknown>): string[] {
  const out: string[] = [];
  const sls = action.set_line_status;
  if (sls && typeof sls === "object" && !Array.isArray(sls)) out.push(...Object.keys(sls));
  const cls = action.change_line_status;
  if (Array.isArray(cls)) out.push(...cls.filter((x): x is string => typeof x === "string"));
  else if (cls && typeof cls === "object") out.push(...Object.keys(cls));
  return out;
}

// Pure reducer: apply one C4 event to state (agent.delta is buffered, not merged).
export function applyEvent(s: State, ev: HarnessEvent): State {
  if (ev.type !== "ping" && ev.seq > s.lastSeq) {
    if (s.lastSeq >= 0 && ev.seq > s.lastSeq + 1) {
      return { ...s, lastSeq: ev.seq, needResync: true };
    }
    s = { ...s, lastSeq: ev.seq };
  }
  switch (ev.type) {
    case "sim.state": {
      const d = ev.data as GridState;
      let series = s.series;
      if (!series.length || series[series.length - 1].t !== d.t) {
        series = [...series, { t: d.t, cum: d.cum_reward, maxRho: d.max_rho, nDown: d.n_down }];
      }
      return { ...s, sim: d, series };
    }
    case "sim.step_outcome": {
      const d = ev.data as StepOutcome;
      const detail = d.action;
      const actionSummary = detail?.summary || (detail?.tool
        ? `${detail.tool}${detail.args && Object.keys(detail.args).length ? ` ${JSON.stringify(detail.args)}` : ""}`
        : detail?.args ? JSON.stringify(detail.args) : "");
      const actionLog = detail ? [...s.actionLog, {
        seq: ev.seq, t: d.t, source: d.source, summary: actionSummary || "Grid action",
        reward: d.reward, illegal: d.illegal, ambiguous: d.ambiguous,
        disc_lines: d.disc_lines, new_overloads: d.new_overloads,
      }].slice(-250) : s.actionLog;
      const tick: TickMark = {
        t: d.t, source: d.source,
        bad: !!d.illegal || d.disc_lines.length > 0,
        overloaded: d.new_overloads.length > 0,
      };
      const turns = s.turns.length
        ? s.turns.map((t, i) => (i === s.turns.length - 1 ? { ...t, outcome: d } : t))
        : s.turns;
      // A step driven by the opponent tags the latest attack with its effect.
      let attacks = s.attacks;
      if (d.source === "opponent" && attacks.length) {
        const effect = { t: d.t, disc_lines: d.disc_lines, new_overloads: d.new_overloads };
        attacks = attacks.map((a, i) => i === attacks.length - 1 ? { ...a, effect } : a);
      }
      return { ...s, ticks: [...s.ticks, tick], turns, attacks, actionLog };
    }
    case "agent.delta":
      return { ...s, pending: [...s.pending, ev.data as AgentDelta], rafScheduled: true };
    case "agent.tool_call": {
      const d = ev.data as ToolCall;
      const turns = upsertTurn(s.turns, d.turn);
      const card = { partId: d.part_id, tool: d.tool, input: d.input, status: "pending" as const };
      return { ...s, turns: turns.map((t) => (t.id === d.turn ? { ...t, tools: [...t.tools, card] } : t)) };
    }
    case "agent.tool_result": {
      const d = ev.data as ToolResult;
      return { ...s, turns: s.turns.map((t) => t.id !== d.turn ? t : {
        ...t,
        tools: t.tools.map((c) => c.partId === d.part_id
          ? { ...c, status: d.status === "error" ? "error" : "completed", output: d.output, durationMs: d.duration_ms,
              input: (d.input && Object.keys(d.input).length) ? d.input : c.input }
          : c),
      }) };
    }
    case "agent.turn_end": {
      const d = ev.data as TurnEnd;
      return { ...s, turns: s.turns.map((t) => t.id !== d.turn ? t : {
        ...t, status: "done",
        stats: { tokens_in: d.tokens_in, tokens_out: d.tokens_out, latency_ms: d.latency_ms, cost_usd: d.cost_usd },
      }) };
    }
    case "session.status": {
      const d = ev.data as SessionStatus;
      return { ...s, sessionStatus: d.status };
    }
    case "user.action": {
      const d = ev.data as UserAction;
      return { ...s, operators: [...s.operators, { id: d.id, cmd: d.cmd, args: d.args, ts: ev.ts }] };
    }
    case "opponent.action": {
      const d = ev.data as OpponentAction;
      const lines = attackLines(d.action);
      if (!lines.length) return s;
      const fresh: Attack[] = lines.map((line) => ({
        id: d.id, line, args: d.action, ts: ev.ts, seq: ev.seq,
      }));
      return { ...s, attacks: [...s.attacks, ...fresh] };
    }
    case "system": {
      const d = ev.data as SystemEvt;
      return { ...s, systems: [...s.systems, { level: d.level, msg: d.msg, ts: ev.ts }] };
    }
    case "episode.summary":
      return { ...s, summary: ev.data as EpisodeSummary };
    case "ping":
      return { ...s, conn: "connected" };
    default:
      return s;
  }
}

// Merge buffered agent deltas into turns (called once per animation frame).
export function flushDeltas(s: State): State {
  if (!s.pending.length) return s;
  let turns = s.turns;
  for (const d of s.pending) {
    turns = upsertTurn(turns, d.turn);
    const field = d.field === "reasoning" ? "reasoning" : "text";
    turns = turns.map((t) => (t.id === d.turn ? { ...t, [field]: t[field] + d.delta } : t));
  }
  return { ...s, turns, pending: [], rafScheduled: false };
}

// ---- zustand store + rAF scheduling ----
interface Store extends State {
  setConn: (c: ConnState) => void;
  setMeta: (m: GridMeta) => void;
  setMode: (m: ControlMode, running?: boolean) => void;
  setModelCards: (cards: { id: string; name: string; variants: string[] }[]) => void;
  setModel: (model: string, variant?: string) => void;
  ingest: (ev: HarnessEvent) => void; // store-level: apply + schedule rAF
  flush: () => void;
  resync: (sim: GridState, lastSeq: number) => void;
  reset: () => void;
}

let rafId: number | null = null;
export function __resetScheduling() {
  rafId = null;
}

const raf = (cb: () => void) =>
  typeof requestAnimationFrame === "function"
    ? requestAnimationFrame(() => cb())
    : (setTimeout(() => cb(), 16) as unknown as number);

export const useStore = create<Store>((set, get) => ({
  ...initial(),
  setConn: (c) => set({ conn: c }),
  setMeta: (m) => set({ meta: m }),
  setMode: (m, running) => set((s) => ({ mode: m, running: running ?? s.running })),
  setModelCards: (cards) => set({ modelCards: cards }),
  setModel: (model, variant) => set({ model, variant: variant ?? "default" }),
  ingest: (ev) => {
    set((s) => applyEvent(s, ev));
    if (ev.type === "agent.delta") {
      if (rafId == null) rafId = raf(() => { rafId = null; get().flush(); });
    }
  },
  flush: () => set((s) => flushDeltas(s)),
  resync: (sim, lastSeq) => set({ sim, lastSeq, needResync: false }),
  reset: () => set({ ...initial() }),
}));
