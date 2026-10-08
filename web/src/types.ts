// C3 — GRID-STATE (see contracts/C3-grid-state.schema.json)
export interface Line {
  id: number; name: string; or: string; ex: string;
  rho: number; p_or: number; p_ex: number;
  status: "up" | "down" | "cooldown" | "maintenance";
  overflow: boolean; cooldown: number; maint: number;
}
export interface Sub {
  id: number; name: string; x: number; y: number;
  type: "gen" | "load" | "both" | "other"; p: number;
}
export interface Gen {
  id: number; name: string; sub: string; p: number;
  renewable?: boolean; redispatchable?: boolean;
}
export interface LastAction {
  source: "agent" | "user" | "auto" | "system";
  summary: string | Record<string, unknown>; args: Record<string, unknown>; t: number; png?: string;
}
export interface GridState {
  schema_version: number; env: string;
  t: number; max_t: number;
  reward: number; cum_reward: number; done: boolean; cause: string | null;
  delta_min: number; sim_clock: string;
  n_line: number; n_sub: number; n_gen: number;
  max_rho: number; n_down: number; n_overflow: number;
  lines: Line[]; subs: Sub[]; gens: Gen[];
  alarms: string[]; last_action: LastAction | null; png: string;
}

// static per-env layout (GET /api/grid/meta)
export interface MetaLine { id: number; name: string; or: string; ex: string; thermal_limit: number }
export interface GridMeta {
  schema_version: number; env: string;
  n_line: number; n_sub: number; n_gen: number; n_load: number; delta_min: number;
  subs: Sub[]; lines: MetaLine[]; gens: { id: number; name: string; sub: string }[];
}

// C4 — EVENT-STREAM (see contracts/C4-event-stream.md)
export type EventType =
  | "sim.state" | "sim.step_outcome" | "session.status"
  | "agent.delta" | "agent.tool_call" | "agent.tool_result" | "agent.turn_end"
  | "user.action" | "opponent.action" | "opponent.step" | "system" | "episode.summary" | "ping";

export interface StepOutcome {
  t: number; source: string;
  action: { tool?: string; args?: Record<string, unknown>; summary?: string | Record<string, unknown> } | null;
  reward: number; cum_reward: number; done: boolean;
  disc_lines: string[]; illegal: boolean; ambiguous: boolean;
  new_overloads: string[]; predicted_disc_lines?: string[];
}
export interface ActionLogEntry {
  seq: number; t: number; source: string; summary: string | Record<string, unknown>;
  reward: number; illegal: boolean; ambiguous: boolean;
  disc_lines: string[]; new_overloads: string[];
}
export type GridTarget = { kind: "line" | "sub" | "gen"; id: string };
export interface AgentDelta {
  turn: string; message_id?: string; part_id: string;
  field: "text" | "reasoning"; delta: string;
}
export interface ToolCall { turn: string; part_id: string; tool: string; input: Record<string, unknown> }
export interface ToolResult {
  turn: string; part_id: string; tool: string;
  status: "completed" | "error" | "pending"; output: string; duration_ms: number;
  input?: Record<string, unknown>;
}
export interface TurnEnd {
  turn: string; tokens_in?: number; tokens_out?: number; reasoning_tokens?: number;
  latency_ms?: number; cost_usd?: number; final_action?: { tool: string; summary: string };
}
export interface SessionStatus { status: "busy" | "idle" | "error"; title?: string }
export interface OpponentAction {
  id?: string;
  action: Record<string, unknown>;
  summary: Record<string, unknown>;
}
export interface OpponentStep {
  t: number; kind?: "start" | "end"; line?: string; action: Record<string, unknown>;
}
export interface UserAction { id: string; cmd: string; args: Record<string, unknown> }
export interface SystemEvt { level: "info" | "warn" | "error"; msg: string }
export interface EpisodeSummary {
  t: number; cum_reward: number; cause: string; peak_max_rho: number; n_down: number;
  n_topology_changes: number; n_user_actions: number; n_instructions: number;
  duration_s: number; llm?: { turns: number; tokens_in: number; tokens_out: number; cost_usd: number };
}

export interface HarnessEvent {
  seq: number; type: EventType; ts: number;
  data: GridState | StepOutcome | AgentDelta | ToolCall | ToolResult | TurnEnd |
        SessionStatus | UserAction | OpponentAction | OpponentStep | SystemEvt | EpisodeSummary | Record<string, unknown>;
}

// UI models
export interface ToolCard {
  partId: string; tool: string; input: Record<string, unknown>;
  status: "pending" | "completed" | "error"; output?: string; durationMs?: number;
}
export interface Turn {
  id: string; status: "active" | "done";
  reasoning: string; text: string;
  tools: ToolCard[];
  stats?: { tokens_in?: number; tokens_out?: number; latency_ms?: number; cost_usd?: number };
  outcome?: StepOutcome;
}
export interface TickMark { t: number; source: string; bad: boolean; overloaded: boolean }
export interface SeriesPoint { t: number; cum: number; maxRho: number; nDown: number }
export interface Attack {
  id?: string;
  line: string;
  args: Record<string, unknown>;
  ts: number;
  seq: number;
  effect?: { t: number; disc_lines: string[]; new_overloads: string[] };
}

export type ConnState = "connecting" | "connected" | "reconnecting" | "disconnected";
export type ControlMode = "agent" | "manual" | "paused";
