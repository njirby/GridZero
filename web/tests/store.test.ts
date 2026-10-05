import { describe, it, expect, beforeEach } from "vitest";
import { useStore, initial, applyEvent, __resetScheduling } from "../src/store";
import type { HarnessEvent, OpponentAction, StepOutcome } from "../src/types";

function delta(seq: number, turn = "t1", field: "text" | "reasoning" = "text", d = "x"): HarnessEvent {
  return { seq, type: "agent.delta", ts: 0, data: { turn, part_id: "p1", field, delta: d } };
}

describe("store rAF batching", () => {
  beforeEach(() => {
    __resetScheduling();
    useStore.setState(initial());
  });

  it("collapses many agent.delta into ONE requestAnimationFrame", () => {
    let calls = 0;
    let cb: (() => void) | null = null;
    (globalThis as any).requestAnimationFrame = (fn: () => void) => { calls++; cb = fn; return 1; };
    const st = useStore.getState();
    for (let i = 0; i < 50; i++) st.ingest(delta(i));
    expect(calls).toBe(1);
    // buffered, not yet merged into a turn
    expect(useStore.getState().turns.find((t) => t.id === "t1")).toBeUndefined();
    cb!();
    expect(useStore.getState().turns.find((t) => t.id === "t1")?.text).toBe("x".repeat(50));
    expect(useStore.getState().pending).toHaveLength(0);
  });

  it("routes reasoning deltas into the reasoning field", () => {
    useStore.getState().ingest(delta(0, "t9", "reasoning", "R"));
    useStore.getState().flush();
    const t = useStore.getState().turns.find((x) => x.id === "t9")!;
    expect(t.reasoning).toBe("R");
    expect(t.text).toBe("");
  });

  it("ingest of a sim.state updates sim immediately (no rAF)", () => {
    useStore.getState().ingest({ seq: 1, type: "sim.state", ts: 0,
      data: { t: 7, n_line: 20 } as any } as unknown as HarnessEvent);
    expect(useStore.getState().sim!.t).toBe(7);
  });
});

function oppEvent(seq: number, action: Record<string, unknown>, ts = 100): HarnessEvent {
  const data: OpponentAction = { action, summary: {} };
  return { seq, type: "opponent.action", ts, data };
}

function outcome(seq: number, partial: Partial<StepOutcome>): HarnessEvent {
  const d: StepOutcome = {
    t: 52, source: "opponent", action: null, reward: 64.9, cum_reward: 64.9, done: false,
    disc_lines: [], illegal: false, ambiguous: false, new_overloads: [], ...partial,
  };
  return { seq, type: "sim.step_outcome", ts: 101, data: d };
}

describe("opponent attacks", () => {
  it("appends to attacks on opponent.action (cut of line 0_4_1)", () => {
    let s = initial();
    expect(s.attacks).toHaveLength(0);
    s = applyEvent(s, oppEvent(1, { set_line_status: { "0_4_1": -1 } }));
    expect(s.attacks).toHaveLength(1);
    const a = s.attacks[0];
    expect(a.line).toBe("0_4_1");
    expect(a.args).toEqual({ set_line_status: { "0_4_1": -1 } });
    expect(a.seq).toBe(1);
    expect(a.ts).toBe(100);
    expect(a.effect).toBeUndefined();
  });

  it("tags the latest attack with the effect of an opponent step_outcome", () => {
    let s = initial();
    s = applyEvent(s, oppEvent(1, { set_line_status: { "0_4_1": -1 } }));
    s = applyEvent(s, outcome(2, { t: 52, new_overloads: ["1_4_4"] }));
    expect(s.attacks).toHaveLength(1);
    expect(s.attacks[0].effect).toEqual({ t: 52, disc_lines: [], new_overloads: ["1_4_4"] });
    // the outcome still lands a tick tagged with the opponent source
    expect(s.ticks).toHaveLength(1);
    expect(s.ticks[0].source).toBe("opponent");
  });

  it("extracts target lines from change_line_status (list form)", () => {
    let s = initial();
    s = applyEvent(s, oppEvent(1, { change_line_status: ["3_4_6"] }));
    expect(s.attacks).toHaveLength(1);
    expect(s.attacks[0].line).toBe("3_4_6");
  });

  it("leaves attacks untouched on a non-opponent step_outcome", () => {
    let s = initial();
    s = applyEvent(s, oppEvent(1, { set_line_status: { "0_4_1": -1 } }));
    s = applyEvent(s, outcome(2, { source: "agent", t: 53, new_overloads: ["1_4_4"] }));
    expect(s.attacks[0].effect).toBeUndefined();
  });
});
