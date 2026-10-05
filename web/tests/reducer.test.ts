import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { initial, applyEvent, flushDeltas, type State } from "../src/store";
import type { HarnessEvent } from "../src/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const EX = join(HERE, "..", "..", "contracts", "examples");

// Load the recorded SSE stream, inlining the "<<" C3 placeholders the same way
// the mock server does (sim.state frames reference a grid-state-tX.json).
function loadEvents(): HarnessEvent[] {
  const raw = readFileSync(join(EX, "event-stream.ndjson"), "utf8");
  return raw.split("\n").filter(Boolean).map((line) => {
    const ev = JSON.parse(line) as HarnessEvent;
    const d = ev.data as unknown;
    if (ev.type === "sim.state" && typeof d === "string" && d.startsWith("<<C3")) {
      const ref = d.match(/grid-state-(t\d+)\.json/)!;
      ev.data = JSON.parse(readFileSync(join(EX, `grid-state-${ref[1]}.json`), "utf8"));
    }
    return ev;
  });
}

function replay(): State {
  let s = initial();
  for (const ev of loadEvents()) s = applyEvent(s, ev);
  return flushDeltas(s);
}

describe("reducer", () => {
  it("applies the latest sim.state (t=50 after the disconnect)", () => {
    const s = replay();
    expect(s.sim).toBeTruthy();
    expect(s.sim!.t).toBe(50);
    expect(s.sim!.n_line).toBe(20);
    expect(s.sim!.lines).toHaveLength(20);
  });

  it("builds two turns with accumulated reasoning, text, and tool cards", () => {
    const s = replay();
    expect(s.turns).toHaveLength(2);
    const t1 = s.turns.find((t) => t.id === "t-001")!;
    expect(t1.reasoning).toContain("grid state");
    expect(t1.text).toContain("monitor");
    expect(t1.tools).toHaveLength(1);
    expect(t1.tools[0].tool).toBe("bash");
    expect(t1.tools[0].status).toBe("completed");
    expect(t1.tools[0].output).toContain("t=0");

    const t2 = s.turns.find((t) => t.id === "t-002")!;
    expect(t2.tools).toHaveLength(2); // render + act
    expect(t2.tools.some((c) => (c.output ?? "").includes("new_overloads"))).toBe(true);
    expect(t2.text).toContain("117%");
  });

  it("captures the sim.step_outcome and tick", () => {
    const s = replay();
    expect(s.ticks.length).toBe(1);
    expect(s.ticks[0].t).toBe(50);
    expect(s.ticks[0].source).toBe("agent");
    // outcome attached to the last active turn (t-002)
    expect(s.turns.find((t) => t.id === "t-002")!.outcome?.new_overloads).toContain("1_4_4");
  });

  it("tracks series + final session status (idle)", () => {
    const s = replay();
    expect(s.series.length).toBe(2); // t=0 and t=50 snapshots
    expect(s.sessionStatus).toBe("idle");
    expect(s.lastSeq).toBe(488);
  });

  it("flags a gap as needResync", () => {
    let s = initial();
    s = applyEvent(s, loadEvents()[0]);      // seq 470
    const jump = { seq: 500, type: "sim.state", ts: 1, data: s.sim! } as HarnessEvent;
    s = applyEvent(s, jump);                 // 470 -> 500 gap
    expect(s.needResync).toBe(true);
    expect(s.lastSeq).toBe(500);
  });

  it("does not flag a resync on the very first event", () => {
    const s = applyEvent(initial(), loadEvents()[0]); // seq 470 from lastSeq=-1
    expect(s.needResync).toBe(false);
  });
});
