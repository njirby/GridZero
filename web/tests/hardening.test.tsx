import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { initial, applyEvent, useStore } from "../src/store";
import { ActionSidebar } from "../src/components/ActionSidebar";
import { ErrorBoundary } from "../src/components/ErrorBoundary";
import { TopBar } from "../src/components/TopBar";
import { resync, __resetToken } from "../src/api";
import type { HarnessEvent } from "../src/types";

const call = (seq: number, id: string): HarnessEvent => ({
  seq, type: "agent.tool_call", ts: 0, data: { turn: "t1", part_id: id, tool: "bash", input: {} },
});
const outcome = (seq: number, summary: unknown): HarnessEvent => ({
  seq, type: "sim.step_outcome", ts: 0, data: {
    t: seq, source: "agent", action: { summary, args: {} }, reward: 1, cum_reward: 1, done: false,
    disc_lines: [], illegal: false, ambiguous: false, new_overloads: [],
  },
});

beforeEach(() => {
  vi.unstubAllGlobals();
  useStore.setState({ ...initial() });
  __resetToken();
});

describe("reducer idempotence and gaps", () => {
  it("ignores a replayed seq entirely", () => {
    let s = applyEvent(initial(), call(1, "p1"));
    s = applyEvent(s, call(2, "p2"));
    const again = applyEvent(s, call(1, "p1"));
    expect(again).toBe(s);
    expect(s.turns[0].tools).toHaveLength(2);
  });

  it("dedupes tool cards by part id", () => {
    let s = applyEvent(initial(), call(1, "p1"));
    s = applyEvent(s, call(2, "p1"));
    expect(s.turns[0].tools).toHaveLength(1);
  });

  it("applies the event that reveals a gap and flags resync", () => {
    let s = applyEvent(initial(), call(1, "p1"));
    s = applyEvent(s, call(5, "p2"));
    expect(s.needResync).toBe(true);
    expect(s.lastSeq).toBe(5);
    expect(s.turns[0].tools.map((c) => c.partId)).toEqual(["p1", "p2"]);
  });

  it("caps ticks", () => {
    let s = initial();
    for (let i = 1; i <= 2100; i++) s = applyEvent(s, outcome(i, "x"));
    expect(s.ticks.length).toBe(2000);
  });
});

describe("resync wiring", () => {
  it("fetches /state once, adopts mode/running, clears needResync", async () => {
    const fetchMock = vi.fn(() => Promise.resolve({
      ok: true, json: async () => ({ sim: null, mode: "paused", running: false, last_seq: 9 }),
    }));
    vi.stubGlobal("fetch", fetchMock);
    useStore.setState({ needResync: true });
    await resync();
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const s = useStore.getState();
    expect(s.mode).toBe("paused");
    expect(s.running).toBe(false);
    expect(s.needResync).toBe(false);
  });
});

describe("dict action summary", () => {
  it("renders in the sidebar without crashing", () => {
    useStore.setState({ ...applyEvent(initial(), outcome(1, { set_line_status: { "0_4_1": -1 } })) });
    render(<ActionSidebar />);
    fireEvent.click(screen.getAllByRole("button", { name: /Open action history/ })[0]);
    expect(screen.getByTestId("action-log").textContent).toContain("set_line_status");
  });

  it("error boundary contains a render throw", () => {
    const Bomb = () => { throw new Error("boom"); };
    vi.spyOn(console, "error").mockImplementation(() => {});
    render(<ErrorBoundary><Bomb /></ErrorBoundary>);
    expect(screen.getByRole("alert").textContent).toContain("boom");
  });
});

describe("token-gated controls", () => {
  it("403 surfaces an error and leaves mode unchanged", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok: false, status: 403, json: async () => ({}) })));
    render(<TopBar sim={null} mode="agent" sessionStatus={null} conn="connected" />);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(useStore.getState().error).toContain("token"));
    expect(useStore.getState().mode).toBe("agent");
  });

  it("success flips mode and sends the bearer token", async () => {
    window.history.pushState({}, "", "/?token=abc");
    const fetchMock = vi.fn((_u?: string, _i?: RequestInit) => Promise.resolve({ ok: true, status: 200, json: async () => ({}) }));
    vi.stubGlobal("fetch", fetchMock);
    render(<TopBar sim={null} mode="agent" sessionStatus={null} conn="connected" />);
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(useStore.getState().mode).toBe("paused"));
    expect((fetchMock.mock.calls[0][1]?.headers as Record<string, string>).Authorization).toBe("Bearer abc");
    window.history.pushState({}, "", "/");
  });
});

describe("bands, opponent.step", () => {
  it("band separates >=1.0 from >=0.9", async () => {
    const { band } = await import("../src/components/GridMap");
    expect(band(0.95)).toBe("red");
    expect(band(1.0)).toBe("critical");
  });
  it("opponent.step is recorded as an attack", () => {
    const s = applyEvent(initial(), { seq: 0, type: "opponent.step", ts: 0, data: { t: 3, line: "1_4_4", action: { set_line_status: { "1_4_4": -1 } } } });
    expect(s.attacks).toHaveLength(1);
    expect(s.attacks[0].line).toBe("1_4_4");
  });
});
