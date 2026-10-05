import { describe, it, expect, vi, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { render, fireEvent, screen, waitFor } from "@testing-library/react";
import { AttackPanel } from "../src/components/AttackPanel";
import { useStore, initial } from "../src/store";
import type { GridState, GridTarget } from "../src/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const EX = join(HERE, "..", "..", "contracts", "examples");
const t50 = JSON.parse(readFileSync(join(EX, "grid-state-t50.json"), "utf8")) as GridState;

const line = (id: string): GridTarget => ({ kind: "line", id });
const gen = (id: string): GridTarget => ({ kind: "gen", id });

function mockFetch() {
  const fetchMock = vi.fn((url?: string, init?: RequestInit) =>
    Promise.resolve({ ok: true, json: async () => ({}) }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.unstubAllGlobals();
  useStore.setState(initial());
  useStore.setState({ sim: t50 });
});

describe("AttackPanel (attack planner)", () => {
  it("shows the launcher (enter attack mode) when not in attack mode", () => {
    render(<AttackPanel attackMode={false} setAttackMode={() => {}} target={null} />);
    expect(screen.getByTestId("attack-mode-toggle").textContent).toContain("Enter attack mode");
    expect(screen.queryByTestId("attack-panel")).toBeNull();
  });

  it("offers 'Stage trip' for an up line and 'Stage restore' for a down line", () => {
    // 1_4_4 is up (rho 1.17) -> trip enabled, restore disabled
    let r = render(<AttackPanel attackMode setAttackMode={() => {}} target={line("1_4_4")} />);
    expect(screen.getByTestId("attack-panel")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Stage trip" }) as HTMLButtonElement).disabled).toBe(false);
    expect((screen.getByRole("button", { name: "Stage restore" }) as HTMLButtonElement).disabled).toBe(true);
    r.unmount();
    // 0_4_1 is down (rho 0) -> restore enabled, trip disabled
    r = render(<AttackPanel attackMode setAttackMode={() => {}} target={line("0_4_1")} />);
    expect((screen.getByRole("button", { name: "Stage trip" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole("button", { name: "Stage restore" }) as HTMLButtonElement).disabled).toBe(false);
    r.unmount();
  });

  it("staging a trip adds it to the draft list and enables Fire", () => {
    render(<AttackPanel attackMode setAttackMode={() => {}} target={line("1_4_4")} />);
    const fireBefore = screen.getByRole("button", { name: /Fire attack/ }) as HTMLButtonElement;
    expect(fireBefore.disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Stage trip" }));
    expect(screen.getByTestId("attack-draft").textContent).toContain("Trip 1_4_4");
    expect(screen.getByRole("button", { name: /Fire attack · 1 action/ })).toBeTruthy();
  });

  it("firing POSTs /sim/attack with the staged set_line_status -1", async () => {
    const fetchMock = mockFetch();
    render(<AttackPanel attackMode setAttackMode={() => {}} target={line("1_4_4")} />);
    fireEvent.click(screen.getByRole("button", { name: "Stage trip" }));
    fireEvent.click(screen.getByRole("button", { name: /Fire attack · 1 action/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/sim/attack");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({ action: { set_line_status: { "1_4_4": -1 } } });
  });

  it("combines multiple staged actions (trip a line + curtail a renewable gen) into one fire", async () => {
    const fetchMock = mockFetch();
    // same component instance across rerenders so the draft (local state) persists
    const r = render(<AttackPanel attackMode setAttackMode={() => {}} target={line("3_4_6")} />);
    fireEvent.click(screen.getByRole("button", { name: "Stage trip" }));
    // switch the selected target to a renewable generator (draft is preserved)
    r.rerender(<AttackPanel attackMode setAttackMode={() => {}} target={gen("gen_5_2")} />);
    fireEvent.click(screen.getByRole("button", { name: /Stage curtailment/ }));
    // fire the 2 combined actions
    fireEvent.click(screen.getByRole("button", { name: /Fire attack · 2 actions/ }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
    const body = JSON.parse(fetchMock.mock.calls[0][1]?.body as string);
    expect(body.action.set_line_status).toEqual({ "3_4_6": -1 });
    expect(body.action.curtail).toEqual({ gen_5_2: 0.5 });
  });
});
