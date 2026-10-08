import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { render } from "@testing-library/react";
import { GridMap } from "../src/components/GridMap";
import type { GridState } from "../src/types";

const HERE = dirname(fileURLToPath(import.meta.url));
const EX = join(HERE, "..", "..", "contracts", "examples");
const t50 = JSON.parse(readFileSync(join(EX, "grid-state-t50.json"), "utf8")) as GridState;
const t0 = JSON.parse(readFileSync(join(EX, "grid-state-t0.json"), "utf8")) as GridState;

describe("GridMap", () => {
  it("renders one <line> per grid2op line (20)", () => {
    const { container } = render(<GridMap state={t50} />);
    expect(container.querySelectorAll("line[data-name]")).toHaveLength(20);
  });

  it("marks the disconnected line 0_4_1 as down (dashed)", () => {
    const { container } = render(<GridMap state={t50} />);
    const el = container.querySelector('line[data-name="0_4_1"]');
    expect(el).toBeTruthy();
    expect(el!.getAttribute("data-status")).toBe("down");
    expect(el!.getAttribute("class")).toContain("line-down");
  });

  it("colors the overloaded line 1_4_4 (rho>1) critical", () => {
    const { container } = render(<GridMap state={t50} />);
    const el = container.querySelector('line[data-name="1_4_4"]');
    expect(el).toBeTruthy();
    expect(el!.getAttribute("data-band")).toBe("critical");
  });

  it("renders 14 substations", () => {
    const { container } = render(<GridMap state={t50} />);
    expect(container.querySelectorAll("[data-sub]")).toHaveLength(14);
  });

  it("t0 has no down lines", () => {
    const { container } = render(<GridMap state={t0} />);
    expect(container.querySelector('line[data-status="down"]')).toBeNull();
  });
});
