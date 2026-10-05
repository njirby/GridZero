import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { Timeline } from "../src/components/Timeline";
import type { SeriesPoint, TickMark } from "../src/types";

describe("Timeline", () => {
  it("renders the 70/90% threshold lines", () => {
    const { getByTestId } = render(<Timeline series={[]} ticks={[]} />);
    expect(getByTestId("threshold-70")).toBeTruthy();
    expect(getByTestId("threshold-90")).toBeTruthy();
  });

  it("grows the cum-reward polyline with the series", () => {
    const series: SeriesPoint[] = [
      { t: 0, cum: 0, maxRho: 0.9, nDown: 0 },
      { t: 5, cum: 320, maxRho: 0.95, nDown: 0 },
      { t: 10, cum: 640, maxRho: 1.0, nDown: 1 },
    ];
    const { getByTestId, container } = render(<Timeline series={series} ticks={[]} />);
    expect(getByTestId("cum-line").getAttribute("points")!.split(" ")).toHaveLength(3);
    expect(getByTestId("rho-line").getAttribute("points")!.split(" ")).toHaveLength(3);
  });

  it("renders one tick mark per step and flags overloaded ones", () => {
    const ticks: TickMark[] = [
      { t: 0, source: "auto", bad: false, overloaded: false },
      { t: 5, source: "agent", bad: false, overloaded: true },
      { t: 10, source: "agent", bad: true, overloaded: false },
    ];
    const { container } = render(<Timeline series={[]} ticks={ticks} />);
    expect(container.querySelectorAll('[data-testid="tick"]')).toHaveLength(3);
    expect(container.querySelectorAll(".tick-over")).toHaveLength(1);
    expect(container.querySelectorAll(".tick-bad")).toHaveLength(1);
  });
});
