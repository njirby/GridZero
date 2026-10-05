import type { SeriesPoint, TickMark } from "../types";

const W = 1000, H = 220, PAD = 20, RAIL = 18;

export function Timeline({ series, ticks }: { series: SeriesPoint[]; ticks: TickMark[] }) {
  const n = Math.max(series.length, 1);
  const chartH = H - PAD - RAIL;
  const maxCum = Math.max(1, ...series.map((s) => s.cum));
  const maxRhoTop = 1.3;
  const x = (i: number) => PAD + (i * (W - 2 * PAD)) / (n - 1 || 1);
  const yCum = (v: number) => PAD + chartH - (v / maxCum) * chartH;
  const yRho = (v: number) => PAD + chartH - (Math.min(v, maxRhoTop) / maxRhoTop) * chartH;

  const cumPts = series.map((s, i) => `${x(i)},${yCum(s.cum)}`).join(" ");
  const rhoPts = series.map((s, i) => `${x(i)},${yRho(s.maxRho)}`).join(" ");
  const th70 = yRho(0.7), th90 = yRho(0.9);

  const m = Math.max(ticks.length, 1);
  const tx = (i: number) => PAD + (i * (W - 2 * PAD)) / (m - 1 || 1);

  return (
    <div className="timeline" data-testid="timeline">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%">
        {/* thresholds */}
        <line x1={PAD} y1={th90} x2={W - PAD} y2={th90} className="threshold t90" data-testid="threshold-90" />
        <line x1={PAD} y1={th70} x2={W - PAD} y2={th70} className="threshold t70" data-testid="threshold-70" />
        {/* cum reward */}
        {series.length > 1 && (
          <polyline data-testid="cum-line" className="series-cum" points={cumPts} fill="none" strokeWidth={2} />
        )}
        {/* max rho */}
        {series.length > 1 && (
          <polyline data-testid="rho-line" className="series-rho" points={rhoPts} fill="none" strokeWidth={2} />
        )}
        {/* tick rail */}
        {ticks.map((t, i) => (
          <rect key={i} x={tx(i) - 1.5} y={H - RAIL + 2} width={3} height={RAIL - 4}
            className={`tick tick-${t.source} ${t.bad ? "tick-bad" : t.overloaded ? "tick-over" : ""}`}
            data-testid="tick" />
        ))}
      </svg>
      <div className="timeline-legend">
        <span className="lg-cum">cum reward</span>
        <span className="lg-rho">max loading</span>
        <span>· thresholds 70/90%</span>
      </div>
    </div>
  );
}
