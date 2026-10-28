// Live fps / p95 readout for the tuning console, by the bench's method (bench/bench.ts finishRun): rAF intervals in ms,
// percentile = sorted[floor(p * n)], and each interval snapped to whole vsync periods (period = the median interval)
// before the p95, because rAF timestamps jitter around vsync (a perfect 60 Hz stream shows raw intervals of 15.7-17.7 ms).

/** The bench's percentile: the sorted value at floor(p * n) (no interpolation). */
export function pct(a: readonly number[], p: number): number {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

export interface FrameStats {
  frames: number;
  /** 1000 / mean interval. */
  fps: number;
  meanMs: number;
  /** The vsync period (median interval). */
  vsyncMs: number;
  /** p95 with intervals snapped to whole vsync periods: the budget figure (E6, ≤ 16.7 ms). */
  p95Vsync: number;
  /** Raw p95 of the intervals. */
  p95: number;
  /** Share of frames that took two or more vsync periods. */
  droppedPct: number;
  /** Share of frames over 33 ms (E6, ≤ 1%). */
  over33Pct: number;
}

export function frameStats(iv: readonly number[]): FrameStats {
  const n = iv.length, vs = pct(iv, 0.5), mean = iv.reduce((x, y) => x + y, 0) / Math.max(1, n);
  return {
    frames: n, fps: n ? 1000 / mean : NaN, meanMs: mean, vsyncMs: vs,
    p95Vsync: pct(iv.map((x) => vs * Math.max(1, Math.round(x / vs))), 0.95),
    p95: pct(iv, 0.95),
    droppedPct: (100 * iv.filter((x) => Math.round(x / vs) >= 2).length) / Math.max(1, n),
    over33Pct: (100 * iv.filter((x) => x > 33.3).length) / Math.max(1, n),
  };
}

/** A rolling window of frame intervals (ms). A gap over `resetMs` (a paused or hidden page, not a slow frame) restarts it. */
export class FrameMeter {
  private iv: number[] = [];
  constructor(readonly size = 600, readonly resetMs = 5000) {}
  push(ms: number) {
    if (!(ms > 0)) return;
    if (ms > this.resetMs) { this.iv.length = 0; return; }
    this.iv.push(ms);
    if (this.iv.length > this.size) this.iv.splice(0, this.iv.length - this.size);
  }
  reset() { this.iv.length = 0; }
  get count() { return this.iv.length; }
  stats(): FrameStats { return frameStats(this.iv); }
}
