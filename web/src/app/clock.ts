// The app's time source, in seconds. Real time by default; a manual (deterministic) mode for frame-by-frame renders (U6):
// clock.setManual(t) freezes it at t, then the caller advances it with setManual again and calls app.renderAt(t).
export interface Clock {
  /** Seconds now (manual mode: the set time). Token arrivals and the deck use this. */
  now(): number;
  /** The current animation frame's time, as passed to deck.frame(). */
  readonly frameTime: number;
  /** Called once per animation frame with the rAF timestamp (ms); returns the frame time in seconds. */
  tick(rafMs: number): number;
  /** Manual mode: freeze the clock at t seconds (null = back to real time, continuing from t without a jump). */
  setManual(t: number | null): void;
  readonly manual: boolean;
}

export function createClock(): Clock {
  let manualT: number | null = null, offset = 0, frameTime = 0;
  const real = () => performance.now() / 1000 + offset;
  return {
    now: () => (manualT ?? real()),
    get frameTime() { return frameTime; },
    tick(rafMs) {
      // rAF timestamps and performance.now() share an origin; use the rAF one so the frame time is the vsync time
      frameTime = manualT ?? Math.max(frameTime, rafMs / 1000 + offset);
      return frameTime;
    },
    setManual(t) {
      if (t === null && manualT !== null) offset = manualT - performance.now() / 1000;
      manualT = t;
      if (t !== null) frameTime = t;
    },
    get manual() { return manualT !== null; },
  };
}
