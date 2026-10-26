// Text-mode renderer for mockups (canvas 2D). The screen is one character grid.
// A map cell shows the strongest neuron that projects into it (max intensity projection),
// glyph = half-octave level, colour = sign. Layer outlines become edge characters. UI shares the grid.
import { level, bright, RAMP, rgb } from "./data.js";

export const STYLE = {
  bg: "#030304",
  pos: [255, 170, 40], posHi: [255, 234, 186],
  neg: [40, 205, 255], negHi: [212, 248, 255],
  latt: [120, 132, 145], edge: [120, 132, 145],
  ui: [165, 172, 178], uiHi: [238, 238, 234],
  font: "'JetBrains Mono'",
  alpha: (b) => 0.03 + 0.97 * Math.pow(b, 2.4),
  lattA: 0.14, edgeA: 0.30,
};

export function makeGrid(W, H, CW, CH) {
  const COLS = Math.floor(W / CW), ROWS = Math.floor(H / CH);
  return {
    W, H, CW, CH, COLS, ROWS,
    best: new Float32Array(COLS * ROWS), occ: new Uint8Array(COLS * ROWS),
    over: new Array(COLS * ROWS).fill(null), big: [],
  };
}

// pos: Float32Array [3 * M] (NaN = hidden). v: values (activation / p99). cam: projector. fog: [near, far] depths.
export function splat(G, pos, v, cam, fog = null) {
  const M = v.length;
  for (let k = 0; k < M; k++) {
    const x = pos[3 * k];
    if (Number.isNaN(x)) continue;
    const p = cam([x, pos[3 * k + 1], pos[3 * k + 2]]);
    if (p.z <= 0.1) continue;
    const cx = Math.floor(p.x / G.CW), cy = Math.floor(p.y / G.CH);
    if (cx < 0 || cy < 0 || cx >= G.COLS || cy >= G.ROWS) continue;
    const i = cy * G.COLS + cx;
    G.occ[i] = 1;
    const f = fog ? 1 - 0.45 * Math.min(1, Math.max(0, (p.z - fog[0]) / (fog[1] - fog[0]))) : 1;
    const w = v[k] * f;
    if (Math.abs(w) > Math.abs(G.best[i])) G.best[i] = w;
  }
}

// A 3D polyline drawn as / \ | - characters in empty or dim cells.
export function polyline(G, pts3, cam, alpha = STYLE.edgeA, color = STYLE.edge) {
  const pts = pts3.map(cam);
  for (let j = 0; j + 1 < pts.length; j++) {
    const a = pts[j], b = pts[j + 1];
    if (a.z <= 0.1 || b.z <= 0.1) continue;
    const ax = a.x / G.CW, ay = a.y / G.CH, bx = b.x / G.CW, by = b.y / G.CH;
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(bx - ax), Math.abs(by - ay)) * 1.5));
    const A = ((Math.atan2(b.y - a.y, b.x - a.x) * 180) / Math.PI + 360) % 180;
    const glyph = A < 22 || A > 158 ? "-" : A < 68 ? "\\" : A < 112 ? "|" : "/";
    for (let s = 0; s <= steps; s++) {
      const cx = Math.floor(ax + ((bx - ax) * s) / steps), cy = Math.floor(ay + ((by - ay) * s) / steps);
      if (cx < 0 || cy < 0 || cx >= G.COLS || cy >= G.ROWS) continue;
      const i = cy * G.COLS + cx;
      if (level(G.best[i]) < 2 && !G.over[i]) G.over[i] = { c: glyph, col: color, a: alpha };
    }
  }
}

export function put(G, col, row, text, color = STYLE.ui, a = 1, bold = false) {
  for (let j = 0; j < text.length; j++) {
    const cx = col + j;
    if (cx < 0 || cx >= G.COLS || row < 0 || row >= G.ROWS) continue;
    const i = row * G.COLS + cx;
    G.over[i] = text[j] === " " ? { c: " ", col: color, a: 0 } : { c: text[j], col: color, a, bold };
    G.best[i] = 0; G.occ[i] = 0;
  }
}
// Double-size text (like a VT100 double-height line): each character takes 2 x 2 cells.
export function putBig(G, col, row, text, color = STYLE.uiHi, a = 1, weight = 300) {
  for (let r = row; r < row + 2; r++) put(G, col, r, " ".repeat(text.length * 2), color, 0);
  G.big.push({ col, row, text, color, a, weight });
}

export function draw(g, G, dpr) {
  const S = STYLE, { CW, CH, COLS, ROWS } = G;
  g.fillStyle = S.bg; g.fillRect(0, 0, G.W, G.H);
  const glowC = document.createElement("canvas"); glowC.width = G.W * dpr; glowC.height = G.H * dpr;
  const gg = glowC.getContext("2d"); gg.scale(dpr, dpr);
  g.textBaseline = gg.textBaseline = "top"; g.textAlign = gg.textAlign = "center";
  const fs = Math.round(CH * 0.86);
  for (let i = 0; i < COLS * ROWS; i++) {
    const x = (i % COLS) * CW + CW / 2, y = Math.floor(i / COLS) * CH + 1;
    const o = G.over[i];
    if (o) {
      if (!o.a) continue;
      g.font = `${o.bold ? 700 : 400} ${fs}px ${S.font}`; g.fillStyle = rgb(o.col, o.a); g.fillText(o.c, x, y);
      if (o.glow) { gg.font = g.font; gg.fillStyle = rgb(o.glow, 1); gg.fillText(o.c, x, y); }
      continue;
    }
    if (!G.occ[i]) continue;
    const val = G.best[i], lv = level(val);
    if (lv === 0) { g.font = `400 ${fs}px ${S.font}`; g.fillStyle = rgb(S.latt, S.lattA); g.fillText("·", x, y); continue; }
    const b = bright(val), pos = val > 0;
    const col = lv >= 7 ? (pos ? S.posHi : S.negHi) : (pos ? S.pos : S.neg);
    g.font = `${lv >= 7 ? 700 : 400} ${fs}px ${S.font}`; g.fillStyle = rgb(col, S.alpha(b)); g.fillText(RAMP[lv], x, y);
    if (lv >= 5) { gg.font = g.font; gg.fillStyle = rgb(pos ? S.pos : S.neg, Math.min(1, 0.3 + b)); gg.fillText(RAMP[lv], x, y); }
  }
  g.textAlign = "left";
  for (const t of G.big) {
    g.font = `${t.weight} ${CH * 1.75}px ${S.font}`; g.fillStyle = rgb(t.color, t.a);
    for (let j = 0; j < t.text.length; j++) g.fillText(t.text[j], (t.col + 2 * j) * CW + CW * 0.1, t.row * CH + 2);
  }
  const blur = document.createElement("canvas"); blur.width = glowC.width; blur.height = glowC.height;
  const bb = blur.getContext("2d");
  bb.filter = `blur(${10 * dpr}px)`; bb.drawImage(glowC, 0, 0);
  bb.filter = `blur(${3 * dpr}px)`; bb.drawImage(glowC, 0, 0);
  g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = "lighter"; g.globalAlpha = 1; g.drawImage(blur, 0, 0); g.restore();
}
