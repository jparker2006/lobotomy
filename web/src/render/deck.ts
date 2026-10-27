// The deck renderer: every neuron, every frame, on the GPU (WebGPU) or with the compute stages on the CPU (WebGL2 fallback).
// Stages: trail + sweep (compute) -> splat into the 5 x 8 CSS-px dot grid (atomic max + count) -> resolve (rarity level per cell)
// -> glyph pass (one instanced quad per cell, SDF dots) -> bloom + astrophage haze from top-1% cells at bloomScale resolution.
// See README.md for the API and design.md for what each channel means.
import * as THREE from "three/webgpu";
import * as TSL from "three/tsl";
import { NEURONS, PER_LAYER, LAYERS, THRESHOLDS, INDEX_MASK, INDEX_BITS, MAG_MAX, KEY_LOG_LO, KEY_SCALE, RarityTable } from "./rarity.js";
import { GAP, PITCH, SHEET_COLS, SHEET_ROWS, DEFAULT_VIEW, View, outlineSegments, labelPos, boxCorners } from "./geometry.js";
import { CpuStages, SLOTS, StageInputs } from "./cpu.js";
import { recordedRarityTable } from "./synth.js";

// TSL's typings fight generic kernel code; the kernels below are checked by running them, not by tsc.
const T = TSL as any;
const { Fn, If, Loop, instancedArray, storage, uniform, instanceIndex, instancedBufferAttribute, float, int, uint, vec2, vec3, vec4,
  abs, log, exp, exp2, floor, clamp, pow, max, mix, length, fwidth, select, varying, positionGeometry, texture, uv, smoothstep, round,
  atomicMax, atomicAdd, atomicLoad, atomicStore } = T;

export type RGB = [number, number, number];
export interface DeckParams {
  /** Stream pace; sets the sweep (0.9 / tok_s) and converts the half-life to seconds. */
  tokPerSec: number;
  halfLifeTokens: number;
  /** Seconds a token takes to cross L0 -> L27; null = 0.9 / tokPerSec. 0 = no sweep. */
  sweepSec: number | null;
  /** Bloom target size relative to the canvas (0.5 = half resolution, the default). */
  bloomScale: number;
  bloomStrength: number;
  hazeStrength: number;
  cellW: number;
  cellH: number;
  /** Device pixels per CSS pixel; null = window.devicePixelRatio. */
  dpr: number | null;
  ink: RGB; hot: RGB; lattice: RGB; background: RGB;
  thresholds: number[];
  /** View-only depth fade of the dim lattice: [start, end] as fractions of the deck's depth range, and how much it dims. */
  lattFade: [number, number];
  lattFadeAmount: number;
  outlineAlpha: number;
}

export const DEFAULT_PARAMS: DeckParams = {
  tokPerSec: 6, halfLifeTokens: 2, sweepSec: null,
  bloomScale: 0.5, bloomStrength: 1, hazeStrength: 1.6,
  cellW: 5, cellH: 8, dpr: null,
  ink: [1, 0xb0 / 255, 0], hot: [1, 0xe4 / 255, 0xa0 / 255], lattice: [0x78 / 255, 0x84 / 255, 0x91 / 255], background: [3 / 255, 3 / 255, 4 / 255],
  thresholds: [...THRESHOLDS], lattFade: [0, 0.9], lattFadeAmount: 0.8, outlineAlpha: 0.18,
};

/** Dot shapes in CSS px per level and sign: [r1, w1, r2, w2] (w = 0 draws a filled disk, r = 0 nothing). Mirrors the glyphs · ∙ • ● ◉ / · ∘ ○ ◯ ◎. */
const SHAPES: number[][] = [
  [0.55, 0, 0, 0], [0.55, 0, 0, 0], // 0 dim lattice ·
  [0.7, 0, 0, 0], [0.7, 0, 0, 0], //   1 · (sign not visible)
  [1.0, 0, 0, 0], [1.3, 0.55, 0, 0], // 2 ∙ ∘
  [1.8, 0, 0, 0], [2.7, 0.65, 0, 0], // 3 • ○ (blooms)
  [4.0, 0, 0, 0], [4.9, 0.8, 0, 0], //  4 ● ◯ (hot)
  [5.3, 0.8, 2.5, 0], [5.3, 0.8, 2.3, 0.7], // 5 ◉ ◎ (hot)
];
const ALPHA = [0.2, 0.34, 0.55, 0.85, 0.95, 1.0];
const BLOOM_FROM = 3; // top 1% and rarer
// Glow widths in CSS px, as the canvas mockups (blur 3 + blur 10 for the bloom, 40 for the haze). The mid blur runs at 1/4
// of the bloom target and the haze at 1/8, so only the tight blur costs full bloom-target fill.
const TIGHT_SIGMA_CSS = 3, MID_SIGMA_CSS = 10, HAZE_SIGMA_CSS = 40;
const BLUR_TAPS = 12;

export interface Pick { neuron: number; layer: number; index: number; level: number; sign: number }

export interface Deck {
  readonly backend: "webgpu" | "webgl2";
  readonly params: DeckParams;
  readonly renderer: THREE.WebGPURenderer;
  readonly camera: THREE.PerspectiveCamera;
  /** r: 86,016 signed activation / p99, layer-major [28][3072]. tSec: when the token starts sweeping (same clock as frame()). */
  pushToken(r: Float32Array, tSec: number): void;
  setRarityTable(F: RarityTable): void;
  setView(v: Partial<View>): void;
  /** Runs every stage and draws. Call once per animation frame with a monotonic clock in seconds. */
  frame(tSec: number): void;
  /** CSS-px position of a world point (for labels), or null behind the camera. */
  project(p: [number, number, number]): { x: number; y: number } | null;
  layerLabel(l: number): { x: number; y: number } | null;
  /** The neuron that set the dot under a CSS-px point (as of the last frame). */
  pick(x: number, y: number): Promise<Pick | null>;
  /** Mean GPU ms per frame for compute and render since the last call (WebGPU with timestamp-query and opts.gpuTiming only). */
  gpuTimes(): Promise<{ compute: number; render: number; frames: number } | null>;
  /** Draws one frame into an offscreen target and returns its RGBA pixels, top row first (stills, clips, headless checks). */
  snapshot(tSec: number): Promise<{ width: number; height: number; data: Uint8Array }>;
  debugRead(): Promise<{ cells: Float32Array; glow: Float32Array }>;
  readonly size: { cssW: number; cssH: number; dpr: number; cols: number; rows: number; pxW: number; pxH: number };
  dispose(): void;
}

export async function createDeck(canvas: HTMLCanvasElement, params: Partial<DeckParams> = {}, opts: { forceWebGL?: boolean; gpuTiming?: boolean } = {}): Promise<Deck> {
  const P: DeckParams = { ...DEFAULT_PARAMS, ...params };
  const forceWebGL = opts.forceWebGL || !(navigator as any).gpu;
  const renderer = new THREE.WebGPURenderer({ canvas, antialias: false, alpha: false, forceWebGL, trackTimestamp: !!opts.gpuTiming && !forceWebGL } as any);
  await renderer.init();
  const gpu = !!(renderer.backend as any).isWebGPUBackend;
  renderer.autoClear = false;
  renderer.outputColorSpace = THREE.LinearSRGBColorSpace; // colours below are display values, blended like canvas 2D
  renderer.toneMapping = THREE.NoToneMapping;

  const camera = new THREE.PerspectiveCamera(DEFAULT_VIEW.fov, 1.6, 1, 200);
  const view: View = { ...DEFAULT_VIEW };
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  // ---- uniforms ----
  const u = {
    dtPrev: uniform(0), hl: uniform(1), sweep: uniform(0), slots: uniform(new THREE.Vector4()), decay: uniform(1),
    vp: uniform(new THREE.Matrix4()), css: uniform(new THREE.Vector2(1, 1)), cell: uniform(new THREE.Vector2(5, 8)), cols: uniform(1),
    logLo: uniform(0), logHi: uniform(1), bins: uniform(1), th: uniform(new THREE.Vector4()), th5: uniform(1),
    near: uniform(0), far: uniform(1), fade: uniform(new THREE.Vector3()),
    ink: uniform(new THREE.Vector3(...P.ink)), hot: uniform(new THREE.Vector3(...P.hot)), latt: uniform(new THREE.Vector3(...P.lattice)),
    bloom: uniform(P.bloomStrength), haze: uniform(P.hazeStrength), outlineA: uniform(P.outlineAlpha),
  };
  // small constant tables, indexed with selects (plain and the same on both backends)
  const table4 = (rows: number[][], i: any) => rows.slice(1).reduce((acc: any, r, j) => select(i.equal(j + 1), vec4(...r), acc), vec4(...rows[0]));
  const STYLE = ALPHA.map((a, i) => [a, i >= 4 ? 1 : 0, Math.max(SHAPES[2 * i][0], SHAPES[2 * i + 1][0]) + 1.5, 0]);

  // ---- per-neuron state ----
  const cpu = gpu ? null : new CpuStages();
  const tokAttr = new THREE.StorageBufferAttribute(new Float32Array(SLOTS * NEURONS), 1);
  const glowBuf = instancedArray(NEURONS, "float");
  const tokBuf = storage(tokAttr, "float", SLOTS * NEURONS).toReadOnly();
  const slotT0 = new Float64Array(SLOTS).fill(-1e9);
  let nextSlot = 0, lastT = -1e9, started = false;

  let table: RarityTable = recordedRarityTable();
  let rarityAttr = new THREE.StorageBufferAttribute(new Float32Array(table.cdf), 1);

  // ---- per-cell state (rebuilt when the grid changes) ----
  let cols = 0, rows = 0, cells = 0;
  let cellAttr: THREE.BufferAttribute | null = null;
  let kernels: any[] = [];
  let glyphMesh: THREE.Mesh | null = null, emisMesh: THREE.Mesh | null = null;
  const glyphScene = new THREE.Scene(), emisScene = new THREE.Scene(), lineScene = new THREE.Scene();
  const quadGeom = new THREE.PlaneGeometry(2, 2);

  // neuron i -> world position (same as geometry.neuronPos)
  const worldOf = (i: any) => {
    const l = i.div(PER_LAYER), n = i.mod(PER_LAYER);
    return vec4(float(l).sub((LAYERS - 1) / 2).mul(GAP), float((SHEET_ROWS - 1) / 2).sub(float(n.div(SHEET_COLS))).mul(PITCH),
      float(n.mod(SHEET_COLS)).sub((SHEET_COLS - 1) / 2).mul(PITCH), 1);
  };

  function buildKernels() {
    for (const k of kernels) k.dispose?.();
    const keyBuf = instancedArray(cells, "uint").toAtomic();
    const cntBuf = instancedArray(cells, "uint").toAtomic();
    const cellBuf = storage(cellAttr!, "vec4", cells);
    const cdf = storage(rarityAttr, "float", table.cdf.length).toReadOnly();

    const trail = Fn(() => {
      const i = instanceIndex;
      const off = float(i.div(PER_LAYER)).div(LAYERS).mul(u.sweep);
      const g = glowBuf.element(i).mul(u.decay).toVar();
      for (const c of ["x", "y", "z", "w"]) {
        const ta = u.slots[c].add(off); // arrival relative to now
        If(ta.greaterThan(u.dtPrev).and(ta.lessThanEqual(0)), () => {
          const v = tokBuf.element(uint(["x", "y", "z", "w"].indexOf(c) * NEURONS).add(i)).mul(exp2(ta.div(u.hl)));
          If(abs(v).greaterThanEqual(abs(g)), () => { g.assign(v); });
        });
      }
      glowBuf.element(i).assign(g);
    })().compute(NEURONS, [256]);

    const splat = Fn(() => {
      const i = instanceIndex;
      const c = u.vp.mul(worldOf(i));
      If(c.w.greaterThan(0.1), () => {
        const sx = c.x.div(c.w).mul(0.5).add(0.5).mul(u.css.x);
        const sy = float(0.5).sub(c.y.div(c.w).mul(0.5)).mul(u.css.y);
        const cx = floor(sx.div(u.cell.x)), cy = floor(sy.div(u.cell.y));
        If(cx.greaterThanEqual(0).and(cy.greaterThanEqual(0)).and(cx.lessThan(u.cols)).and(cy.mul(u.cols).add(cx).lessThan(cells)), () => {
          const cell = uint(cy.mul(u.cols).add(cx));
          const a = max(abs(glowBuf.element(i)), 1e-20);
          const q = uint(clamp(floor(log(a).sub(KEY_LOG_LO).mul(KEY_SCALE)), 0, MAG_MAX));
          atomicMax(keyBuf.element(cell), q.shiftLeft(uint(INDEX_BITS)).bitOr(i));
          atomicAdd(cntBuf.element(cell), uint(1));
        });
      });
    })().compute(NEURONS, [256]);

    const resolve = Fn(() => {
      const c = instanceIndex;
      const k = atomicLoad(cntBuf.element(c)).toVar(), key = atomicLoad(keyBuf.element(c)).toVar();
      atomicStore(cntBuf.element(c), uint(0));
      atomicStore(keyBuf.element(c), uint(0));
      const out = vec4(-1, 0, 0, 0).toVar();
      If(k.greaterThan(uint(0)), () => {
        const w = key.bitAnd(uint(INDEX_MASK));
        const g = glowBuf.element(w);
        // F(|g|): linear inside a log-spaced bin, as rarity.ts rarityF
        const x = log(max(abs(g), 1e-20)).sub(u.logLo).div(u.logHi.sub(u.logLo)).mul(u.bins).toVar();
        const b = int(clamp(floor(x), 0, u.bins.sub(1)));
        const hi = cdf.element(b), lo = select(b.greaterThan(0), cdf.element(max(b.sub(1), 0)), float(0));
        const F = select(x.lessThanEqual(0), float(0), select(x.greaterThanEqual(u.bins), float(1), mix(lo, hi, x.sub(floor(x)))));
        const uu = pow(F, float(k));
        const lv = float(uu.greaterThanEqual(u.th.x)).add(float(uu.greaterThanEqual(u.th.y))).add(float(uu.greaterThanEqual(u.th.z)))
          .add(float(uu.greaterThanEqual(u.th.w))).add(float(uu.greaterThanEqual(u.th5)));
        out.assign(vec4(lv, select(g.lessThan(0), float(-1), float(1)), u.vp.mul(worldOf(w)).w, float(w)));
      });
      cellBuf.element(c).assign(out);
    })().compute(cells, [64]);

    kernels = [trail, splat, resolve];
  }

  function glyphMaterial(emissive: boolean) {
    const d = instancedBufferAttribute(cellAttr, "vec4");
    const lv = d.x, li = int(max(round(lv), 0)), neg = int(d.y.lessThan(0));
    const shape = table4(SHAPES, li.mul(2).add(neg)), sty = table4(STYLE, li);
    const live = emissive ? lv.greaterThan(BLOOM_FROM - 0.5) : lv.greaterThan(-0.5);
    // view-only depth fade, lattice (level 0) only
    const t = d.z.sub(u.near).div(max(u.far.sub(u.near), 1e-3));
    const fade = select(lv.lessThan(0.5), float(1).sub(u.fade.z.mul(smoothstep(u.fade.x, u.fade.y, t))), float(1));
    const vShape = varying(shape), vSty = varying(vec4(sty.x.mul(fade), sty.y, lv, 0));
    const local = positionGeometry.xy.mul(sty.z);
    const vLocal = varying(local);
    const idx = float(instanceIndex), row = floor(idx.div(u.cols)), col = idx.sub(row.mul(u.cols));
    const px = vec2(col.add(0.5).mul(u.cell.x), row.add(0.5).mul(u.cell.y)).add(local);
    const m = new THREE.NodeMaterial();
    m.vertexNode = select(live, vec4(px.x.div(u.css.x).mul(2).sub(1), float(1).sub(px.y.div(u.css.y).mul(2)), 0, 1), vec4(2, 2, 2, 1));
    m.fragmentNode = Fn(() => {
      const r = length(vLocal), aa = max(fwidth(r), 0.05);
      const cov = (rad: any, w: any) => select(w.greaterThan(0), clamp(w.mul(0.5).sub(abs(r.sub(rad))).div(aa).add(0.5), 0, 1),
        clamp(rad.sub(r).div(aa).add(0.5), 0, 1)).mul(float(rad.greaterThan(0)));
      const a = max(cov(vShape.x, vShape.y), cov(vShape.z, vShape.w)).mul(vSty.x);
      if (emissive) return vec4(u.ink, a);
      const col = select(vSty.z.lessThan(0.5), u.latt, select(vSty.y.greaterThan(0.5), u.hot, u.ink));
      return vec4(col, a);
    })();
    m.transparent = true; m.depthTest = false; m.depthWrite = false; m.side = THREE.DoubleSide; // y is flipped, so winding flips
    m.blending = emissive ? THREE.AdditiveBlending : THREE.NormalBlending;
    return m;
  }

  function buildCells() {
    cells = cols * rows;
    const arr = new Float32Array(cells * 4);
    for (let c = 0; c < cells; c++) arr[4 * c] = -1;
    cellAttr = gpu ? new THREE.StorageInstancedBufferAttribute(arr, 4) : new THREE.InstancedBufferAttribute(arr, 4);
    if (gpu) buildKernels();
    for (const [mesh, scene, emissive] of [[glyphMesh, glyphScene, false], [emisMesh, emisScene, true]] as const) {
      if (mesh) { scene.remove(mesh); (mesh.material as THREE.Material).dispose(); }
      const m = new THREE.Mesh(quadGeom, glyphMaterial(emissive));
      m.count = cells; m.frustumCulled = false;
      scene.add(m);
      if (emissive) emisMesh = m; else glyphMesh = m;
    }
  }

  // ---- sheet outlines ----
  const lineGeom = new THREE.BufferGeometry();
  lineGeom.setAttribute("position", new THREE.BufferAttribute(outlineSegments(), 3));
  const lineMat = new THREE.LineBasicNodeMaterial({ transparent: true, depthTest: false, depthWrite: false });
  lineMat.colorNode = u.latt; lineMat.opacityNode = u.outlineA;
  const lines = new THREE.LineSegments(lineGeom, lineMat);
  lines.frustumCulled = false;
  lineScene.add(lines);

  // ---- bloom + haze ----
  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: false, magFilter: THREE.LinearFilter, minFilter: THREE.LinearFilter, generateMipmaps: false };
  const E0 = new THREE.RenderTarget(1, 1, rtOpts), E1 = new THREE.RenderTarget(1, 1, rtOpts);
  const D = [1, 2, 3].map(() => new THREE.RenderTarget(1, 1, rtOpts)); // 1/2, 1/4, 1/8 of E0
  const M = new THREE.RenderTarget(1, 1, rtOpts), Mt = new THREE.RenderTarget(1, 1, rtOpts), Ht = new THREE.RenderTarget(1, 1, rtOpts);
  const quad = new THREE.QuadMesh(new THREE.NodeMaterial());
  function blurMat(src: THREE.Texture, dir: THREE.Vector2, sigma: any) {
    const tex = texture(src), uDir = uniform(dir);
    const m = new THREE.NodeMaterial();
    m.fragmentNode = Fn(() => {
      const acc = vec3(0).toVar(), ws = float(0).toVar();
      const step = max(float(1), sigma.mul(2.5 / BLUR_TAPS));
      Loop({ start: int(-BLUR_TAPS), end: int(BLUR_TAPS), type: "int", condition: "<=" }, ({ i }: any) => {
        const x = float(i).mul(step), w = exp(x.mul(x).div(sigma.mul(sigma).mul(-2)));
        acc.addAssign(tex.sample(uv().add(uDir.mul(x))).rgb.mul(w));
        ws.addAssign(w);
      });
      return vec4(acc.div(ws), 1);
    })();
    return { m, tex, uDir };
  }
  function downMat(src: THREE.Texture) {
    const tex = texture(src), texel = uniform(new THREE.Vector2());
    const m = new THREE.NodeMaterial();
    const o = texel.mul(0.5);
    m.fragmentNode = tex.sample(uv().add(vec2(o.x.negate(), o.y.negate()))).add(tex.sample(uv().add(vec2(o.x, o.y.negate()))))
      .add(tex.sample(uv().add(vec2(o.x.negate(), o.y)))).add(tex.sample(uv().add(o))).mul(0.25);
    return { m, texel };
  }
  const sigTight = uniform(1), sigMid = uniform(1), sigHaze = uniform(1);
  const tightH = blurMat(E0.texture, new THREE.Vector2(), sigTight), tightV = blurMat(E1.texture, new THREE.Vector2(), sigTight);
  const downs = [downMat(E0.texture), downMat(D[0].texture), downMat(D[1].texture)];
  const midH = blurMat(D[1].texture, new THREE.Vector2(), sigMid), midV = blurMat(Mt.texture, new THREE.Vector2(), sigMid);
  const hazeH = blurMat(D[2].texture, new THREE.Vector2(), sigHaze), hazeV = blurMat(Ht.texture, new THREE.Vector2(), sigHaze);
  const compMat = new THREE.NodeMaterial();
  compMat.fragmentNode = vec4(texture(E0.texture, uv()).rgb.add(texture(M.texture, uv()).rgb).mul(u.bloom).add(texture(D[2].texture, uv()).rgb.mul(u.haze)), 1);
  compMat.transparent = true; compMat.blending = THREE.AdditiveBlending; compMat.depthTest = false; compMat.depthWrite = false;
  const pass = (m: THREE.Material, rt: THREE.RenderTarget | null, clear = true) => {
    renderer.setRenderTarget(rt);
    if (clear) renderer.clear();
    quad.material = m as any;
    quad.render(renderer);
  };

  // ---- sizing ----
  const size = { cssW: 0, cssH: 0, dpr: 0, cols: 0, rows: 0, pxW: 0, pxH: 0 };
  let bloomKey = "";
  function fit() {
    const cssW = Math.max(1, canvas.clientWidth), cssH = Math.max(1, canvas.clientHeight), dpr = P.dpr ?? window.devicePixelRatio ?? 1;
    if (cssW !== size.cssW || cssH !== size.cssH || dpr !== size.dpr) {
      renderer.setPixelRatio(dpr);
      renderer.setSize(cssW, cssH, false);
      Object.assign(size, { cssW, cssH, dpr, pxW: Math.round(cssW * dpr), pxH: Math.round(cssH * dpr) });
      camera.aspect = cssW / cssH;
    }
    const nc = Math.ceil(cssW / P.cellW), nr = Math.ceil(cssH / P.cellH);
    if (nc !== cols || nr !== rows) { cols = nc; rows = nr; size.cols = cols; size.rows = rows; buildCells(); }
    const key = `${size.pxW}x${size.pxH}@${P.bloomScale}`;
    if (key !== bloomKey) {
      bloomKey = key;
      const w = Math.max(1, Math.round(size.pxW * P.bloomScale)), h = Math.max(1, Math.round(size.pxH * P.bloomScale));
      E0.setSize(w, h); E1.setSize(w, h);
      let dw = w, dh = h;
      D.forEach((rt, j) => { downs[j].texel.value.set(1 / dw, 1 / dh); dw = Math.max(1, dw >> 1); dh = Math.max(1, dh >> 1); rt.setSize(dw, dh); });
      Ht.setSize(dw, dh);
      const qw = D[1].width, qh = D[1].height;
      M.setSize(qw, qh); Mt.setSize(qw, qh);
      tightH.uDir.value.set(1 / w, 0); tightV.uDir.value.set(0, 1 / h);
      midH.uDir.value.set(1 / qw, 0); midV.uDir.value.set(0, 1 / qh);
      hazeH.uDir.value.set(1 / dw, 0); hazeV.uDir.value.set(0, 1 / dh);
    }
    // blur widths are in CSS px, so the look holds at any bloom scale and DPR
    const pxPerCss = size.dpr * P.bloomScale;
    sigTight.value = TIGHT_SIGMA_CSS * pxPerCss; sigMid.value = (MID_SIGMA_CSS * pxPerCss) / 4; sigHaze.value = (HAZE_SIGMA_CSS * pxPerCss) / 8;
  }

  function updateCamera() {
    camera.fov = view.fov;
    camera.position.set(...view.eye);
    camera.up.set(0, 1, 0);
    camera.lookAt(new THREE.Vector3(...view.target));
    // principal point at view.center: shift the frustum window
    camera.setViewOffset(size.cssW, size.cssH, (0.5 - view.center[0]) * size.cssW, (0.5 - view.center[1]) * size.cssH, size.cssW, size.cssH);
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();
  }
  const vpM = new THREE.Matrix4(), tmpV = new THREE.Vector4();

  function setRarityTable(F: RarityTable) {
    const resized = F.cdf.length !== table.cdf.length;
    table = { logLo: F.logLo, logHi: F.logHi, cdf: new Float32Array(F.cdf) };
    if (resized) { rarityAttr = new THREE.StorageBufferAttribute(new Float32Array(table.cdf), 1); if (gpu && cells) buildKernels(); }
    else { (rarityAttr.array as Float32Array).set(table.cdf); rarityAttr.needsUpdate = true; }
  }

  function pushToken(r: Float32Array, tSec: number) {
    if (r.length !== NEURONS) throw new Error(`pushToken: expected ${NEURONS} values, got ${r.length}`);
    const s = nextSlot++ % SLOTS;
    slotT0[s] = Math.max(tSec, lastT + 1e-6); // a token stamped in the past starts sweeping now
    if (cpu) cpu.tok.set(r, s * NEURONS);
    else {
      (tokAttr.array as Float32Array).set(r, s * NEURONS);
      tokAttr.clearUpdateRanges();
      tokAttr.addUpdateRange(s * NEURONS, NEURONS);
      tokAttr.needsUpdate = true;
    }
  }

  const stage: StageInputs = { t: 0, tPrev: 0, hlSec: 1, sweepSec: 0, t0: slotT0, vp: vpM.elements, cssW: 1, cssH: 1, cellW: 5, cellH: 8, cols: 1, rows: 1 };

  let screenRT: THREE.RenderTarget | null = null; // null = the canvas
  function frame(t: number) {
    fit();
    updateCamera();
    if (!started) { lastT = t; started = true; }
    const hlSec = Math.max(1e-4, P.halfLifeTokens / P.tokPerSec), sweepSec = P.sweepSec ?? 0.9 / P.tokPerSec;
    vpM.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // depth range of the deck for the lattice fade
    let near = Infinity, far = -Infinity;
    for (const p of boxCorners()) { tmpV.set(p[0], p[1], p[2], 1).applyMatrix4(vpM); near = Math.min(near, tmpV.w); far = Math.max(far, tmpV.w); }
    u.near.value = near; u.far.value = far; u.fade.value.set(P.lattFade[0], P.lattFade[1], P.lattFadeAmount);
    u.ink.value.set(...P.ink); u.hot.value.set(...P.hot); u.latt.value.set(...P.lattice);
    u.bloom.value = P.bloomStrength; u.haze.value = P.hazeStrength; u.outlineA.value = P.outlineAlpha;
    u.css.value.set(size.cssW, size.cssH); u.cell.value.set(P.cellW, P.cellH); u.cols.value = cols;
    const th = P.thresholds;
    u.th.value.set(th[0], th[1], th[2], th[3]); u.th5.value = th[4];
    u.logLo.value = table.logLo; u.logHi.value = table.logHi; u.bins.value = table.cdf.length;

    Object.assign(stage, { t, tPrev: lastT, hlSec, sweepSec, cssW: size.cssW, cssH: size.cssH, cellW: P.cellW, cellH: P.cellH, cols, rows });
    if (cpu) {
      cpu.trail(stage); cpu.splat(stage); cpu.resolve(stage, table, cellAttr!.array as Float32Array, th);
      cellAttr!.needsUpdate = true;
    } else {
      u.vp.value.copy(vpM);
      u.dtPrev.value = lastT - t; u.hl.value = hlSec; u.sweep.value = sweepSec; u.decay.value = Math.pow(0.5, (t - lastT) / hlSec);
      u.slots.value.set(slotT0[0] - t, slotT0[1] - t, slotT0[2] - t, slotT0[3] - t);
      renderer.compute(kernels);
    }
    lastT = t;

    // emissive (top-1% cells only) -> tight bloom + wide haze
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(E0); renderer.clear(); renderer.render(emisScene, ortho);
    pass(tightH.m, E1); pass(tightV.m, E0); // E0: tight bloom
    pass(downs[0].m, D[0]); pass(downs[1].m, D[1]);
    pass(midH.m, Mt); pass(midV.m, M); // M: mid bloom at 1/4
    pass(downs[2].m, D[2]);
    pass(hazeH.m, Ht); pass(hazeV.m, D[2]); // D[2]: haze at 1/8
    // screen
    renderer.setRenderTarget(screenRT);
    renderer.setClearColor(new THREE.Color().setRGB(...P.background, THREE.LinearSRGBColorSpace), 1);
    renderer.clear();
    renderer.render(lineScene, camera);
    renderer.render(glyphScene, ortho);
    pass(compMat, screenRT, false);
  }

  function project(p: [number, number, number]) {
    tmpV.set(p[0], p[1], p[2], 1).applyMatrix4(vpM);
    if (tmpV.w <= 0.1) return null;
    return { x: (tmpV.x / tmpV.w * 0.5 + 0.5) * size.cssW, y: (0.5 - tmpV.y / tmpV.w * 0.5) * size.cssH };
  }

  async function pick(x: number, y: number): Promise<Pick | null> {
    const cx = Math.floor(x / P.cellW), cy = Math.floor(y / P.cellH);
    if (cx < 0 || cy < 0 || cx >= cols || cy >= rows || !cellAttr) return null;
    const c = cy * cols + cx;
    const data = cpu ? (cellAttr.array as Float32Array) : new Float32Array(await renderer.getArrayBufferAsync(cellAttr as any));
    const lv = data[4 * c];
    if (lv < 0) return null;
    const w = data[4 * c + 3];
    return { neuron: w, layer: Math.floor(w / PER_LAYER), index: w % PER_LAYER, level: lv, sign: data[4 * c + 1] };
  }

  async function snapshot(t: number) {
    fit();
    const rt = new THREE.RenderTarget(size.pxW, size.pxH, { depthBuffer: false, generateMipmaps: false });
    screenRT = rt;
    try { frame(t); } finally { screenRT = null; }
    const raw = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, size.pxW, size.pxH) as Uint8Array;
    rt.dispose();
    const W = size.pxW, H = size.pxH, data = new Uint8Array(W * H * 4);
    // WebGPU targets come back top row first; WebGL ones bottom row first
    const stride = H > 1 ? (raw.length - W * 4) / (H - 1) : W * 4; // WebGPU pads rows to 256 bytes (not the last one)
    for (let y = 0; y < H; y++) { const sy = gpu ? y : H - 1 - y; data.set(raw.subarray(sy * stride, sy * stride + W * 4), y * W * 4); }
    return { width: W, height: H, data };
  }

  // GPU timestamps: resolved in the background every frame (so the query pool never fills), averaged on request
  const timing = { on: gpu && !!(renderer.backend as any).trackTimestamp, pending: false, frames: 0, compute: 0, render: 0, resolved: 0 };
  function pollTimestamps() {
    timing.frames++;
    if (!timing.on || timing.pending) return;
    timing.pending = true;
    const n = timing.frames; timing.frames = 0;
    Promise.all([renderer.resolveTimestampsAsync("compute" as any), renderer.resolveTimestampsAsync("render" as any)])
      .then(([c, r]) => { timing.compute += c ?? 0; timing.render += r ?? 0; timing.resolved += n; })
      .catch(() => { timing.on = false; })
      .finally(() => { timing.pending = false; });
  }
  async function gpuTimes() {
    if (!timing.on || !timing.resolved) return null;
    const out = { compute: timing.compute / timing.resolved, render: timing.render / timing.resolved, frames: timing.resolved };
    timing.compute = timing.render = timing.resolved = 0;
    return out;
  }


  fit();
  updateCamera();
  vpM.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  return {
    backend: gpu ? "webgpu" : "webgl2", params: P, renderer, camera, size,
    pushToken, setRarityTable,
    setView(v) { Object.assign(view, v); },
    frame(t) { frame(t); pollTimestamps(); },
    project, layerLabel: (l) => project(labelPos(l)), pick, gpuTimes, snapshot,
    dispose() { renderer.dispose(); },
    /** Internal, for checks: raw cell (level, sign, depth, winner) and glow arrays as of the last frame. */
    async debugRead() {
      if (cpu) return { cells: cellAttr!.array as Float32Array, glow: cpu.glow };
      const [c, g] = await Promise.all([renderer.getArrayBufferAsync(cellAttr as any), renderer.getArrayBufferAsync(glowBuf.value)]);
      return { cells: new Float32Array(c), glow: new Float32Array(g) };
    },
  };
}
