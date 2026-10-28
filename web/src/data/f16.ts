// float16 bits -> float32, by table.
export const F16 = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  F16[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
}

/** r = activation / p99, signed: out[k] = f16(acts[k]) * invP99[k]. */
export function toR(acts: Uint16Array, invP99: Float32Array, out = new Float32Array(acts.length)): Float32Array {
  for (let k = 0; k < acts.length; k++) out[k] = F16[acts[k]] * invP99[k];
  return out;
}
