// Shared deterministic PRNGs for burn-math (used by fuzz_cases.mjs,
// prng_check.mjs, cost_model.mjs). Small, auditable algorithms; f64 outputs
// in [0,1). Streams are certified by sha256 over the raw f64 bytes
// (little-endian), so any agent can re-verify a recorded stream exactly.
import { createHash } from 'node:crypto';

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function xorshift32(seed) {
  let x = (seed >>> 0) || 1; // zero seed would lock up; map to 1
  return function () {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x / 4294967296;
  };
}

export const PRNGS = { mulberry32, xorshift32 };

// Generate `count` values in [lo, hi); returns {values, sha256}.
export function generateStream(algo, seed, count, lo = 0, hi = 1) {
  const factory = PRNGS[algo];
  if (!factory) throw new Error(`unknown prng '${algo}' (available: ${Object.keys(PRNGS).join(', ')})`);
  const next = factory(seed);
  const values = new Array(count);
  const buf = Buffer.allocUnsafe(count * 8);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  for (let i = 0; i < count; i++) {
    const v = lo + next() * (hi - lo);
    values[i] = v;
    dv.setFloat64(i * 8, v, true);
  }
  const sha256 = 'sha256:' + createHash('sha256').update(buf).digest('hex');
  return { values, sha256 };
}
