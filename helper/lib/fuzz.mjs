// burn-math fuzz: deterministic test-case generation + Recursive Adversarial Interval Narrowing (`findWorstCaseInput`).
// Shared by fuzz_cases.mjs (CLI) and verify.mjs --fuzz. ONE producer: both CLIs
// call generateCases, so the two can never drift apart.
//
// Case layout (identical to the original fuzz_cases.mjs for continuous
// ports, so pre-refactor outputs stay byte-identical):
//   case 0 : the spec's own inputs (the golden case), when present
//   then   : boundary cases — zeros, ones, alternating +/-1, large (+50),
//            small (+1e-4), negative large (-50)
//   then   : --count seeded-random cases, uniform [lo,hi) via mulberry32
//
// Per-port domains: a port with layout 'token_ids_axis1_singleton' is
// DISCRETE (embedding input: integer-valued token ids carried through
// the f32 bridge). Its boundary/random values are clamped into its
// domain — it never receives -1, a fraction, or an out-of-range id:
//   boundaries: 0, 1, alternating 0/1, vocab-1, 0, vocab-1
//   random: uniform integers in [0, vocab), from a separate mulberry32
//   stream with the same seed (documented; the continuous stream keeps
//   its original consumption order, so old outputs are unchanged).
// Vocab comes from the embedding layer that consumes the slot
// (args[1]); a discrete port with no embedding consumer, or with two
// consumers disagreeing on vocab, is a fail-closed error, never a guess.
import { portBySlot } from './spec.mjs';
import { generateStream, mulberry32 } from './prng.mjs';
import { buildGraph, makeBundle } from './engine.mjs';

const DISCRETE_LAYOUT = 'token_ids_axis1_singleton';

function isU32(v) {
  return Number.isInteger(v) && v >= 0 && v <= 4294967295;
}

// slot -> vocab for every discrete port in `ports`, or throw.
function discreteVocabs(spec, ports) {
  const bySlot = new Map();
  for (const p of ports) {
    if (p.layout === DISCRETE_LAYOUT) bySlot.set(p.slot, new Set());
  }
  if (bySlot.size === 0) return new Map();
  for (const step of (spec.steps ?? [])) {
    if (!step || !Array.isArray(step.slots)) continue;
    const layer = spec.layers?.[step.layer];
    if (!layer || layer.constructor !== 'embedding') continue;
    const inputs = step.kind === 'binary' ? step.slots.slice(0, 2) : step.slots.slice(0, 1);
    for (const s of inputs) {
      if (!bySlot.has(s)) continue;
      const vocab = layer.args?.[1];
      if (!isU32(vocab) || vocab === 0) {
        throw new Error(`embedding layer ${step.layer} has no valid vocab size for discrete port slot ${s}`);
      }
      bySlot.get(s).add(vocab);
    }
  }
  const out = new Map();
  for (const [slot, vocabs] of bySlot) {
    if (vocabs.size === 0) {
      throw new Error(`discrete port slot ${slot} (layout ${DISCRETE_LAYOUT}) has no consuming embedding layer; cannot determine its vocab size`);
    }
    if (vocabs.size > 1) {
      throw new Error(`discrete port slot ${slot} is consumed by embedding layers with different vocab sizes (${[...vocabs].join(', ')}); ambiguous`);
    }
    out.set(slot, [...vocabs][0]);
  }
  return out;
}

export function generateCases(spec, opts = {}) {
  const seed = opts.seed ?? 42;
  const count = opts.count ?? 8;
  const lo = opts.lo ?? -2;
  const hi = opts.hi ?? 2;
  if (!Number.isInteger(seed) || !Number.isInteger(count) || count < 0)
    throw new Error('--seed and --count must be integers, --count >= 0');
  const ports = [...portBySlot(spec).values()]
    .filter((p) => !opts.onlySlots || opts.onlySlots.includes(p.slot))
    .sort((a, b) => a.slot - b.slot);
  if (ports.length === 0) throw new Error('no ports selected');
  const vocabs = discreteVocabs(spec, ports);
  const lenOf = (p) => p.shape.reduce((x, y) => x * y, 1);

  const cases = [];
  // 0. golden case: the spec's own inputs, passed through unchanged.
  if (spec.inputs && Object.keys(spec.inputs).length > 0) {
    const g = {};
    for (const p of ports) {
      const v = spec.inputs[String(p.slot)];
      if (v === undefined) throw new Error(`spec.inputs missing port slot ${p.slot} (use --only-slots or add it)`);
      g[String(p.slot)] = [...v];
    }
    cases.push(g);
  }
  // 1. boundary cases, mapped per port domain.
  const contBoundary = [
    () => 0,
    () => 1,
    (i) => (i % 2 === 0 ? 1 : -1),
    () => 50,
    () => 1e-4,
    () => -50,
  ];
  const discBoundary = (vocab) => [
    () => 0,
    () => 1,
    (i) => (i % 2 === 0 ? 0 : 1),
    () => vocab - 1,
    () => 0,
    () => vocab - 1,
  ];
  for (let b = 0; b < contBoundary.length; b++) {
    const c = {};
    for (const p of ports) {
      const n = lenOf(p);
      const fn = vocabs.has(p.slot) ? discBoundary(vocabs.get(p.slot))[b] : contBoundary[b];
      c[String(p.slot)] = Array.from({ length: n }, (_, i) => fn(i, n));
    }
    cases.push(c);
  }
  // 2. seeded random cases: continuous ports share the original stream
  // (same slot order, same values as before the refactor); discrete
  // ports draw integers from a separate stream with the same seed.
  const contPorts = ports.filter((p) => !vocabs.has(p.slot));
  const discPorts = ports.filter((p) => vocabs.has(p.slot));
  const contTotal = contPorts.reduce((a, p) => a + lenOf(p), 0);
  const discTotal = discPorts.reduce((a, p) => a + lenOf(p), 0);
  const contStream = contTotal > 0
    ? generateStream('mulberry32', seed, count * contTotal, lo, hi).values
    : [];
  const discStream = discTotal > 0
    ? generateStream('mulberry32', seed, count * discTotal, 0, 1).values
    : [];
  for (let k = 0; k < count; k++) {
    const c = {};
    let coff = k * contTotal;
    for (const p of contPorts) {
      const n = lenOf(p);
      c[String(p.slot)] = contStream.slice(coff, coff + n);
      coff += n;
    }
    let doff = k * discTotal;
    for (const p of discPorts) {
      const n = lenOf(p);
      const vocab = vocabs.get(p.slot);
      c[String(p.slot)] = discStream.slice(doff, doff + n).map((v) => Math.floor(v * vocab));
      doff += n;
    }
    // Restore slot order in the case object (insertion order above is
    // continuous-first); key order is cosmetic, but keep it canonical.
    const ordered = {};
    for (const p of ports) ordered[String(p.slot)] = c[String(p.slot)];
    cases.push(ordered);
  }

  return {
    cases,
    meta: {
      seed,
      prng: 'mulberry32',
      random_range: [lo, hi],
      ports: ports.map((p) => p.slot),
      includes_golden: !!(spec.inputs && Object.keys(spec.inputs).length > 0),
      discrete_ports: [...vocabs.entries()].map(([slot, vocab]) => ({ slot, vocab })),
    },
  };
}

// Recursive Adversarial Interval Narrowing (Zoom Fuzzing):
// Recursively narrows a box around the input case that maximizes |Base(x) - Cand(x)|_oo
// across `zoomDepth` levels (halving the search radius at each depth).
export function findWorstCaseInput(rt, baseSpec, candSpec, {
  seed = 42,
  samplesPerLevel = 12,
  zoomDepth = 4,
  lo = -5,
  hi = 5,
} = {}) {
  const baseBuilt = buildGraph(rt, baseSpec);
  const candBuilt = buildGraph(rt, candSpec);
  const basePorts = portBySlot(baseSpec);
  const candPorts = portBySlot(candSpec);
  const ports = [...basePorts.values()].sort((a, b) => a.slot - b.slot);
  const vocabs = discreteVocabs(baseSpec, ports);
  const rng = mulberry32(seed);

  let outLenA = null, outLenB = null;
  const evalDelta = (caseInputs) => {
    const bA = makeBundle(rt, baseBuilt.plan, caseInputs, basePorts);
    const bB = makeBundle(rt, candBuilt.plan, caseInputs, candPorts);
    const outA = baseBuilt.g.run(baseBuilt.reg, bA);
    const outB = candBuilt.g.run(candBuilt.reg, bB);
    try {
      const vA = Array.from(outA.to_array());
      const vB = Array.from(outB.to_array());
      outLenA = vA.length; outLenB = vB.length;
      // Length mismatch IS a difference: the old prefix-only loop reported
      // 0 for pairs whose shared prefix matched (e.g. [3,7] vs [3,7,3,7]).
      // Infinity keeps mismatch cases as worst-case; the report below
      // states the mismatch explicitly instead of a bare 0.
      if (vA.length !== vB.length) return Number.POSITIVE_INFINITY;
      let m = 0;
      for (let i = 0; i < Math.min(vA.length, vB.length); i++) {
        const d = Math.abs(vA[i] - vB[i]);
        if (d > m) m = d;
      }
      return m;
    } finally {
      outA.free();
      outB.free();
      if (bA.free) bA.free();
      if (bB.free) bB.free();
    }
  };

  try {
    // Level 0: initial generateCases sweep
    const initial = generateCases(baseSpec, { seed, count: samplesPerLevel, lo, hi }).cases;
    let bestCase = initial[0];
    let bestDelta = -1;
    for (const c of initial) {
      const d = evalDelta(c);
      if (d > bestDelta) { bestDelta = d; bestCase = c; }
    }
    const level0MaxDelta = bestDelta;
    const history = [{ depth: 0, radius: (hi - lo) / 2, max_abs_error: bestDelta }];

    let radius = (hi - lo) / 4;
    for (let d = 1; d <= zoomDepth; d++) {
      for (let s = 0; s < samplesPerLevel; s++) {
        const trial = {};
        for (const p of ports) {
          const k = String(p.slot);
          const center = bestCase[k];
          if (vocabs.has(p.slot)) {
            trial[k] = [...center];
          } else {
            trial[k] = center.map((v) => {
              const jitter = (rng() * 2 - 1) * radius;
              return Math.min(hi, Math.max(lo, v + jitter));
            });
          }
        }
        const delta = evalDelta(trial);
        if (delta > bestDelta) {
          bestDelta = delta;
          bestCase = trial;
        }
      }
      history.push({ depth: d, radius, max_abs_error: bestDelta });
      radius *= 0.5;
    }

    // Infinity (length mismatch) would serialize to null in JSON — emit the
    // string 'Infinity' instead, so a 0/mismatch can never be confused again.
    const fmtDelta = (d) => (Number.isFinite(d) ? d : 'Infinity');
    return {
      ok: true,
      seed,
      zoom_depth: zoomDepth,
      samples_per_level: samplesPerLevel,
      initial_max_abs_error: fmtDelta(level0MaxDelta),
      zoomed_max_abs_error: fmtDelta(bestDelta),
      output_lengths: { base: outLenA, candidate: outLenB },
      length_mismatch: outLenA !== null && outLenB !== null && outLenA !== outLenB,
      amplification_ratio: (level0MaxDelta > 0 && Number.isFinite(level0MaxDelta) && Number.isFinite(bestDelta)) ? +(bestDelta / level0MaxDelta).toFixed(4) : null,
      worst_case_input: bestCase,
      zoom_history: history.map((h) => ({ ...h, max_abs_error: fmtDelta(h.max_abs_error) })),
    };
  } finally {
    if (baseBuilt.g?.free) baseBuilt.g.free();
    if (baseBuilt.plan?.free) baseBuilt.plan.free();
    if (baseBuilt.reg?.free) baseBuilt.reg.free();
    if (candBuilt.g?.free) candBuilt.g.free();
    if (candBuilt.plan?.free) candBuilt.plan.free();
    if (candBuilt.reg?.free) candBuilt.reg.free();
  }
}
