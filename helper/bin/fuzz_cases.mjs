#!/usr/bin/env node
// burn-math fuzz_cases: deterministic test-case generation for verify/diff_ref.
//
// Usage: node fuzz_cases.mjs <spec.json> [out.json] [--seed 42] [--count 8] [--only-slots 0,1] [--lo -2] [--hi 2]
//
// Builds a cases.json array (verify-compatible):
//   case 0 : the spec's own inputs (the golden case), when present
//   then   : boundary cases — zeros, ones, alternating +/-1, large (+50),
//            small (+1e-4), negative large (-50)
//   then   : --count seeded-random cases, uniform [lo,hi) via mulberry32
// Per-port, honoring each port's shape. Fully deterministic for a given seed:
// the seed is reported in the envelope so runs are reproducible.
//
// Prints {ok, seed, cases, out_file|null} and writes the bare array to
// out.json (or stdout as {envelope, cases} when no file given).
// NOTE: for continuous-valued inputs. Discrete index ports (e.g. embedding
// ids) should be excluded with --only-slots.
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import { generateStream } from '../lib/prng.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
const specFile = rawArgs[0];
if (!specFile) {
  console.error('usage: node fuzz_cases.mjs <spec.json> [out.json] [--seed 42] [--count 8] [--only-slots 0,1] [--lo -2] [--hi 2]');
  process.exit(2);
}
const opts = { seed: 42, count: 8, lo: -2, hi: 2, onlySlots: null, out: null };
for (let i = 1; i < rawArgs.length; i++) {
  const a = rawArgs[i];
  if (a === '--seed') opts.seed = Number(rawArgs[++i]);
  else if (a === '--count') opts.count = Number(rawArgs[++i]);
  else if (a === '--only-slots') opts.onlySlots = rawArgs[++i].split(',').map(Number);
  else if (a === '--lo') opts.lo = Number(rawArgs[++i]);
  else if (a === '--hi') opts.hi = Number(rawArgs[++i]);
  else if (!opts.out) opts.out = a;
  else throw new Error(`unexpected argument: ${a}`);
}
if (!Number.isInteger(opts.seed) || !Number.isInteger(opts.count) || opts.count < 0)
  throw new Error('--seed and --count must be integers, --count >= 0');

try {
  const spec = loadSpec(specFile);
  const ports = [...portBySlot(spec).values()]
    .filter((p) => !opts.onlySlots || opts.onlySlots.includes(p.slot))
    .sort((a, b) => a.slot - b.slot);
  if (ports.length === 0) throw new Error('no ports selected');
  const lenOf = (p) => p.shape.reduce((x, y) => x * y, 1);

  const mk = (fn) => {
    const c = {};
    for (const p of ports) {
      const n = lenOf(p);
      c[String(p.slot)] = Array.from({ length: n }, (_, i) => fn(i, n));
    }
    return c;
  };
  const cases = [];
  // 0. golden case: the spec's own inputs
  if (spec.inputs && Object.keys(spec.inputs).length > 0) {
    const g = {};
    for (const p of ports) {
      const v = spec.inputs[String(p.slot)];
      if (v === undefined) throw new Error(`spec.inputs missing port slot ${p.slot} (use --only-slots or add it)`);
      g[String(p.slot)] = [...v];
    }
    cases.push(g);
  }
  // 1. boundary cases
  cases.push(mk(() => 0));
  cases.push(mk(() => 1));
  cases.push(mk((i) => (i % 2 === 0 ? 1 : -1)));
  cases.push(mk(() => 50));
  cases.push(mk(() => 1e-4));
  cases.push(mk(() => -50));
  // 2. seeded random cases
  const total = ports.reduce((a, p) => a + lenOf(p), 0);
  const { values } = generateStream('mulberry32', opts.seed, opts.count * total, opts.lo, opts.hi);
  for (let k = 0; k < opts.count; k++) {
    const c = {};
    let off = k * total;
    for (const p of ports) {
      const n = lenOf(p);
      c[String(p.slot)] = values.slice(off, off + n);
      off += n;
    }
    cases.push(c);
  }

  const envelope = {
    ok: true,
    seed: opts.seed,
    prng: 'mulberry32',
    random_range: [opts.lo, opts.hi],
    ports: ports.map((p) => p.slot),
    cases: cases.length,
    includes_golden: !!(spec.inputs && Object.keys(spec.inputs).length > 0),
  };
  if (opts.out) {
    fs.writeFileSync(opts.out, JSON.stringify(cases));
    printJson({ ...envelope, out_file: opts.out });
  } else {
    printJson({ ...envelope, cases_data: cases });
  }
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
