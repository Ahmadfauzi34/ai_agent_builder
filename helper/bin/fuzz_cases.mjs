#!/usr/bin/env node
// burn-math fuzz_cases: deterministic test-case generation for verify/diff_ref.
//
// Usage: node fuzz_cases.mjs <spec.json> [out.json] [--seed 42] [--count 8] [--only-slots 0,1] [--lo -2] [--hi 2]
//
// Builds a cases.json array (verify-compatible) via lib/fuzz.mjs — the
// same generator verify.mjs --fuzz uses, so the two can never drift:
//   case 0 : the spec's own inputs (the golden case), when present
//   then   : boundary cases — zeros, ones, alternating +/-1, large (+50),
//            small (+1e-4), negative large (-50)
//   then   : --count seeded-random cases, uniform [lo,hi) via mulberry32
// Per-port, honoring each port's shape and domain. Discrete ports
// (layout token_ids_axis1_singleton, e.g. embedding ids) get integer
// ids in [0, vocab) instead of continuous values. Fully deterministic
// for a given seed: the seed is reported in the envelope so runs are
// reproducible.
//
// Prints {ok, seed, cases, out_file|null} and writes the bare array to
// out.json (or stdout as {envelope, cases} when no file given).
import { loadSpec } from '../lib/spec.mjs';
import { generateCases } from '../lib/fuzz.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
const specFile = rawArgs[0];
if (!specFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
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
  const { cases, meta } = generateCases(spec, opts);
  const envelope = {
    ok: true,
    seed: meta.seed,
    prng: meta.prng,
    random_range: meta.random_range,
    ports: meta.ports,
    cases: cases.length,
    includes_golden: meta.includes_golden,
    discrete_ports: meta.discrete_ports,
  };
  // Both branches exit drain-safely; the trailing exit(0) was dead code
  // once both prints exit, so it is gone (it also used to truncate the
  // large cases_data output past ~64KB).
  if (opts.out) {
    fs.writeFileSync(opts.out, JSON.stringify(cases));
    printJsonAndExit({ ...envelope, out_file: opts.out }, 0);
  } else {
    printJsonAndExit({ ...envelope, cases_data: cases }, 0);
  }
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
