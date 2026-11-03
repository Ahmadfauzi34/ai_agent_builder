#!/usr/bin/env node
// burn-math fuzz_cases: deterministic test-case generation for verify / diff_ref
// + Recursive Adversarial Interval Narrowing (`--zoom-vs <cand.json> [--zoom-depth 4]`).
//
// Usage: node fuzz_cases.mjs <spec.json> [out.json] [--seed N] [--count N] [--lo X] [--hi X] [--only-slots s0,s1] [--zoom-vs cand.json] [--zoom-depth D]
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
// Prints the envelope {ok, seed, cases, includes_golden, discrete_ports,
// meta, out_file|null} (plus cases_data on stdout when no file is given)
// and writes the cases array to out.json.
import fs from 'node:fs';
import { loadSpec } from '../lib/spec.mjs';
import { printJson, printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';
import { generateCases, findWorstCaseInput } from '../lib/fuzz.mjs';
import { loadWasm } from '../lib/wasm_direct.mjs';

const args = process.argv.slice(2);
const specFile = args[0];
if (!specFile || specFile.startsWith('--') || args.includes('--help') || args.includes('-h')) {
  console.error('usage: node fuzz_cases.mjs <spec.json> [out.json] [--seed 42] [--count 8] [--lo -2] [--hi 2] [--only-slots 0,1] [--zoom-vs cand.json] [--zoom-depth 4]');
  process.exit(2);
}
assertKnownFlags(args.slice(1), { valued: ['--seed', '--count', '--lo', '--hi', '--only-slots', '--zoom-vs', '--zoom-depth'], prog: 'fuzz_cases' });

let outFile = null, seed = 42, count = 8, lo = -2, hi = 2, onlySlots = null;
let zoomVsFile = null, zoomDepth = 4;
for (let i = 1; i < args.length; i++) {
  const a = args[i];
  if (a === '--seed') seed = Number(args[++i]);
  else if (a === '--count') count = Number(args[++i]);
  else if (a === '--lo') lo = Number(args[++i]);
  else if (a === '--hi') hi = Number(args[++i]);
  else if (a === '--only-slots') onlySlots = args[++i].split(',').map(Number);
  else if (a === '--zoom-vs') zoomVsFile = args[++i];
  else if (a === '--zoom-depth') zoomDepth = Number(args[++i]);
  else if (!a.startsWith('--')) {
    if (outFile) throw new Error(`unexpected argument: ${a}`);
    outFile = a;
  }
}
if (!Number.isInteger(seed) || !Number.isInteger(count) || count < 0)
  throw new Error('--seed and --count must be integers, --count >= 0');

try {
  const spec = loadSpec(specFile);
  if (zoomVsFile) {
    const candSpec = loadSpec(zoomVsFile);
    const rt = await loadWasm();
    const zoomRes = findWorstCaseInput(rt, spec, candSpec, {
      seed,
      samplesPerLevel: Math.max(4, count),
      zoomDepth,
      lo,
      hi,
    });
    if (outFile) {
      fs.writeFileSync(outFile, JSON.stringify([zoomRes.worst_case_input], null, 1));
      zoomRes.out_file = outFile;
    }
    printJsonAndExit(zoomRes, 0);
  }

  const { cases, meta } = generateCases(spec, { seed, count, lo, hi, onlySlots });
  const envelope = {
    ok: true,
    seed: meta.seed,
    prng: meta.prng,
    random_range: meta.random_range,
    ports: meta.ports,
    cases: cases.length,
    includes_golden: meta.includes_golden,
    discrete_ports: meta.discrete_ports,
    meta,
  };
  if (outFile) {
    fs.writeFileSync(outFile, JSON.stringify(cases, null, 1));
    printJson({ ...envelope, out_file: outFile });
  } else {
    printJson({ ...envelope, cases_data: cases });
  }
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
