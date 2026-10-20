#!/usr/bin/env node
// burn-math verify: baseline-vs-candidate GRAPH verification (WASM-direct).
// Usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]
//   cases.json: [{"0":[...],"1":[...]}, ...]  (slot -> values per case)
//   default: one case taken from baseline.inputs, applied to both graphs.
// Prints the verification JSON: {equivalent, max_abs_error, max_rel_error,
// compared_f32_count, first_failure_case_index, ...}. Exit 0 when equivalent,
// 2 when not (or on error).
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import fs from 'node:fs';

const [baseFile, candFile, casesFile, absTolArg, relTolArg] = process.argv.slice(2);
if (!baseFile || !candFile) {
  console.error('usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]');
  process.exit(2);
}

const absTol = absTolArg !== undefined ? Number(absTolArg) : 1e-6;
const relTol = relTolArg !== undefined ? Number(relTolArg) : 1e-6;

try {
  const baseSpec = loadSpec(baseFile);
  const candSpec = loadSpec(candFile);
  const cases = casesFile
    ? JSON.parse(fs.readFileSync(casesFile, 'utf8'))
    : [baseSpec.inputs];
  if (!Array.isArray(cases) || cases.length === 0) throw new Error('cases must be a non-empty array');

  const rt = await loadWasm();
  const base = buildGraph(rt, baseSpec);
  const cand = buildGraph(rt, candSpec);
  const basePorts = portBySlot(baseSpec);
  const candPorts = portBySlot(candSpec);

  const vc = new rt.MultiInputVerificationCases(base.g, cand.g);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, base.plan, c, basePorts), makeBundle(rt, cand.plan, c, candPorts));
  }
  const result = JSON.parse(vc.verify(base.reg, base.g, cand.reg, cand.g, absTol, relTol));
  printJson({ ok: true, cases: cases.length, absTol, relTol, ...result });
  process.exit(result.equivalent ? 0 : 2);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
