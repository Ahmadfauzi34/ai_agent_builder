#!/usr/bin/env node
// burn-math diff_ref: differential testing — engine vs an INDEPENDENT reference.
//
// Usage: node diff_ref.mjs <spec.json> [cases.json] [tol=1e-5]
//        node diff_ref.mjs <spec.json> --ref <ref.json> [cases.json] [tol=1e-5]
//
// Two modes:
//   (default)  engine (f32 WASM) vs the built-in f64 JS reference
//              (../lib/ref.mjs — separate code path, higher precision).
//   --ref      engine vs externally computed reference vectors, e.g. from
//              torch/numpy on another machine. ref.json: {"values": [[...], ...]}
//              parallel to the cases array.
//
// A discrepancy beyond f32 rounding is a SIGNAL, not a verdict: investigate
// which side is wrong. Prints {ok, mode, per_case, verdict}. Exit 0 iff all
// cases within tol, 2 otherwise.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import { runReference, SUPPORTED } from '../lib/ref.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
const specFile = rawArgs[0];
if (!specFile) {
  console.error('usage: node diff_ref.mjs <spec.json> [cases.json] [tol=1e-5]');
  console.error('       node diff_ref.mjs <spec.json> --ref <ref.json> [cases.json] [tol=1e-5]');
  process.exit(2);
}
let refFile = null;
const positionals = [];
for (let i = 1; i < rawArgs.length; i++) {
  if (rawArgs[i] === '--ref') refFile = rawArgs[++i];
  else positionals.push(rawArgs[i]);
}
const [casesFile, tolArg] = positionals;
const tol = tolArg !== undefined ? Number(tolArg) : 1e-5;
if (!(tol > 0)) {
  printJsonAndExit({ ok: false, error: 'tol must be a positive number' }, 2);
}

function engineRun(rt, built, ports, inputs) {
  const bundle = makeBundle(rt, built.plan, inputs, ports);
  const out = built.g.run(built.reg, bundle);
  const vals = Array.from(out.to_array());
  out.free();
  if (bundle.free) bundle.free();
  return vals;
}

try {
  const spec = loadSpec(specFile);
  const cases = casesFile ? JSON.parse(fs.readFileSync(casesFile, 'utf8')) : [spec.inputs];
  if (!Array.isArray(cases) || cases.length === 0) throw new Error('cases must be a non-empty array');
  const extRef = refFile ? JSON.parse(fs.readFileSync(refFile, 'utf8')) : null;
  if (extRef) {
    if (!Array.isArray(extRef.values) || extRef.values.length !== cases.length)
      throw new Error('ref.json must be {"values": [[...], ...]} with one entry per case');
  }

  const rt = await loadWasm();
  const built = buildGraph(rt, spec);
  const ports = portBySlot(spec);

  const perCase = [];
  let maxAbs = 0, maxRel = 0, allPass = true;
  cases.forEach((c, ci) => {
    const eng = engineRun(rt, built, ports, c);
    let ref;
    if (extRef) {
      ref = extRef.values[ci];
      if (!Array.isArray(ref) || ref.length !== eng.length)
        throw new Error(`ref case ${ci}: length ${ref?.length} != engine output ${eng.length}`);
    } else {
      ref = runReference(spec, c);
    }
    let cAbs = 0, cRel = 0;
    for (let i = 0; i < eng.length; i++) {
      const a = Math.abs(eng[i] - ref[i]);
      const r = a / Math.max(Math.abs(ref[i]), 1e-12);
      if (a > cAbs) cAbs = a;
      if (r > cRel) cRel = r;
    }
    const passed = cAbs <= tol || cRel <= tol;
    if (!passed) allPass = false;
    if (cAbs > maxAbs) maxAbs = cAbs;
    if (cRel > maxRel) maxRel = cRel;
    perCase.push({ index: ci, max_abs_error: cAbs, max_rel_error: cRel, passed });
  });
  if (built.g.free) built.g.free();

  printJsonAndExit({
    ok: true,
    mode: extRef ? 'external-ref' : 'js-f64-reference',
    reference_ops: extRef ? null : SUPPORTED,
    cases: cases.length,
    tol,
    per_case: perCase,
    max_abs_error: maxAbs,
    max_rel_error: maxRel,
    verdict: allPass
      ? 'PASS: engine agrees with reference within tolerance'
      : 'SIGNAL: engine/reference discrepancy beyond f32 rounding — investigate which side is wrong',
  }, allPass ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
