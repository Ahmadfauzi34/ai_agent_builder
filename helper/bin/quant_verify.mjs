#!/usr/bin/env node
// burn-math quant_verify: weight-quantization simulation + verification.
//
// Usage: node quant_verify.mjs <baseline.json> [--int8|--fp16] [cases.json] [absTol=1e-2] [relTol=1e-2]
//
// Simulates static weight-only quantization of the baseline spec and verifies
// the quantized graph against the fp32 baseline over the given cases:
//   --int8 : symmetric per-tensor int8 (scale = maxAbs/127), the deployment case
//   --fp16 : fp16 round-trip per weight (Float16Array)
// Reports per-layer quantization error (where the error ENTERS) plus the
// end-to-end output error (where it MATTERS), weight bytes fp32 vs quantized,
// and a PASS/FAIL verdict against the given tolerance budget.
// Prints {ok, mode, layers, totals, verification, verdict}. Exit 0 iff PASS.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
const specFile = rawArgs[0];
if (!specFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node quant_verify.mjs <baseline.json> [--int8|--fp16] [cases.json] [absTol=1e-2] [relTol=1e-2]');
  process.exit(2);
}
let mode = 'int8';
const positionals = [];
for (const a of rawArgs.slice(1)) {
  if (a === '--int8') mode = 'int8';
  else if (a === '--fp16') mode = 'fp16';
  else positionals.push(a);
}
const [casesFile, absTolArg, relTolArg] = positionals;
const absTol = absTolArg !== undefined ? Number(absTolArg) : 1e-2;
const relTol = relTolArg !== undefined ? Number(relTolArg) : 1e-2;

function quantizeInt8(w) {
  let maxAbs = 0;
  for (const v of w) { const a = Math.abs(v); if (a > maxAbs) maxAbs = a; }
  const scale = maxAbs === 0 ? 1 : maxAbs / 127;
  const q = w.map((v) => {
    const qi = Math.round(v / scale);
    return Math.max(-127, Math.min(127, qi)) * scale;
  });
  return { q, scale };
}
function quantizeFp16(w) {
  const q = Array.from(new Float16Array(w));
  return { q, scale: null };
}

try {
  const baseSpec = loadSpec(specFile);
  const cases = casesFile
    ? JSON.parse(fs.readFileSync(casesFile, 'utf8'))
    : [baseSpec.inputs];
  if (!Array.isArray(cases) || cases.length === 0) throw new Error('cases must be a non-empty array');

  // --- quantize a deep copy of the spec ---
  const qSpec = JSON.parse(JSON.stringify(baseSpec));
  const layers = [];
  let bytesFp32 = 0;
  let bytesQuant = 0;
  for (const [indexStr, values] of Object.entries(baseSpec.weights ?? {})) {
    const idx = Number(indexStr);
    const { q, scale } = mode === 'int8' ? quantizeInt8(values) : quantizeFp16(values);
    qSpec.weights[indexStr] = q;
    let wAbs = 0;
    for (let i = 0; i < values.length; i++) {
      const a = Math.abs(values[i] - q[i]);
      if (a > wAbs) wAbs = a;
    }
    const b32 = values.length * 4;
    const bq = values.length * (mode === 'int8' ? 1 : 2) + (mode === 'int8' ? 4 : 0);
    bytesFp32 += b32;
    bytesQuant += bq;
    layers.push({
      layer_index: idx,
      constructor: baseSpec.layers[idx]?.constructor ?? 'unknown',
      weight_count: values.length,
      scale,
      max_abs_weight_error: wAbs,
      bytes_fp32: b32,
      bytes_quant: bq,
    });
  }
  if (layers.length === 0) throw new Error('spec has no weights to quantize');

  // --- verify quantized graph vs fp32 baseline ---
  const rt = await loadWasm();
  const base = buildGraph(rt, baseSpec);
  const quant = buildGraph(rt, qSpec);
  const basePorts = portBySlot(baseSpec);
  const quantPorts = portBySlot(qSpec);
  const vc = new rt.MultiInputVerificationCases(base.g, quant.g);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, base.plan, c, basePorts), makeBundle(rt, quant.plan, c, quantPorts));
  }
  const v = JSON.parse(vc.verify(base.reg, base.g, quant.reg, quant.g, absTol, relTol));

  const pass = v.equivalent === true;
  printJsonAndExit({
    ok: true,
    mode,
    cases: cases.length,
    absTol,
    relTol,
    layers,
    totals: {
      weight_bytes_fp32: bytesFp32,
      weight_bytes_quant: bytesQuant,
      compression_ratio: +(bytesFp32 / bytesQuant).toFixed(2),
    },
    verification: {
      equivalent: v.equivalent,
      max_abs_error: v.max_abs_error,
      max_rel_error: v.max_rel_error,
      compared_f32_count: v.compared_f32_count,
    },
    verdict: pass ? 'PASS: quantized graph within tolerance budget' : 'FAIL: quantized graph exceeds tolerance budget',
    notes: [
      'weight-only static quantization; activations stay fp32',
      'per-layer max_abs_weight_error shows where error ENTERS; verification max_abs_error shows where it MATTERS (end to end)',
    ],
  }, pass ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
