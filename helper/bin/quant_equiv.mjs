#!/usr/bin/env node
// burn-math #13 Fase 2: verify a layer's REAL quantized state round-trip.
//
// Usage:
//   node bin/quant_equiv.mjs --in 4 --out 8 [--bias] [--scheme q8-per-block]
//                            [--tol auto|<number>] [--cases 5] [--seed 7]
//
// Builds a WasmLinear(inDim, outDim, bias), quantizes via the engine's own
// getQuantizedState, loads the bytes into a fresh layer (exercising the BQZ1
// envelope + load_state auto-detect), and compares forward outputs over
// deterministic input cases.
//
// --tol auto    : derive the conservative analytical tolerance from the
//                 engine's quant report (|Δy| <= bound_W*Σ|x| + bound_b)
// --tol <num>   : fixed tolerance
// (omitted)     : MEASURED verdict — numbers reported, no judgment
//
// Prints the verdict JSON. Exit 0 on ACCEPT or MEASURED, 2 on REJECT,
// 1 on error.
import { loadWasm } from '../lib/engine.mjs';
import { quantizeLayerEquivalence, linearQuantTolerance } from '../lib/quant_equiv.mjs';
import { mulberry32 } from '../lib/prng.mjs';
import { printJson } from '../lib/runner.mjs';

const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) {
  console.error('usage: node bin/quant_equiv.mjs --in 4 --out 8 [--bias] [--scheme q8-per-block] [--tol auto|<number>] [--cases 5] [--seed 7]');
  process.exit(2);
}
function opt(name, def) {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
}
function flag(name) {
  return args.includes(name);
}

const inDim = Number(opt('--in', '4'));
const outDim = Number(opt('--out', '8'));
const bias = flag('--bias');
const scheme = opt('--scheme', 'q8-per-block');
const tolArg = opt('--tol', null);
const nCases = Number(opt('--cases', '5'));
const seed = Number(opt('--seed', '7'));

if (!Number.isInteger(inDim) || inDim <= 0 || !Number.isInteger(outDim) || outDim <= 0) {
  console.error('need positive integer --in/--out');
  process.exit(1);
}

try {
  const rt = await loadWasm();
  const rng = mulberry32(seed);
  const inputCases = [];
  for (let c = 0; c < nCases; c++) {
    const values = Array.from({ length: inDim }, () => rng() * 4 - 2);
    inputCases.push({ values, shape: [1, inDim, 1, 1] });
  }
  // Deterministic non-zero weights (separate stream), with one outlier to
  // exercise per-block quantization. Layout: [in*out] row-major [in,out]
  // + bias appended (audited: flat[k*out_dim + i]).
  const wrng = mulberry32(seed + 1);
  const weights = Array.from({ length: inDim * outDim }, () => wrng() * 4 - 2);
  weights[0] = 10.0; // outlier
  if (bias) for (let i = 0; i < outDim; i++) weights.push(wrng() * 2 - 1);
  const makeLayer = () => {
    const l = new rt.WasmLinear(inDim, outDim, bias);
    l.setWeightsFlat(weights);
    return l;
  };
  const opts = {};
  if (tolArg === 'auto') {
    // Analytical tolerance derived from the engine's own quant report.
    opts.tolerance = (report, inputCase) => linearQuantTolerance(report)(inputCase.values);
  } else if (tolArg !== null) {
    const t = Number(tolArg);
    if (!Number.isFinite(t) || t < 0) throw new Error(`bad --tol ${tolArg}`);
    opts.tolerance = t;
  }
  const result = await quantizeLayerEquivalence(rt, makeLayer, scheme, inputCases, opts);
  printJson({ ok: true, layer: `linear(${inDim},${outDim},bias=${bias})`, ...result });
  process.exit(result.verdict === 'REJECT' ? 2 : 0);
} catch (e) {
  printJson({ ok: false, error: e?.message ?? String(e) });
  process.exit(1);
}
