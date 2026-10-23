#!/usr/bin/env node
// burn-math #13 Fase 3: per-layer quantization budget planner.
//
// Usage:
//   node bin/quant_budget.mjs <spec.json> <cases.json> --tol 0.05
//       [--scheme q8-per-block] [--mode bound|measured]
//   node bin/quant_budget.mjs --selftest   (9 regression checks, needs the wasm pkg)
//
// Decides WHICH linear layers to quantize (int8) so the checkpoint shrinks
// as much as possible while the end-to-end output perturbation stays within
// --tol (oo-norm: max |y_base - y_plan| over cases and output elements).
//
// --mode bound    (default) analytical & instant: compositional worst-case
//                 bound from the weights BEFORE running. Rigorous but
//                 conservative. Handles linear + relu/tanh/sigmoid + add.
// --mode measured empirical iterative greedy: candidate plans are actually
//                 run; the reported total is the measured delta of the
//                 final plan itself. Tighter, costs O(N^2) runs.
//
// cases.json: array of input maps, e.g. [{"0":[1,-2],"1":[3,10.5]}].
// Prints the plan JSON. Exit 0 always (plan may be empty); 1 on error.
import { loadWasm } from '../lib/engine.mjs';
import { loadSpec } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import {
  linearLayerInfos,
  fetchQuantReports,
  analyticalContribs,
  analyticalPlan,
  measuredPlan,
  inputBoundsFromCases,
  dequantizedWeights,
  operatorNormInf,
  runSpecOutputs,
  maxAbsDiff,
} from '../lib/quant_budget.mjs';
import fs from 'node:fs';

const args = process.argv.slice(2);
function opt(name, def) {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
}
const positionals = [];
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--tol' || a === '--scheme' || a === '--mode') { i++; continue; }
  if (!a.startsWith('--')) positionals.push(a);
}
const [specFile, casesFile] = positionals;
const tol = Number(opt('--tol', 'NaN'));
const scheme = opt('--scheme', 'q8-per-block');
const mode = opt('--mode', 'bound');

if (args.includes('--selftest')) {
  const results = [];
  const t = (name, ok, extra = {}) => { results.push({ name, ok, ...extra }); if (!ok) console.error('FAIL:', name, extra); };
  const rt = await loadWasm();
  // Deterministic PRNG (mulberry32) — no Math.random anywhere.
  const rng = (() => { let s = 1234; return () => { s |= 0; s = (s + 0x6D2B79F5) | 0; let z = Math.imul(s ^ (s >>> 15), 1 | s); z = (z + Math.imul(z ^ (z >>> 7), 61 | z)) ^ z; return ((z ^ (z >>> 14)) >>> 0) / 4294967296; }; })();
  // Synthetic spec: linear 8->16 (outlier weight) + relu + linear 16->4.
  // Sized >50 params/layer so int8 actually saves bytes.
  const w0 = []; for (let k = 0; k < 8; k++) for (let i = 0; i < 16; i++) w0.push((rng() - 0.5) * 2);
  w0[0] = 8.0; // outlier in row 0's block
  const b0 = Array.from({ length: 16 }, () => (rng() - 0.5));
  const w2 = []; for (let k = 0; k < 16; k++) for (let i = 0; i < 4; i++) w2.push((rng() - 0.5) * 2);
  const b2 = Array.from({ length: 4 }, () => (rng() - 0.5));
  const spec = {
    layers: [
      { constructor: 'linear', args: [0, 8, 16, true] },
      { constructor: 'relu', args: [1] },
      { constructor: 'linear', args: [2, 16, 4, true] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 1] },
      { kind: 'unary', layer: 1, slots: [1, 2] },
      { kind: 'unary', layer: 2, slots: [2, 3] },
    ],
    numSlots: 4, outputSlot: 3,
    ports: [{ slot: 0, role: 'observation', shape: [1, 8, 1, 1], layout: 'feature_axis1_singleton' }],
    weights: { '0': [...w0, ...b0], '2': [...w2, ...b2] },
  };
  const cases = [0, 1, 2].map(() => ({ '0': Array.from({ length: 8 }, () => (rng() - 0.5) * 4) }));
  const scheme = 'q8-per-block';
  const infos = linearLayerInfos(spec);
  const reports = fetchQuantReports(rt, spec, scheme, infos);
  t('two reports, both save bytes', reports.length === 2 && reports.every((r) => r.savedBytes > 0),
    { saved: reports.map((r) => r.savedBytes) });

  // 1. Exact error norms never exceed the report's analytical worst case.
  for (const info of infos) {
    const rep = reports.find((r) => r.index === info.index);
    const wParam = rep.report.params.reduce((a, b) => (a.shape.length >= b.shape.length ? a : b));
    const bParam = rep.report.params.find((p) => p !== wParam && p.shape.length <= 1);
    const deq = dequantizedWeights(rt, info, scheme);
    const nW = info.inDim * info.outDim;
    const errW = operatorNormInf(info.weights.slice(0, nW).map((w, i) => w - deq[i]), info.inDim, info.outDim);
    const errB = info.bias ? Math.max(...info.weights.slice(nW).map((b, i) => Math.abs(b - deq[nW + i]))) : 0;
    t(`layer ${info.index}: ||dW||_oo <= bound_W*inDim (${errW.toFixed(4)} <= ${(wParam.bound * info.inDim).toFixed(4)})`,
      errW <= wParam.bound * info.inDim + 1e-6);
    if (bParam) t(`layer ${info.index}: ||db||_oo <= bound_b`, errB <= bParam.bound + 1e-6);
  }

  // 2. Rigor: analytical total >= measured all-quantized delta.
  const { total } = analyticalContribs(spec, reports, inputBoundsFromCases(cases), rt, scheme);
  const vspec = JSON.parse(JSON.stringify(spec));
  vspec.weights = { ...spec.weights };
  for (const info of infos) vspec.weights[String(info.index)] = dequantizedWeights(rt, info, scheme);
  const base = runSpecOutputs(rt, spec, cases), q = runSpecOutputs(rt, vspec, cases);
  let delta = 0;
  for (let c = 0; c < cases.length; c++) delta = Math.max(delta, maxAbsDiff(base[c], q[c]));
  t(`rigor: bound total ${total.toFixed(4)} >= measured delta ${delta.toFixed(4)}`, total >= delta);

  // 3. Decisions: huge tol quantizes all saving layers; tol 0 keeps all f32.
  const pAll = analyticalPlan(spec, cases, scheme, 1e9, rt, infos);
  t('tol=1e9 quantizes both layers', JSON.stringify(pAll.quantize) === '[0,2]', pAll.quantize);
  const pNone = analyticalPlan(spec, cases, scheme, 0, rt, infos);
  t('tol=0 quantizes nothing', pNone.quantize.length === 0, pNone.quantize);

  // 4. measuredPlan still discriminates (sanity, unchanged behavior).
  const pm = measuredPlan(rt, spec, cases, scheme, 1e9, infos);
  t('measured tol=1e9 quantizes both', JSON.stringify(pm.quantize) === '[0,2]', pm.quantize);

  const fails = results.filter((r) => !r.ok);
  console.log(`selftest: ${results.length - fails.length}/${results.length} pass`);
  process.exit(fails.length ? 1 : 0);
}

if (!specFile || !casesFile) {
  console.error('usage: node bin/quant_budget.mjs <spec.json> <cases.json> --tol <num> [--scheme q8-per-block] [--mode bound|measured]');
  process.exit(1);
}
if (!Number.isFinite(tol) || tol < 0) { console.error('need --tol >= 0'); process.exit(1); }
if (mode !== 'bound' && mode !== 'measured') { console.error('mode must be bound|measured'); process.exit(1); }

try {
  const rt = await loadWasm();
  const spec = loadSpec(specFile);
  const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
  if (!Array.isArray(cases) || cases.length === 0) throw new Error('cases must be a non-empty array');
  const infos = linearLayerInfos(spec);
  if (infos.length === 0) throw new Error('spec has no weighted linear layers to plan over');

  let result;
  if (mode === 'bound') {
    const p = analyticalPlan(spec, cases, scheme, tol, rt, infos);
    const keep = infos.map((i) => i.index).filter((i) => !p.quantize.includes(i));
    result = {
      ok: true, mode, scheme, tol,
      layers: p.reports.map((r) => {
        const c = p.contribs.find((x) => x.index === r.index);
        return {
          index: r.index, constructor: 'linear',
          shape: r.report.params[0]?.shape ?? null,
          bytes_f32: r.bytesF32, bytes_q8: r.bytesQ8, saved_bytes: r.savedBytes,
          bound_contrib: c ? c.contrib : null,
          decision: p.quantize.includes(r.index) ? 'quantize' : 'keep_f32',
        };
      }),
      plan: { quantize: p.quantize, keep_f32: keep },
      totals: {
        bytes_f32: p.reports.reduce((s, r) => s + r.bytesF32, 0),
        bytes_q8_plan: p.reports.reduce((s, r) => s + (p.quantize.includes(r.index) ? r.bytesQ8 : r.bytesF32), 0),
        saved_bytes: p.saved,
        bound_total_all: p.boundTotal,
        bound_total_plan: p.cost,
        within_tol: p.cost <= tol,
      },
      notes: [
        'bound mode: worst case computed BEFORE running; rigorous but conservative',
        'contrib = (||dW||_oo*||x||_oo + ||db||_oo) * downstream_amplification (dW, db exact)',
      ],
    };
  } else {
    const p = measuredPlan(rt, spec, cases, scheme, tol, infos);
    const repByIndex = new Map(p.reports.map((r) => [r.index, r]));
    const keep = infos.map((i) => i.index).filter((i) => !p.quantize.includes(i));
    const saved = p.quantize.reduce((s, i) => s + repByIndex.get(i).savedBytes, 0);
    result = {
      ok: true, mode, scheme, tol,
      layers: p.reports.map((r) => ({
        index: r.index, constructor: 'linear',
        shape: r.report.params[0]?.shape ?? null,
        bytes_f32: r.bytesF32, bytes_q8: r.bytesQ8, saved_bytes: r.savedBytes,
        decision: p.quantize.includes(r.index) ? 'quantize' : 'keep_f32',
      })),
      plan: { quantize: p.quantize, keep_f32: keep },
      totals: {
        bytes_f32: p.reports.reduce((s, r) => s + r.bytesF32, 0),
        bytes_q8_plan: p.reports.reduce((s, r) => s + (p.quantize.includes(r.index) ? r.bytesQ8 : r.bytesF32), 0),
        saved_bytes: saved,
        measured_delta_plan: p.totalDelta,
        within_tol: p.totalDelta <= tol,
      },
      notes: [
        'measured mode: iterative greedy with exact re-measurement; reported delta is the measured end-to-end perturbation of the final plan',
      ],
    };
  }
  printJson(result);
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e?.message ?? String(e) });
  process.exit(1);
}
