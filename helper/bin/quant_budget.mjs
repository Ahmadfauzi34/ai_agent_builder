#!/usr/bin/env node
// burn-math #13 Fase 3: per-layer quantization budget planner.
//
// Usage:
//   node bin/quant_budget.mjs <spec.json> <cases.json> --tol 0.05
//       [--scheme q8-per-block] [--mode bound|measured]
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
  analyticalPlan,
  measuredPlan,
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
        'contrib = (bound_W*inDim*||x||_oo + bound_b) * downstream_amplification',
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
