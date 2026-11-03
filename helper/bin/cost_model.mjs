#!/usr/bin/env node
// burn-math cost_model: layout- & domain-aware per-op cost table + predicted-vs-measured total.
//
// Usage: node cost_model.mjs <spec.json> [iterations=11] [--iters N] [--table out.json]
//
// Method (honest microbenchmarking, engine-measured):
//   1. checkSpec gates the graph (shape/dataflow) — fails fast like shapecheck.
//   2. Each step is microbenchmarked in isolation with the SAME input/output
//      shapes: unary ops as <op> + <add-zero>, minus an <add-zero>-alone
//      baseline (differential, so the dummy add doesn't pollute the number);
//      binary ops directly with two input ports.
//      Micro specs carry the propagated port layouts, and embedding inputs
//      use discrete indices in vocab range (not continuous values).
//   3. Per-step out_bytes come from the engine's own runWithTrace.
//   4. predicted_total = Σ per-step ms; measured_total = whole-graph steady
//      median (profileSpec). The reported prediction error tells the agent how
//      much to trust the model — it is a planning signal, not gospel.
//
// Prints {ok, per_op:[...sorted by ms desc], predicted_total_ms,
// measured_total_ms, prediction_error_pct, hotspot}. --table writes the
// per-op rows to a file for reuse across graphs. Exit 0; 2 on shape failure.
import { buildGraph, makeBundle, pkgDir } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';
import { checkSpec } from '../lib/shapecore.mjs';
import { propagateLayouts } from '../lib/explain.mjs';
import { traceSteps, profileSpec, median, loadEngineWithOracle } from '../lib/opt.mjs';
import { mulberry32 } from '../lib/prng.mjs';
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';
import path from 'node:path';

const rawArgs = process.argv.slice(2);
const specFile = rawArgs[0];
if (!specFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node cost_model.mjs <spec.json> [iterations=11] [--iters N] [--table out.json]');
  process.exit(2);
}
let iterations = 11, tableFile = null;
for (let i = 1; i < rawArgs.length; i++) {
  if (rawArgs[i] === '--table') tableFile = rawArgs[++i];
  else if (rawArgs[i] === '--iters' || rawArgs[i] === '--iterations') iterations = Number(rawArgs[++i]);
  else iterations = Number(rawArgs[i]);
}
if (!Number.isInteger(iterations) || iterations < 2) {
  printJsonAndExit({ ok: false, error: 'iterations must be an integer >= 2' }, 2);
}

const prod = (s) => s.reduce((a, b) => a * b, 1);
function fill(rng, n) { return Array.from({ length: n }, () => rng() * 2 - 1); }
function fillDiscrete(rng, n, vocab) { return Array.from({ length: n }, () => Math.floor(rng() * vocab)); }
const zeros = (n) => new Array(n).fill(0);

function loadLayoutCtors() {
  try {
    const p = path.join(pkgDir(), 'docs', 'agent-layout-contracts.v1.json');
    return JSON.parse(fs.readFileSync(p, 'utf8')).constructors ?? {};
  } catch {
    return {};
  }
}

function resolvePortLayout(slotLayout, ctorLayoutDef, isInput = true) {
  if (slotLayout && !['preserve_input', 'dynamic', 'unknown'].includes(slotLayout)) return slotLayout;
  const target = isInput ? ctorLayoutDef?.input : ctorLayoutDef?.output;
  if (target && !['preserve_input', 'dynamic', 'unknown'].includes(target)) return target;
  return 'any_rank4';
}

// Median steady ms of g.run over `iterations` (first = cold, skipped).
function measureSteadyMs(rt, built, ports, inputs) {
  const bundle = makeBundle(rt, built.plan, inputs, ports);
  const ms = [];
  for (let i = 0; i < iterations; i++) {
    const t0 = performance.now();
    const out = built.g.run(built.reg, bundle);
    out.free();
    ms.push(performance.now() - t0);
  }
  if (bundle.free) bundle.free();
  return median(ms.slice(1));
}

try {
  const spec = loadSpec(specFile);
  const chk = checkSpec(spec);
  if (!chk.ok) {
    printJsonAndExit({ ok: false, error: 'spec failed static shape/dataflow check', mismatches: chk.mismatches, dataflow_issues: chk.dataflow.issues }, 2);
  }
  const layoutCtors = loadLayoutCtors();
  const slotLayouts = propagateLayouts(spec, layoutCtors);
  const maxId = Math.max(0, ...spec.layers.map((l) => l.args[0] ?? 0));
  const rng = mulberry32(0xc05);
  const { rt } = await loadEngineWithOracle();

  // Per-step byte map from the engine's own trace.
  const full = buildGraph(rt, spec);
  const fullPorts = portBySlot(spec);
  const fullBundle = makeBundle(rt, full.plan, spec.inputs, fullPorts);
  let traced;
  try {
    traced = traceSteps(rt, spec, full.reg, full.g, fullBundle);
  } finally {
    if (fullBundle?.free) fullBundle.free();
    if (full.g?.free) full.g.free();
    if (full.plan?.free) full.plan.free();
    if (full.reg?.free) full.reg.free();
  }
  const bytesByStep = new Map(traced.map((t) => [t.index, t.out_bytes ?? null]));

  const perOp = [];
  const warnings = [];
  chk.steps.forEach((st, i) => {
    const stepDef = spec.steps[i];
    const layerIdx = stepDef.layer;
    const layerDef = spec.layers[layerIdx];
    const lc = layoutCtors[layerDef.constructor] ?? null;
    const base = {
      step: i, layer: layerDef.constructor, kind: st.kind,
      in_shapes: st.in_shapes, out_shape: st.out_shape, out_bytes: bytesByStep.get(i) ?? null,
    };
    if (st.rule !== 'checked') {
      warnings.push(`step ${i} (${layerDef.constructor}): no reliable shape rule — cost not measured`);
      perOp.push({ ...base, ms_median: null, note: 'unchecked' });
      return;
    }
    const S = st.in_shapes[0], T = st.out_shape;
    const inLayout0 = resolvePortLayout(slotLayouts.get(stepDef.slots[0]), lc, true);
    const outLayout = resolvePortLayout(slotLayouts.get(stepDef.slots[stepDef.slots.length - 1]), lc, false);
    const w = spec.weights?.[String(layerIdx)];
    let ms, clamped = false;

    if (st.kind === 'unary') {
      const isEmb = layerDef.constructor === 'embedding';
      const vocab = isEmb ? (layerDef.args[1] ?? 16) : 16;
      const inVals = isEmb ? fillDiscrete(rng, prod(S), vocab) : fill(rng, prod(S));
      const micro = {
        numSlots: 4,
        layers: [layerDef, { constructor: 'add', args: [maxId + 1] }],
        steps: [
          { kind: 'unary', layer: 0, slots: [0, 2] },
          { kind: 'binary', layer: 1, slots: [2, 1, 3] },
        ],
        outputSlot: 3,
        ports: [
          { slot: 0, role: 'observation', shape: [...S], layout: inLayout0 },
          { slot: 1, role: 'observation', shape: [...T], layout: outLayout },
        ],
        weights: w ? { 0: [...w] } : {},
        inputs: { 0: inVals, 1: zeros(prod(T)) },
      };
      const baseAdd = {
        numSlots: 3,
        layers: [{ constructor: 'add', args: [maxId + 2] }],
        steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
        outputSlot: 2,
        ports: [
          { slot: 0, role: 'observation', shape: [...T], layout: outLayout },
          { slot: 1, role: 'observation', shape: [...T], layout: outLayout },
        ],
        weights: {},
        inputs: { 0: fill(rng, prod(T)), 1: zeros(prod(T)) },
      };
      const mb = buildGraph(rt, micro), bb = buildGraph(rt, baseAdd);
      const mMs = measureSteadyMs(rt, mb, portBySlot(micro), micro.inputs);
      const bMs = measureSteadyMs(rt, bb, portBySlot(baseAdd), baseAdd.inputs);
      ms = mMs - bMs;
      if (ms < 0) { ms = 0; clamped = true; }
      if (mb.g?.free) mb.g.free();
      if (mb.plan?.free) mb.plan.free();
      if (mb.reg?.free) mb.reg.free();
      if (bb.g?.free) bb.g.free();
      if (bb.plan?.free) bb.plan.free();
      if (bb.reg?.free) bb.reg.free();
    } else {
      const S2 = st.in_shapes[1] ?? S;
      const inLayout1 = resolvePortLayout(slotLayouts.get(stepDef.slots[1]), lc, true);
      const micro = {
        numSlots: 3,
        layers: [layerDef],
        steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
        outputSlot: 2,
        ports: [
          { slot: 0, role: 'observation', shape: [...S], layout: inLayout0 },
          { slot: 1, role: 'observation', shape: [...S2], layout: inLayout1 },
        ],
        weights: {},
        inputs: { 0: fill(rng, prod(S)), 1: fill(rng, prod(S2)) },
      };
      const mb = buildGraph(rt, micro);
      ms = measureSteadyMs(rt, mb, portBySlot(micro), micro.inputs);
      if (mb.g?.free) mb.g.free();
      if (mb.plan?.free) mb.plan.free();
      if (mb.reg?.free) mb.reg.free();
    }
    perOp.push({ ...base, ms_median: +ms.toFixed(4), ...(clamped ? { clamped_zero: true } : {}) });
  });

  // Whole-graph measured total (steady median).
  const prof = profileSpec(rt, spec, { iterations });
  const measured = prof.steady.run_ms_median;
  const predicted = perOp.reduce((a, r) => a + (r.ms_median ?? 0), 0);
  const errPct = measured > 0 ? Math.abs(predicted - measured) / measured * 100 : null;
  const ranked = [...perOp].sort((a, b) => (b.ms_median ?? -1) - (a.ms_median ?? -1));
  const total = predicted > 0 ? predicted : null;
  for (const r of ranked) r.share_pct = r.ms_median == null || !total ? null : +(r.ms_median / total * 100).toFixed(1);

  if (tableFile) fs.writeFileSync(tableFile, JSON.stringify(ranked, null, 1));
  printJsonAndExit({
    ok: true,
    iterations,
    per_op: ranked,
    predicted_total_ms: +predicted.toFixed(4),
    measured_total_ms: measured,
    prediction_error_pct: errPct == null ? null : +errPct.toFixed(1),
    hotspot: ranked[0]?.ms_median != null ? { step: ranked[0].step, layer: ranked[0].layer, share_pct: ranked[0].share_pct } : null,
    warnings,
    ...(tableFile ? { table_file: tableFile } : {}),
    notes: [
      'per-op costs are isolated microbenchmarks (steady median); predicted_total = Σ per-op',
      'prediction_error_pct measures model accuracy against the whole-graph run — the planning signal, not gospel',
    ],
  }, 0);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
