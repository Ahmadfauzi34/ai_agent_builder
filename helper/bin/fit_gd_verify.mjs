#!/usr/bin/env node
// burn-math fit_gd_verify: verify a CLAIMED gradient-descent trajectory
// over an input port slot (`"slot": "0"`), a single layer's weights (`"layer": 0` or `--layer 0`),
// OR joint multi-layer weights (`"layers": [0, 1]` or `--layer all`), with optional
// `--targets <targets.json>` (or embedded `trajectory.targets`) for supervised MSE loss.
//
// Usage: node fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3] [--layer <idx|all>] [--targets <targets.json>]
//
// A trajectory is a list of steps an agent (or an optimizer) claims to have
// taken: weights, loss, and optionally the gradient and learning rate used.
// The weights vector fills the target: one input slot, one layer's weights,
// or joint multi-layer weights (see the header above). This tool does not
// trust the claims: it recomputes the loss at each step's weights with a
// forward run of the spec (rebuilding the graph per step for layer targets),
// then checks:
//   (1) recomputation — recomputed loss matches claimed loss within tol.
//       A mismatch is a HARD failure (exit 2): the trajectory does not
//       describe this spec.
//   (2) descent — did the recomputed loss decrease vs the previous step?
//       An ascent is REPORTED, not fatal: GD with too large a step
//       legitimately ascends, and that is a finding about the trajectory.
//   (3) first-order prediction — when the previous step's grad and lr are
//       supplied, the actual decrease is compared to lr * |grad|^2 of the
//       step that was taken (departure gradient). Ratio outside [0, 2] is
//       flagged as prediction_off (a heuristic flag, documented in
//       lib/fit.mjs — not a verdict).
//   (4) gradient norm trend — first vs last supplied norm.
//
//   trajectory.json : {"slot"|"layer"|"layers"?, "targets"?,
//                      "steps": [{"weights": [...],
//                      "loss": <number>, "grad"?: [...], "lr"?: <number>}]}
//   tol             : acceptance tolerance for recomputation (abs and rel).
//
// Loss: a single-element output IS the loss; a vector output is the sum of
// squares, or MSE against targets when --targets / trajectory.targets is
// present. If your GD optimizes a different loss, express it as a spec
// whose output IS that loss.
//
// Exit 0 iff every recomputation matched (ascents/prediction flags do not
// change the exit code — read ascent_count and prediction_off_count);
// exit 2 on bad format, a recomputation mismatch, or a crashed run.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { discoverTrainableLayout } from '../lib/recurse.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';
import { scalarize, verifyGdSteps } from '../lib/fit.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3] [--layer <idx|all>] [--targets <targets.json>]');
  process.exit(2);
}

let cliLayer = null;
let cliTargetsFile = null;
const posArgs = [];
for (let i = 0; i < rawArgs.length; i++) {
  if (rawArgs[i] === '--layer' && i + 1 < rawArgs.length) {
    cliLayer = String(rawArgs[++i]);
  } else if (rawArgs[i] === '--targets' && i + 1 < rawArgs.length) {
    cliTargetsFile = String(rawArgs[++i]);
  } else {
    posArgs.push(rawArgs[i]);
  }
}

const [specFile, trajFile, tolArg] = posArgs;
if (!specFile || !trajFile) {
  console.error('usage: node fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3] [--layer <idx|all>] [--targets <targets.json>]');
  process.exit(2);
}
const tol = tolArg !== undefined ? Number(tolArg) : 1e-3;
if (!(tol > 0)) {
  printJsonAndExit({ ok: false, error: 'tol must be positive' }, 2);
}

function runOnce(g, reg, rt, plan, ports, inputs) {
  const bundle = makeBundle(rt, plan, inputs, ports);
  const out = g.run(reg, bundle);
  const vals = Array.from(out.to_array());
  out.free();
  if (bundle.free) bundle.free();
  return vals;
}

function evalGraphObjective(g, reg, rt, plan, ports, defaultInputs, targets) {
  if (targets && Array.isArray(targets.cases) && targets.cases.length > 0) {
    let mse = 0;
    for (const c of targets.cases) {
      const vals = runOnce(g, reg, rt, plan, ports, c.inputs);
      if (vals.length !== c.expected.length) {
        throw new Error(`case output length ${vals.length} != expected ${c.expected.length}`);
      }
      for (let i = 0; i < vals.length; i++) {
        const d = vals[i] - c.expected[i];
        mse += d * d;
      }
    }
    return mse;
  }
  return scalarize(runOnce(g, reg, rt, plan, ports, defaultInputs));
}

try {
  const spec = loadSpec(specFile);
  const traj = JSON.parse(fs.readFileSync(trajFile, 'utf8'));
  if (!traj || !Array.isArray(traj.steps) || traj.steps.length === 0) {
    throw new Error('trajectory must be {"steps": [...]} with at least one step');
  }
  const targets = cliTargetsFile
    ? JSON.parse(fs.readFileSync(cliTargetsFile, 'utf8'))
    : (traj.targets ?? null);

  const isMultiLayer = cliLayer === 'all' || Array.isArray(traj.layers);
  const targetLayer = !isMultiLayer
    ? (cliLayer !== null ? cliLayer : (traj.layer !== undefined ? String(traj.layer) : null))
    : null;
  const slot = (!isMultiLayer && targetLayer === null)
    ? (traj.slot !== undefined ? String(traj.slot) : '0')
    : null;

  let expectedLen;
  let multiLayerSlices = null;
  if (isMultiLayer) {
    const layout = discoverTrainableLayout(spec);
    const layerIds = Array.isArray(traj.layers)
      ? traj.layers.map(Number)
      : layout.bindings.map((b) => b.layer_index);
    let off = 0;
    multiLayerSlices = layerIds.map((li) => {
      const w = spec.weights?.[String(li)];
      if (!Array.isArray(w)) throw new Error(`spec has no weights for layer ${li}`);
      const sl = { layer_index: li, offset: off, length: w.length };
      off += w.length;
      return sl;
    });
    expectedLen = off;
  } else if (targetLayer !== null) {
    if (!spec.weights || spec.weights[targetLayer] === undefined) {
      throw new Error(`spec has no weights for layer ${targetLayer} (has: ${Object.keys(spec.weights ?? {}).join(', ')})`);
    }
    expectedLen = spec.weights[targetLayer].length;
  } else {
    if (spec.inputs[slot] === undefined) {
      throw new Error(`spec has no input slot ${slot} (has: ${Object.keys(spec.inputs).join(', ')})`);
    }
    expectedLen = spec.inputs[slot].length;
  }

  for (let s = 0; s < traj.steps.length; s++) {
    const st = traj.steps[s];
    if (!Array.isArray(st.weights) || st.weights.length !== expectedLen) {
      throw new Error(`step ${s}: weights must be an array of length ${expectedLen}`);
    }
    if (typeof st.loss !== 'number' || !Number.isFinite(st.loss)) {
      throw new Error(`step ${s}: loss must be a finite number`);
    }
    if (st.grad !== undefined && (!Array.isArray(st.grad) || st.grad.length !== expectedLen)) {
      throw new Error(`step ${s}: grad must be an array of length ${expectedLen} when present`);
    }
    if (st.lr !== undefined && !(st.lr > 0)) {
      throw new Error(`step ${s}: lr must be positive when present`);
    }
  }

  const rt = await loadWasm();
  const ports = portBySlot(spec);
  const recomputed = [];

  if (isMultiLayer) {
    for (const st of traj.steps) {
      const s2 = JSON.parse(JSON.stringify(spec));
      for (const sl of multiLayerSlices) {
        s2.weights[String(sl.layer_index)] = st.weights.slice(sl.offset, sl.offset + sl.length);
      }
      const { reg, plan, g } = buildGraph(rt, s2);
      try {
        recomputed.push(evalGraphObjective(g, reg, rt, plan, ports, s2.inputs, targets));
      } finally {
        if (g.free) g.free();
        if (plan.free) plan.free();
        if (reg.free) reg.free();
      }
    }
  } else if (targetLayer !== null) {
    for (const st of traj.steps) {
      const s2 = JSON.parse(JSON.stringify(spec));
      s2.weights[targetLayer] = st.weights;
      const { reg, plan, g } = buildGraph(rt, s2);
      try {
        recomputed.push(evalGraphObjective(g, reg, rt, plan, ports, s2.inputs, targets));
      } finally {
        if (g.free) g.free();
        if (plan.free) plan.free();
        if (reg.free) reg.free();
      }
    }
  } else {
    const { reg, plan, g } = buildGraph(rt, spec);
    try {
      for (const st of traj.steps) {
        const inputs = JSON.parse(JSON.stringify(spec.inputs));
        inputs[slot] = st.weights;
        recomputed.push(evalGraphObjective(g, reg, rt, plan, ports, inputs, targets));
      }
    } finally {
      if (g.free) g.free();
      if (plan.free) plan.free();
      if (reg.free) reg.free();
    }
  }

  const verdict = verifyGdSteps(traj.steps, recomputed, tol);
  printJsonAndExit({
    ok: verdict.ok,
    ...(isMultiLayer
      ? { layers: multiLayerSlices.map((s) => s.layer_index), target_kind: 'multi_layer_weights' }
      : (targetLayer !== null ? { layer: Number(targetLayer), target_kind: 'layer_weights' } : { slot })),
    tol,
    steps: traj.steps.length,
    forward_runs: traj.steps.length * (targets?.cases?.length ?? 1),
    mismatch_count: verdict.mismatch_count,
    ascent_count: verdict.ascent_count,
    prediction_off_count: verdict.prediction_off_count,
    grad_norm_first: verdict.grad_norm_first,
    grad_norm_last: verdict.grad_norm_last,
    per_step: verdict.per_step,
    notes: [
      'loss scalarization: single-element output is the loss; vector output is sum of squares (or MSE against targets when present)',
      'ascent is a finding about the trajectory, not a tool failure — see ascent_count',
      'prediction ratio outside [0, 2] is a heuristic flag, not a verdict',
    ],
  }, verdict.ok ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
