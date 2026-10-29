#!/usr/bin/env node
// burn-math fit_gd_verify: verify a CLAIMED gradient-descent trajectory.
//
// A trajectory is a list of steps an agent (or an optimizer) claims to have
// taken: weights, loss, and optionally the gradient and learning rate used.
// This tool does not trust the claims: it recomputes the loss at each step's
// weights with a forward run of the spec, then checks:
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
// Usage: node fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3]
//
//   spec.json       : graph spec; the trajectory's weights fill ONE input
//                     slot (default "0", or the trajectory's top-level
//                     "slot" field). The spec's output is scalarized per
//                     lib/fit.mjs: a single-element output IS the loss,
//                     a vector output is the sum of squares. If your GD
//                     optimizes a different loss, express it as a spec
//                     whose output IS that loss.
//   trajectory.json : {"slot"?: "0", "steps": [{"weights": [...],
//                     "loss": <number>, "grad"?: [...], "lr"?: <number>}]}
//   tol             : acceptance tolerance for recomputation (abs and rel).
//
// Exit 0 iff every recomputation matched (ascents/prediction flags do not
// change the exit code — read ascent_count and prediction_off_count);
// exit 2 on bad format, a recomputation mismatch, or a crashed run.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import { scalarize, verifyGdSteps } from '../lib/fit.mjs';
import fs from 'node:fs';

const [specFile, trajFile, tolArg] = process.argv.slice(2);
if (!specFile || !trajFile) {
  console.error('usage: node fit_gd_verify.mjs <spec.json> <trajectory.json> [tol=1e-3]');
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

try {
  const spec = loadSpec(specFile);
  const traj = JSON.parse(fs.readFileSync(trajFile, 'utf8'));
  if (!traj || !Array.isArray(traj.steps) || traj.steps.length === 0) {
    throw new Error('trajectory must be {"steps": [...]} with at least one step');
  }
  const slot = traj.slot !== undefined ? String(traj.slot) : '0';
  if (spec.inputs[slot] === undefined) {
    throw new Error(`spec has no input slot ${slot} (has: ${Object.keys(spec.inputs).join(', ')})`);
  }
  const inLen = spec.inputs[slot].length;
  for (let s = 0; s < traj.steps.length; s++) {
    const st = traj.steps[s];
    if (!Array.isArray(st.weights) || st.weights.length !== inLen) {
      throw new Error(`step ${s}: weights must be an array of length ${inLen}`);
    }
    if (typeof st.loss !== 'number' || !Number.isFinite(st.loss)) {
      throw new Error(`step ${s}: loss must be a finite number`);
    }
    if (st.grad !== undefined && (!Array.isArray(st.grad) || st.grad.length !== inLen)) {
      throw new Error(`step ${s}: grad must be an array of length ${inLen} when present`);
    }
    if (st.lr !== undefined && !(st.lr > 0)) {
      throw new Error(`step ${s}: lr must be positive when present`);
    }
  }

  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);

  const recomputed = [];
  for (const st of traj.steps) {
    const inputs = JSON.parse(JSON.stringify(spec.inputs));
    inputs[slot] = st.weights;
    recomputed.push(scalarize(runOnce(g, reg, rt, plan, ports, inputs)));
  }
  if (g.free) g.free();

  const verdict = verifyGdSteps(traj.steps, recomputed, tol);
  printJsonAndExit({
    ok: verdict.ok,
    slot,
    tol,
    steps: traj.steps.length,
    forward_runs: traj.steps.length,
    mismatch_count: verdict.mismatch_count,
    ascent_count: verdict.ascent_count,
    prediction_off_count: verdict.prediction_off_count,
    grad_norm_first: verdict.grad_norm_first,
    grad_norm_last: verdict.grad_norm_last,
    per_step: verdict.per_step,
    notes: [
      'loss scalarization: single-element output is the loss; vector output is sum of squares',
      'ascent is a finding about the trajectory, not a tool failure — see ascent_count',
      'prediction ratio outside [0, 2] is a heuristic flag, not a verdict',
    ],
  }, verdict.ok ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
