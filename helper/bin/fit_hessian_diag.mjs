#!/usr/bin/env node
// burn-math fit_hessian_diag: Hessian DIAGONAL of a spec by second-order
// central differences over forward runs. No backward pass, no engine change.
//
// Usage: node fit_hessian_diag.mjs <spec.json> [slot=0] [h=1e-2] [tolAbs=0.05] [tolRel=0.1]
//
//   spec.json : graph spec with continuous-valued inputs.
//   slot      : input slot holding the parameter vector (default "0").
//   h         : coarse step; the fine step is h/2 (trust signal).
//   tolAbs    : absolute tolerance for cross-step agreement and for
//               "zero within noise" (defaults calibrated on the fit battery:
//               f32 second differences carry ~1e-2 noise at h=1e-2).
//   tolRel    : relative tolerance for cross-step agreement.
//
// Output, per output element o and input element i:
//   H_ii^(o) = (f_o(x + h e_i) - 2 f_o(x) + f_o(x - h e_i)) / h^2
// measured at h and h/2, with a trust status per entry (see lib/fit.mjs):
// trusted | zero_within_noise | untrusted.
//
// Cost (known before running): 4n + 1 forward runs for n inputs in the slot.
// Exit 0 when the measurement succeeded (untrusted entries are findings,
// not failures); exit 2 on bad input or a crashed run.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import { hessianDiagSecondOrder, hessianTrust } from '../lib/fit.mjs';

const [specFile, slotArg, hArg, tolAbsArg, tolRelArg] = process.argv.slice(2);
if (!specFile) {
  console.error('usage: node fit_hessian_diag.mjs <spec.json> [slot=0] [h=1e-2] [tolAbs=0.05] [tolRel=0.1]');
  process.exit(2);
}
const slot = slotArg !== undefined ? slotArg : '0';
const h = hArg !== undefined ? Number(hArg) : 1e-2;
const tolAbs = tolAbsArg !== undefined ? Number(tolAbsArg) : 0.05;
const tolRel = tolRelArg !== undefined ? Number(tolRelArg) : 0.1;
if (!(h > 0) || !(tolAbs >= 0) || !(tolRel >= 0)) {
  printJsonAndExit({ ok: false, error: 'h must be positive; tolAbs/tolRel must be non-negative' }, 2);
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
  if (spec.inputs[slot] === undefined) {
    throw new Error(`spec has no input slot ${slot} (has: ${Object.keys(spec.inputs).join(', ')})`);
  }
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);

  const baseOut = runOnce(g, reg, rt, plan, ports, spec.inputs);
  const outLen = baseOut.length;
  const inVals = spec.inputs[slot];
  const inLen = inVals.length;
  const hFine = h / 2;

  // diagonal[out][in] = trust record at the fine step.
  const diagonal = Array.from({ length: outLen }, () => new Array(inLen).fill(null));
  let trusted = 0;
  let zeroNoise = 0;
  let untrusted = 0;
  for (let j = 0; j < inLen; j++) {
    for (const [hh, tag] of [[h, 'coarse'], [hFine, 'fine']]) {
      const plus = JSON.parse(JSON.stringify(spec.inputs));
      const minus = JSON.parse(JSON.stringify(spec.inputs));
      plus[slot][j] += hh;
      minus[slot][j] -= hh;
      const fp = runOnce(g, reg, rt, plan, ports, plus);
      const fm = runOnce(g, reg, rt, plan, ports, minus);
      for (let o = 0; o < outLen; o++) {
        const est = hessianDiagSecondOrder(fp[o], baseOut[o], fm[o], hh);
        if (tag === 'coarse') diagonal[o][j] = { coarse: est };
        else diagonal[o][j] = { ...diagonal[o][j], fine: est };
      }
    }
    for (let o = 0; o < outLen; o++) {
      const t = hessianTrust(diagonal[o][j].coarse, diagonal[o][j].fine, tolAbs, tolRel);
      diagonal[o][j] = t;
      if (t.status === 'trusted') trusted++;
      else if (t.status === 'zero_within_noise') zeroNoise++;
      else untrusted++;
    }
  }
  if (g.free) g.free();

  printJsonAndExit({
    ok: true,
    slot,
    h_coarse: h,
    h_fine: hFine,
    tol_abs: tolAbs,
    tol_rel: tolRel,
    output_f32_count: outLen,
    input_count: inLen,
    forward_runs: 4 * inLen + 1,
    diagonal,
    trust_counts: { trusted, zero_within_noise: zeroNoise, untrusted },
    notes: [
      'diagonal only: cross terms d2f/dx_i dx_j (i != j) are not measured',
      'second differences on f32 carry cancellation noise; trust the status, not just the value',
      'linear and piecewise-linear specs (linear, relu away from the kink) have true diagonal 0',
    ],
  }, 0);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
