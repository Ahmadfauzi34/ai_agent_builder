#!/usr/bin/env node
// burn-math gradcheck: numeric gradient (finite-difference) Jacobian of a spec,
// optionally verified against an agent-supplied analytic Jacobian.
//
// Usage: node gradcheck.mjs <spec.json> [analytic.json] [eps=1e-3] [tol=1e-3]
//
//   spec.json      : graph spec with inputs (continuous-valued ports).
//   analytic.json  : optional {"<inputSlot>": [[dOut0/dIn0, ...], ...]}
//                    rows = output elements, cols = input elements of that slot.
//   eps            : central-difference step.
//   tol            : acceptance tolerance (applies to abs and rel error).
//
// Without analytic.json the tool prints the numeric Jacobian as a sensitivity
// map ("how much does each output move when each input moves") — useful on its
// own for sensitivity analysis. With analytic.json it verifies the agent's
// claimed derivatives (e.g. a hand-derived backward pass) and reports
// PASS/FAIL with max abs/rel error per input slot.
//
// Prints {ok, eps, tol, jacobians, compared, passed, ...}.
// Exit 0 iff no analytic given, or analytic matches within tol; 2 on mismatch.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import fs from 'node:fs';

const [specFile, analyticFile, epsArg, tolArg] = process.argv.slice(2);
if (!specFile) {
  console.error('usage: node gradcheck.mjs <spec.json> [analytic.json] [eps=1e-3] [tol=1e-3]');
  process.exit(2);
}
const eps = epsArg !== undefined ? Number(epsArg) : 1e-3;
const tol = tolArg !== undefined ? Number(tolArg) : 1e-3;
if (!(eps > 0) || !(tol > 0)) {
  printJsonAndExit({ ok: false, error: 'eps and tol must be positive numbers' }, 2);
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
  const analytic = analyticFile ? JSON.parse(fs.readFileSync(analyticFile, 'utf8')) : null;
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);

  const baseOut = runOnce(g, reg, rt, plan, ports, spec.inputs);
  const outLen = baseOut.length;

  const jacobians = {};
  const perSlot = {};
  let compared = false;
  let passed = true;
  let maxAbs = 0;
  let maxRel = 0;

  for (const [slotStr, values] of Object.entries(spec.inputs)) {
    const inLen = values.length;
    // J[row=output][col=input] via central differences.
    const J = Array.from({ length: outLen }, () => new Array(inLen).fill(0));
    for (let j = 0; j < inLen; j++) {
      const plus = JSON.parse(JSON.stringify(spec.inputs));
      const minus = JSON.parse(JSON.stringify(spec.inputs));
      plus[slotStr][j] += eps;
      minus[slotStr][j] -= eps;
      const fp = runOnce(g, reg, rt, plan, ports, plus);
      const fm = runOnce(g, reg, rt, plan, ports, minus);
      for (let i = 0; i < outLen; i++) J[i][j] = (fp[i] - fm[i]) / (2 * eps);
    }
    jacobians[slotStr] = { rows: outLen, cols: inLen, values: J };

    if (analytic && analytic[slotStr] !== undefined) {
      compared = true;
      const A = analytic[slotStr];
      if (!Array.isArray(A) || A.length !== outLen || A.some((r) => !Array.isArray(r) || r.length !== inLen)) {
        throw new Error(`analytic[${slotStr}] must be ${outLen}x${inLen} (rows=outputs, cols=inputs)`);
      }
      let sAbs = 0, sRel = 0;
      for (let i = 0; i < outLen; i++) {
        for (let j = 0; j < inLen; j++) {
          const a = Math.abs(J[i][j] - A[i][j]);
          const denom = Math.max(Math.abs(A[i][j]), 1e-12);
          const r = a / denom;
          if (a > sAbs) sAbs = a;
          if (r > sRel) sRel = r;
        }
      }
      const slotPass = sAbs <= tol || sRel <= tol;
      perSlot[slotStr] = { max_abs_error: sAbs, max_rel_error: sRel, passed: slotPass };
      if (sAbs > maxAbs) maxAbs = sAbs;
      if (sRel > maxRel) maxRel = sRel;
      if (!slotPass) passed = false;
    }
  }
  if (g.free) g.free();

  printJsonAndExit({
    ok: true,
    eps,
    tol,
    output_f32_count: outLen,
    compared,
    passed: compared ? passed : true,
    max_abs_error: compared ? maxAbs : null,
    max_rel_error: compared ? maxRel : null,
    per_input_slot: perSlot,
    jacobians,
    notes: [
      'central differences; non-smooth ops (relu at the kink) yield a subgradient average',
      'only meaningful for continuous-valued inputs; discrete index inputs (e.g. embedding ids) are not differentiable',
    ],
  }, compared && !passed ? 2 : 0);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
