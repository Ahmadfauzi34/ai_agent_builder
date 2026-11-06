#!/usr/bin/env node
// burn-math gradcheck: numeric gradient (finite-difference) Jacobian of a spec
// with respect to input ports AND (with --weights) weighted layers,
// optionally verified against an agent-supplied analytic Jacobian.
//
// Usage: node gradcheck.mjs <spec.json> [analytic.json] [eps=1e-3] [tol=1e-3] [--weights]
//
//   spec.json      : graph spec with inputs (continuous-valued ports).
//   analytic.json  : optional {"<inputSlot>": [[dOut0/dIn0, ...], ...]}
//                    rows = output elements, cols = input elements of that slot.
//                    (also accepts the nested form {"inputs": {...}, "weights": {...}};
//                    with --weights, or analytic.weights present, weight Jacobians
//                    are computed and compared too.)
//   eps            : central-difference step.
//                    (arg2 may be a bare number instead of analytic.json: then it is eps.)
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
import { printJsonAndExit , errText } from '../lib/runner.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node gradcheck.mjs <spec.json> [analytic.json] [eps=1e-3] [tol=1e-3] [--weights]');
  process.exit(2);
}

const includeWeightsFlag = rawArgs.includes('--weights');
const posArgs = rawArgs.filter((a) => a !== '--weights');
const [specFile, arg2, arg3, arg4] = posArgs;

// Disambiguate whether arg2 is analytic.json or numeric eps
const isNumericArg = (s) => s !== undefined && s !== '' && !Number.isNaN(Number(s)) && !String(s).endsWith('.json');
const analyticFile = (arg2 !== undefined && !isNumericArg(arg2)) ? arg2 : null;
const epsArg = analyticFile ? arg3 : arg2;
const tolArg = analyticFile ? arg4 : arg3;

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

function runWithWeights(rt, spec, layerKey, newWeights) {
  const s2 = JSON.parse(JSON.stringify(spec));
  s2.weights[layerKey] = newWeights;
  const { reg, plan, g } = buildGraph(rt, s2);
  try {
    return runOnce(g, reg, rt, plan, portBySlot(s2), s2.inputs);
  } finally {
    if (g.free) g.free();
    if (plan.free) plan.free();
    if (reg.free) reg.free();
  }
}

try {
  const spec = loadSpec(specFile);
  const analytic = analyticFile ? JSON.parse(fs.readFileSync(analyticFile, 'utf8')) : null;
  const includeWeights = includeWeightsFlag || Boolean(analytic?.weights);

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

    const A = analytic?.[slotStr] ?? analytic?.inputs?.[slotStr];
    if (A !== undefined) {
      compared = true;
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
  if (plan.free) plan.free();
  if (reg.free) reg.free();

  const weightJacobians = {};
  const perWeightLayer = {};
  if (includeWeights && spec.weights && typeof spec.weights === 'object') {
    for (const [layerStr, wVals] of Object.entries(spec.weights)) {
      if (!Array.isArray(wVals) || wVals.length === 0) continue;
      const wLen = wVals.length;
      const JW = Array.from({ length: outLen }, () => new Array(wLen).fill(0));
      for (let j = 0; j < wLen; j++) {
        const wPlus = [...wVals]; wPlus[j] += eps;
        const wMinus = [...wVals]; wMinus[j] -= eps;
        const fp = runWithWeights(rt, spec, layerStr, wPlus);
        const fm = runWithWeights(rt, spec, layerStr, wMinus);
        for (let i = 0; i < outLen; i++) JW[i][j] = (fp[i] - fm[i]) / (2 * eps);
      }
      weightJacobians[layerStr] = {
        layer_index: Number(layerStr),
        constructor: spec.layers[Number(layerStr)]?.constructor ?? null,
        rows: outLen,
        cols: wLen,
        values: JW,
      };

      const AW = analytic?.weights?.[layerStr];
      if (AW !== undefined) {
        compared = true;
        if (!Array.isArray(AW) || AW.length !== outLen || AW.some((r) => !Array.isArray(r) || r.length !== wLen)) {
          throw new Error(`analytic.weights[${layerStr}] must be ${outLen}x${wLen} (rows=outputs, cols=weights)`);
        }
        let wAbs = 0, wRel = 0;
        for (let i = 0; i < outLen; i++) {
          for (let j = 0; j < wLen; j++) {
            const a = Math.abs(JW[i][j] - AW[i][j]);
            const denom = Math.max(Math.abs(AW[i][j]), 1e-12);
            const r = a / denom;
            if (a > wAbs) wAbs = a;
            if (r > wRel) wRel = r;
          }
        }
        const wPass = wAbs <= tol || wRel <= tol;
        perWeightLayer[layerStr] = { max_abs_error: wAbs, max_rel_error: wRel, passed: wPass };
        if (wAbs > maxAbs) maxAbs = wAbs;
        if (wRel > maxRel) maxRel = wRel;
        if (!wPass) passed = false;
      }
    }
  }

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
    ...(includeWeights ? {
      per_weight_layer: perWeightLayer,
      weight_jacobians: weightJacobians,
    } : {}),
    notes: [
      'central differences; non-smooth ops (relu at the kink) yield a subgradient average',
      'only meaningful for continuous-valued inputs; discrete index inputs (e.g. embedding ids) are not differentiable',
      ...(includeWeights ? ['weight_jacobians capture total sensitivity dOut/dW across all steps sharing a weight-tied layer'] : []),
    ],
  }, compared && !passed ? 2 : 0);
} catch (e) {
  printJsonAndExit({ ok: false, error: errText(e) }, 2);
}
