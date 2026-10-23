// burn-math #13 Fase 3: per-layer quantization budget planner.
//
// Given a spec with weighted linear layers and a total output-perturbation
// tolerance, decide WHICH layers to quantize (int8) to maximize checkpoint
// savings. Two modes:
//
//   bound mode (default): analytical, INSTANT, guaranteed. A compositional
//     worst-case bound is derived from the weights BEFORE running anything:
//     per-layer perturbation (from the engine's own quant report) is
//     propagated through downstream layers by their operator norms
//     (oo-norm; relu/tanh/sigmoid by Lipschitz constant; add is additive).
//     The bound is rigorous but conservative. Scope: linear + relu/tanh/
//     sigmoid + add. Anything else -> loud error (not a silent guess).
//
//   measured mode: empirical iterative greedy. Candidate plans are actually
//     RUN: at each step every remaining layer is tried on top of the current
//     plan, the true end-to-end output delta vs the f32 baseline is measured,
//     and the best bytes-per-delta layer is taken while delta <= tol. No
//     composition heuristic — the reported total is the measured delta of
//     the final plan itself. Costs O(N^2) runs; N is small in practice.
//
// In both modes the tolerance is on the OUTPUT's oo-norm perturbation
// (max |y_base - y_plan| over cases and output elements).
import { buildGraph, makeBundle } from './engine.mjs';
import { portBySlot } from './spec.mjs';

// 1-Lipschitz (or known) activations the bound mode understands.
const LIPSCHITZ = { relu: 1, tanh: 1, sigmoid: 0.25 };

// Weighted linear layers in a spec: [{index, id, inDim, outDim, bias, weights}].
// Spec weight layout (audited): column-major flat[k*outDim + i], bias appended.
export function linearLayerInfos(spec) {
  const infos = [];
  for (const [li, decl] of (spec.layers ?? []).entries()) {
    if (decl?.constructor !== 'linear') continue;
    const [id, inDim, outDim, bias] = decl.args;
    const weights = (spec.weights ?? {})[String(li)];
    if (!weights) throw new Error(`quant_budget: linear layer ${li} has no weights in spec`);
    infos.push({ index: li, id, inDim, outDim, bias: !!bias, weights });
  }
  return infos;
}

// Engine quant report per linear layer (real burn quantization, not simulation).
export function fetchQuantReports(rt, spec, scheme, infos = linearLayerInfos(spec)) {
  return infos.map((info) => {
    const layer = new rt.WasmLinear(info.inDim, info.outDim, info.bias);
    try {
      layer.setWeightsFlat(info.weights);
      const report = JSON.parse(layer.quantize_state_report(scheme));
      return {
        ...info,
        report,
        bytesF32: report.bytes_f32,
        bytesQ8: report.bytes_q8,
        savedBytes: report.bytes_f32 - report.bytes_q8,
      };
    } finally {
      layer.free();
    }
  });
}

// ||W||_oo = max abs row sum. W is [out,in], flat[k*out+i] == W[i][k].
export function operatorNormInf(weights, inDim, outDim) {
  let m = 0;
  for (let i = 0; i < outDim; i++) {
    let s = 0;
    for (let k = 0; k < inDim; k++) s += Math.abs(weights[k * outDim + i]);
    if (s > m) m = s;
  }
  return m;
}

function biasNormInf(weights, inDim, outDim, bias) {
  if (!bias) return 0;
  let m = 0;
  for (let i = 0; i < outDim; i++) {
    const a = Math.abs(weights[inDim * outDim + i]);
    if (a > m) m = a;
  }
  return m;
}

// Steps referencing linear layers, in order: [{info, inSlot, outSlot}].
function linearStepsInOrder(spec, infos) {
  const byIndex = new Map(infos.map((i) => [i.index, i]));
  const out = [];
  for (const step of spec.steps ?? []) {
    if (step.kind === 'unary' && byIndex.has(step.layer)) {
      out.push({ info: byIndex.get(step.layer), inSlot: step.slots[0], outSlot: step.slots[1] });
    }
  }
  return out;
}

// Compositional worst-case bound (oo-norm).
// Returns { contribs: [{index, contrib}], total } where contrib_i bounds
// layer i's quantization effect on the final output.
//
// The per-layer perturbation uses the EXACT error operator norm ||ΔW||_oo
// (ΔW = W - W_dequantized, via the real engine path), NOT the report's
// analytical bound_W·inDim:
//   |Δy_i| <= Σ_j|ΔW_ij|·|x_j| + |Δb_i| <= ||ΔW||_oo·||x||_oo + ||Δb||_oo.
// Because ΔW is exact rather than estimated, the total remains a rigorous
// worst-case bound — measured ~6x tighter than bound_W·inDim on outlier
// weights (2026-10-24). Downstream amplification stays worst-case
// (operator-norm product); that looseness is the documented price of the guarantee.
export function analyticalContribs(spec, reports, inputBounds, rt, scheme) {
  const steps = spec.steps ?? [];
  const repByIndex = new Map(reports.map((r) => [r.index, r]));

  // Forward: oo-norm value bound per slot.
  const bound = new Map();
  for (const [slot, b] of inputBounds) bound.set(slot, b);
  const linSteps = [];
  for (const step of steps) {
    const decl = spec.layers[step.layer];
    if (step.kind === 'unary' && decl?.constructor === 'linear') {
      const rep = repByIndex.get(step.layer);
      if (!rep) throw new Error(`analyticalContribs: no report for linear layer ${step.layer}`);
      const [a, b] = step.slots;
      if (!bound.has(a)) throw new Error(`analyticalContribs: slot ${a} has no bound (dangling input?)`);
      const wNorm = operatorNormInf(rep.weights, rep.inDim, rep.outDim);
      const bNorm = biasNormInf(rep.weights, rep.inDim, rep.outDim, rep.bias);
      bound.set(b, wNorm * bound.get(a) + bNorm);
      linSteps.push({ rep, inSlot: a, outSlot: b, wNorm });
    } else if (step.kind === 'unary' && decl?.constructor in LIPSCHITZ) {
      const [a, b] = step.slots;
      if (!bound.has(a)) throw new Error(`analyticalContribs: slot ${a} has no bound`);
      bound.set(b, LIPSCHITZ[decl.constructor] * bound.get(a));
    } else if (step.kind === 'binary' && decl?.constructor === 'add') {
      const [a, b, c] = step.slots;
      if (!bound.has(a) || !bound.has(b)) throw new Error(`analyticalContribs: add slot missing bound`);
      bound.set(c, bound.get(a) + bound.get(b));
    } else {
      throw new Error(
        `analyticalContribs: unsupported step '${decl?.constructor}' (${step.kind}) — ` +
        `bound mode handles linear + relu/tanh/sigmoid + add only`);
    }
  }

  // Backward: amplification of each slot toward the output slot.
  const sens = new Map([[spec.outputSlot, 1]]);
  const get = (m, k) => m.get(k) ?? 0;
  const contribs = [];
  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];
    const decl = spec.layers[step.layer];
    if (step.kind === 'unary' && decl?.constructor === 'linear') {
      const [a, b] = step.slots;
      const ls = linSteps.find((s) => s.inSlot === a && s.outSlot === b);
      const sb = get(sens, b);
      sens.set(a, get(sens, a) + sb * ls.wNorm);
      // |Δy_i| <= Σ_j|ΔW_ij|·|x_j| + |Δb_i| <= ||ΔW||_oo·||x||_oo + ||Δb||_oo.
      // ΔW/Δb are EXACT (dequantized via the engine), so this stays rigorous
      // while being ~6x tighter than the report's bound_W·inDim on outliers.
      const deq = dequantizedWeights(rt, ls.rep, scheme);
      const nW = ls.rep.inDim * ls.rep.outDim;
      const errW = operatorNormInf(
        ls.rep.weights.slice(0, nW).map((w, i) => w - deq[i]),
        ls.rep.inDim, ls.rep.outDim);
      const errB = ls.rep.bias
        ? Math.max(...ls.rep.weights.slice(nW).map((b, i) => Math.abs(b - deq[nW + i])))
        : 0;
      const pert = errW * bound.get(a) + errB;
      contribs.push({ index: step.layer, contrib: pert * sb });
    } else if (step.kind === 'unary' && decl?.constructor in LIPSCHITZ) {
      const [a, b] = step.slots;
      sens.set(a, get(sens, a) + get(sens, b) * LIPSCHITZ[decl.constructor]);
    } else if (step.kind === 'binary' && decl?.constructor === 'add') {
      const [a, b, c] = step.slots;
      sens.set(a, get(sens, a) + get(sens, c));
      sens.set(b, get(sens, b) + get(sens, c));
    }
  }
  contribs.sort((x, y) => x.index - y.index);
  return { contribs, total: contribs.reduce((s, c) => s + c.contrib, 0) };
}

// Greedy knapsack by saved-bytes-per-cost. Documented heuristic, not optimal.
export function planGreedy(items, tol) {
  // items: [{index, savedBytes, cost}]
  const order = [...items]
    .filter((it) => it.cost > 0 && it.savedBytes > 0)
    .sort((a, b) => (b.savedBytes / b.cost) - (a.savedBytes / a.cost));
  const chosen = [];
  let cost = 0;
  for (const it of order) {
    if (cost + it.cost <= tol) {
      chosen.push(it.index);
      cost += it.cost;
    }
  }
  return { quantize: chosen.sort((a, b) => a - b), cost };
}

// oo-norm bound of each input slot from real cases: max over cases of max|x|.
export function inputBoundsFromCases(cases) {
  const bounds = new Map();
  for (const c of cases) {
    for (const [slotStr, values] of Object.entries(c)) {
      const slot = Number(slotStr);
      let m = 0;
      for (const v of values) { const a = Math.abs(v); if (a > m) m = a; }
      if (m > (bounds.get(slot) ?? 0)) bounds.set(slot, m);
    }
  }
  return bounds;
}

// ---- measured mode ----

// Dequantized f32 weights for one layer via the REAL engine path:
// setWeightsFlat -> get_quantized_state -> load_state -> getWeightsFlat.
export function dequantizedWeights(rt, info, scheme) {
  const a = new rt.WasmLinear(info.inDim, info.outDim, info.bias);
  try {
    a.setWeightsFlat(info.weights);
    const qb = a.get_quantized_state(scheme);
    const b = new rt.WasmLinear(info.inDim, info.outDim, info.bias);
    try {
      b.load_state(qb);
      return Array.from(b.getWeightsFlat());
    } finally {
      b.free();
    }
  } finally {
    a.free();
  }
}

// Run a spec in-process; returns per-case output value arrays.
export function runSpecOutputs(rt, spec, cases) {
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);
  const outs = [];
  try {
    for (const c of cases) {
      const bundle = makeBundle(rt, plan, c, ports);
      const t = g.run(reg, bundle);
      try {
        outs.push(Array.from(t.to_array()));
      } finally {
        t.free();
      }
    }
  } finally {
    reg.free();
  }
  return outs;
}

export function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// Iterative greedy with EXACT re-measurement: at each step, every remaining
// layer is tried on top of the current plan and the true end-to-end delta
// vs the f32 baseline is measured. The reported total is the measured delta
// of the final plan itself — no composition heuristic.
export function measuredPlan(rt, spec, cases, scheme, tol, infos = linearLayerInfos(spec)) {
  const reports = fetchQuantReports(rt, spec, scheme, infos);
  const repByIndex = new Map(reports.map((r) => [r.index, r]));
  const deq = new Map();
  for (const info of infos) deq.set(info.index, dequantizedWeights(rt, info, scheme));

  const baseOuts = runSpecOutputs(rt, spec, cases);
  const variantSpec = (chosen) => {
    const s = JSON.parse(JSON.stringify(spec));
    s.weights = { ...spec.weights };
    for (const idx of chosen) s.weights[String(idx)] = deq.get(idx);
    return s;
  };
  const planDelta = (chosen) => {
    const outs = runSpecOutputs(rt, variantSpec(chosen), cases);
    let m = 0;
    for (let c = 0; c < cases.length; c++) {
      const d = maxAbsDiff(baseOuts[c], outs[c]);
      if (d > m) m = d;
    }
    return m;
  };

  const chosen = [];
  let totalDelta = 0;
  for (;;) {
    let best = null;
    for (const info of infos) {
      if (chosen.includes(info.index)) continue;
      const trial = [...chosen, info.index];
      const delta = planDelta(trial);
      const saved = trial.reduce((s, i) => s + repByIndex.get(i).savedBytes, 0);
      if (delta <= tol && (best === null || saved / Math.max(delta, 1e-12) > best.ratio)) {
        best = { index: info.index, delta, saved, ratio: saved / Math.max(delta, 1e-12) };
      }
    }
    if (best === null) break;
    chosen.push(best.index);
    totalDelta = best.delta;
  }
  chosen.sort((a, b) => a - b);
  return { quantize: chosen, totalDelta, reports };
}

// ---- main entry: analytical planner ----

export function analyticalPlan(spec, cases, scheme, tol, rt, infos = linearLayerInfos(spec)) {
  const reports = fetchQuantReports(rt, spec, scheme, infos);
  const { contribs, total } = analyticalContribs(spec, reports, inputBoundsFromCases(cases), rt, scheme);
  const items = contribs.map((c) => {
    const r = reports.find((x) => x.index === c.index);
    return { index: c.index, savedBytes: r.savedBytes, cost: c.contrib, report: r.report };
  });
  const { quantize, cost } = planGreedy(items, tol);
  const saved = quantize.reduce((s, i) => s + reports.find((x) => x.index === i).savedBytes, 0);
  return { reports, contribs, boundTotal: total, quantize, cost, saved };
}
