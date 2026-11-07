// burn-math #13 Fase 3 (v3.9): per-layer quantization budget planner
// with rigorous Lipschitz DAG bound propagation across 39 constructors
// (including binary `mul`, envelope post-quantization operator norms, and
// epsilon/parameter-aware bounds for `featureNorm`/`rmsNorm`/`layerNorm`).
//
// Given a spec with weighted linear layers and a total output-perturbation
// tolerance, decide WHICH layers to quantize (int8) to maximize checkpoint
// savings. Two modes:
//
//   bound mode (default): analytical, INSTANT, guaranteed. A compositional
//     worst-case bound is derived from the weights BEFORE running anything:
//     per-layer perturbation (from the engine's own quant report) is
//     propagated through downstream layers by their operator norms
//     (oo-norm; Lipschitz unary ops by Lipschitz constant; add/sub are
//     additive; mul multiplies the value bounds; concat takes the max).
//     The bound is rigorous but conservative. Scope: the constructors this
//     file's bound mode handles (linear + Lipschitz unary + add/sub/mul/
//     concat + matmul + conv + embedding + the norm constructors);
//     anything else -> loud error (not a silent guess). div, sqrt and
//     swiGlu stay loud errors BY DESIGN: div has no finite bound without
//     a proven denominator floor, sqrt's derivative is unbounded at 0,
//     and swiGlu has no quantizable weights on the spec path (zero op).
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
import { ssaNormalizeSpec } from './autodiff.mjs';

const STATIC_LIPSCHITZ = {
  relu: 1,
  tanh: 1,
  sigmoid: 0.25,
  softplus: 1,
  gelu: 1.13,
  hardSwish: 1.5,
  transpose: 1,
  shiftUp: 1,
  shiftDown: 1,
  shiftLeft: 1,
  shiftRight: 1,
  avgPool1d: 1,
  avgPool2d: 1,
  // Tahap 6 (PR6): mish 1.11 = dense-grid max of |mish'| (1.0885 at x~1.49,
  // grid step 5e-4 over [-30, 30]; tails -> 1 and 0) + margin, same standard
  // as gelu's 1.13. softmax 0.5 = max row sum 2p(1-p) of the Jacobian.
  // logSoftmax 2 = max row sum 2(1-p). maxPool/adaptiveAvgPool 1 (max and
  // average are 1-Lipschitz in oo-norm). embedding 1 (lookup passes the
  // sensitivity through; its input is an observation index slot).
  mish: 1.11,
  softmax: 0.5,
  logSoftmax: 2,
  maxPool1d: 1,
  maxPool2d: 1,
  adaptiveAvgPool2d: 1,
  embedding: 1,
};

const CONV_CONSTRUCTORS = new Set(['conv1d', 'conv2d', 'convTranspose2d']);

const NORM_CONSTRUCTORS = new Set(['featureNorm', 'rmsNorm', 'layerNorm', 'batchNorm', 'groupNorm', 'instanceNorm']);

// [gamma(C), beta(C)] for a norm layer (defaults gamma=1, beta=0 when the
// spec carries no weights for the layer — the engine's module defaults).
function normGammaBeta(decl, spec, layerIdx, numF) {
  const w = spec?.weights?.[String(layerIdx)];
  const gamma = Array.isArray(w) && w.length >= numF ? w.slice(0, numF) : new Array(numF).fill(1);
  const beta = Array.isArray(w) && w.length >= 2 * numF ? w.slice(numF, 2 * numF) : new Array(numF).fill(0);
  return {
    gammaMax: Math.max(0, ...gamma.map(Math.abs)),
    betaMax: Math.max(0, ...beta.map(Math.abs)),
  };
}

function unaryLipschitz(decl, spec = null, layerIdx = -1, inBound = null) {
  const c = decl?.constructor;
  if (!c) return null;
  if (Object.hasOwn(STATIC_LIPSCHITZ, c)) return STATIC_LIPSCHITZ[c];
  if (c === 'leakyRelu') return Math.max(1, Math.abs(decl.args?.[1] ?? 0.01));
  if (c === 'prelu') return Math.max(1, Math.abs(decl.args?.[2] ?? 0.25));
  if (c === 'hardSigmoid') return Math.abs(decl.args?.[1] ?? 0.2);
  if (c === 'glu') {
    // y = a*sigmoid(b) on the split halves: |dy/da| <= 1, |dy/db| <= |a|/4.
    // No finite GLOBAL constant (grows with |a|); rigorous only with the
    // forward value bound inBound (the file's case-derived input bound).
    if (inBound === null || inBound === undefined) return null;
    return 1 + inBound / 4;
  }
  if (c === 'featureNorm') {
    const eps = decl.args?.[1] ?? 1e-12;
    return 2 / Math.sqrt(Math.max(eps, 1e-30));
  }
  if (c === 'rmsNorm' || c === 'layerNorm') {
    const F = decl.args?.[1] ?? 1;
    const eps = decl.args?.[2] ?? 1e-5;
    const { gammaMax } = normGammaBeta(decl, spec, layerIdx, F);
    return (2 * gammaMax) / Math.sqrt(Math.max(eps, 1e-30));
  }
  if (c === 'batchNorm') {
    // Running stats (mean 0, var 1): y = gamma*x/sqrt(1+eps) + beta — a
    // scale+shift, so the Lipschitz constant is exact, not a norm bound.
    const F = decl.args?.[1] ?? 1;
    const eps = decl.args?.[2] ?? 1e-5;
    const { gammaMax } = normGammaBeta(decl, spec, layerIdx, F);
    return gammaMax / Math.sqrt(1 + Math.max(eps, 0));
  }
  if (c === 'groupNorm' || c === 'instanceNorm') {
    // Same centering+scaling pattern as rmsNorm/layerNorm.
    const F = c === 'groupNorm' ? (decl.args?.[2] ?? 1) : (decl.args?.[1] ?? 1);
    const eps = c === 'groupNorm' ? (decl.args?.[3] ?? 1e-5) : (decl.args?.[2] ?? 1e-5);
    const { gammaMax } = normGammaBeta(decl, spec, layerIdx, F);
    return (2 * gammaMax) / Math.sqrt(Math.max(eps, 1e-30));
  }
  if (CONV_CONSTRUCTORS.has(c)) {
    const info = convWeightInfo(decl, spec, layerIdx);
    if (!info) return null;
    return convOpNormInf(info);
  }
  return null;
}

// Conv weight layout (fitted against the engine, Tahap 6): conv1d/conv2d
// row-major [out][in][k...] + bias [out]; convTranspose2d row-major
// [in][out][k...] + bias [out]. Returns null when weights are missing.
function convWeightInfo(decl, spec, layerIdx) {
  const c = decl?.constructor;
  const w = spec?.weights?.[String(layerIdx)];
  if (!Array.isArray(w)) return null;
  if (c === 'conv1d') {
    const [, inCh, outCh, k] = decl.args;
    return { kind: 'conv1d', w, inCh, outCh, kh: k, kw: 1, bias: true };
  }
  if (c === 'conv2d') {
    const [, inCh, outCh, kh, kw] = decl.args;
    return { kind: 'conv2d', w, inCh, outCh, kh, kw, bias: true };
  }
  if (c === 'convTranspose2d') {
    const [, inCh, outCh, kh, kw] = decl.args;
    return { kind: 'convTranspose2d', w, inCh, outCh, kh, kw, bias: true };
  }
  return null;
}

// ||W||_oo for a conv as a linear map: max over output channels of the
// sum of |weights| in that channel's slice (each output position sees
// every kernel tap once; boundary effects only remove terms).
function convOpNormInf(info) {
  const { w, inCh, outCh, kh, kw } = info;
  const taps = inCh * kh * kw;
  let m = 0;
  for (let o = 0; o < outCh; o++) {
    let s = 0;
    if (info.kind === 'convTranspose2d') {
      for (let i = 0; i < inCh; i++)
        for (let t = 0; t < kh * kw; t++) s += Math.abs(w[i * outCh * kh * kw + o * kh * kw + t]);
    } else {
      for (let t = 0; t < taps; t++) s += Math.abs(w[o * taps + t]);
    }
    if (s > m) m = s;
  }
  return m;
}

function convBiasMax(info) {
  const { w, inCh, outCh, kh, kw } = info;
  const nW = inCh * outCh * kh * kw;
  let m = 0;
  for (let o = 0; o < outCh; o++) {
    const a = Math.abs(w[nW + o] ?? 0);
    if (a > m) m = a;
  }
  return m;
}

function convOutputBound(decl, inBound, spec, layerIdx) {
  const info = convWeightInfo(decl, spec, layerIdx);
  if (!info) throw new Error(`analyticalContribs: ${decl?.constructor} layer ${layerIdx} has no weights in spec`);
  return convOpNormInf(info) * inBound + convBiasMax(info);
}

// Forward oo-norm value bound for a unary step whose output bound is not
// L*inBound. inShape is the tracked input shape (needed by logSoftmax).
function unaryOutputBound(decl, inBound, spec, layerIdx, inShape, L) {
  const c = decl?.constructor;
  if (c === 'glu') return inBound; // |a*sigmoid(b)| <= |a| <= inBound
  if (c === 'softmax') return 1; // probabilities
  if (c === 'logSoftmax') {
    // |log p_i| <= 2*inBound + log(n), n = softmax dim size.
    const dim = decl.args?.[1] ?? 1;
    const n = inShape?.[dim];
    if (!n) throw new Error('analyticalContribs: logSoftmax input shape unknown (shape tracking gap)');
    return 2 * inBound + Math.log(n);
  }
  if (c === 'embedding') {
    // Every output element is one weight element.
    const w = spec?.weights?.[String(layerIdx)];
    if (!Array.isArray(w)) throw new Error(`analyticalContribs: embedding layer ${layerIdx} has no weights in spec`);
    return Math.max(0, ...w.map(Math.abs));
  }
  return L * inBound;
}

function normOutputBound(decl, inBound, spec = null, layerIdx = -1) {
  const c = decl?.constructor;
  if (c === 'featureNorm') {
    const eps = decl.args?.[1] ?? 1e-12;
    return Math.min(1, inBound / Math.sqrt(Math.max(eps, 1e-30)));
  }
  if (c === 'batchNorm') {
    const F = decl.args?.[1] ?? 1;
    const eps = decl.args?.[2] ?? 1e-5;
    const { gammaMax, betaMax } = normGammaBeta(decl, spec, layerIdx, F);
    return (gammaMax * inBound) / Math.sqrt(1 + Math.max(eps, 0)) + betaMax;
  }
  if (c === 'groupNorm' || c === 'instanceNorm') {
    // |x - mean| <= 2*inBound and 1/sqrt(var+eps) <= 1/sqrt(eps): no shape
    // needed (the sqrt(F) cap the rmsNorm/layerNorm path uses is unavailable
    // without the group size, so this is looser but rigorous).
    const F = c === 'groupNorm' ? (decl.args?.[2] ?? 1) : (decl.args?.[1] ?? 1);
    const eps = c === 'groupNorm' ? (decl.args?.[3] ?? 1e-5) : (decl.args?.[2] ?? 1e-5);
    const { gammaMax, betaMax } = normGammaBeta(decl, spec, layerIdx, F);
    return (gammaMax * 2 * inBound) / Math.sqrt(Math.max(eps, 1e-30)) + betaMax;
  }
  const F = decl.args?.[1] ?? 1;
  const eps = decl.args?.[2] ?? 1e-5;
  const w = spec?.weights?.[String(layerIdx)];
  const gamma = Array.isArray(w) && w.length >= F ? w.slice(0, F) : [1];
  const gammaMax = Math.max(0, ...gamma.map(Math.abs));
  if (c === 'rmsNorm') {
    return gammaMax * Math.min(Math.sqrt(F), inBound / Math.sqrt(Math.max(eps, 1e-30)));
  }
  const beta = Array.isArray(w) && w.length >= 2 * F ? w.slice(F, 2 * F) : [0];
  const betaMax = Math.max(0, ...beta.map(Math.abs));
  return gammaMax * Math.min(Math.sqrt(F), (2 * inBound) / Math.sqrt(Math.max(eps, 1e-30))) + betaMax;
}

// Output shape of a step, from the tracked input shape(s). Needed by the
// matmul contraction dim (K) and logSoftmax's dim size. Returns null when
// the shape cannot be determined (fail-closed upstream).
function stepOutputShape(decl, step, shapeOf) {
  const ctor = decl?.constructor;
  const inSh = shapeOf.get(step.slots[0]);
  if (!inSh) return null;
  if (step.kind === 'unary' && ctor === 'linear') {
    const [, , outDim] = decl.args;
    return [inSh[0], outDim, inSh[2], inSh[3]];
  }
  if (step.kind === 'unary' && ctor === 'transpose') {
    const [, d0, d1] = decl.args;
    const out = [...inSh];
    [out[d0], out[d1]] = [out[d1], out[d0]];
    return out;
  }
  if (step.kind === 'unary' && ctor === 'glu') {
    const dim = decl.args?.[1] ?? 1;
    const out = [...inSh];
    out[dim] = Math.floor(inSh[dim] / 2);
    return out;
  }
  if (step.kind === 'unary' && ctor === 'embedding') {
    const [, , dModel] = decl.args;
    return [inSh[0], inSh[1], dModel, 1];
  }
  if (step.kind === 'unary' && ctor === 'conv1d') {
    const [, inCh, outCh, k, s0, p0] = decl.args;
    const s = s0 ?? 1, p = p0 ?? 0;
    return [inSh[0], outCh, Math.floor((inSh[2] + 2 * p - k) / s) + 1, 1];
  }
  if (step.kind === 'unary' && (ctor === 'conv2d' || ctor === 'convTranspose2d')) {
    const [, , outCh, kh, kw, sh0, sw0, ph0, pw0] = decl.args;
    const sH = sh0 ?? 1, sW = sw0 ?? 1, pH = ph0 ?? 0, pW = pw0 ?? 0;
    if (ctor === 'conv2d') {
      return [inSh[0], outCh, Math.floor((inSh[2] + 2 * pH - kh) / sH) + 1, Math.floor((inSh[3] + 2 * pW - kw) / sW) + 1];
    }
    return [inSh[0], outCh, (inSh[2] - 1) * sH + kh - 2 * pH, (inSh[3] - 1) * sW + kw - 2 * pW];
  }
  if (step.kind === 'unary' && ctor === 'adaptiveAvgPool2d') {
    const [, oH, oW] = decl.args;
    return [inSh[0], inSh[1], oH, oW];
  }
  if (step.kind === 'unary' && ctor === 'ghost') {
    const [, , outCh, kh, kw, , sh0, sw0, ph0, pw0] = decl.args;
    const sH = sh0 ?? 1, sW = sw0 ?? 1, pH = ph0 ?? 0, pW = pw0 ?? 0;
    return [inSh[0], outCh, Math.floor((inSh[2] + 2 * pH - kh) / sH) + 1, Math.floor((inSh[3] + 2 * pW - kw) / sW) + 1];
  }
  if (step.kind === 'unary') return [...inSh];
  if (step.kind === 'binary' && ctor === 'concat') {
    const bSh = shapeOf.get(step.slots[1]);
    if (!bSh) return null;
    const dim = decl.args?.[1] ?? 1;
    const out = [...inSh];
    out[dim] = inSh[dim] + bSh[dim];
    return out;
  }
  if (step.kind === 'binary' && ctor === 'matmul') {
    const bSh = shapeOf.get(step.slots[1]);
    if (!bSh) return null;
    return [inSh[0], inSh[1], inSh[2], bSh[3]];
  }
  if (step.kind === 'binary') return [...inSh];
  return null;
}

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
export function analyticalContribs(rawSpec, reports, inputBounds, rt, scheme) {
  const { ssaSpec: spec } = ssaNormalizeSpec(rawSpec);
  const steps = spec.steps ?? [];
  const repByIndex = new Map(reports.map((r) => [r.index, r]));
  const deqByIndex = new Map(reports.map((r) => [r.index, dequantizedWeights(rt, r, scheme)]));

  // Forward: oo-norm value bound per slot.
  const bound = new Map();
  for (const [slot, b] of inputBounds) bound.set(slot, b);
  for (const p of (spec.ports ?? [])) {
    if (!bound.has(p.slot)) bound.set(p.slot, 0);
  }
  // Tracked slot shapes (from ports + per-op output rules). Needed by the
  // matmul contraction dim and logSoftmax's dim size.
  const shapeOf = new Map();
  for (const p of (spec.ports ?? [])) {
    if (p.shape) shapeOf.set(p.slot, [...p.shape]);
  }
  const linSteps = new Map();
  const stepInBounds = new Map();

  for (let si = 0; si < steps.length; si++) {
    const step = steps[si];
    const decl = spec.layers[step.layer];
    const ctor = decl?.constructor;
    if (step.kind === 'unary' && ctor === 'linear') {
      const rep = repByIndex.get(step.layer);
      if (!rep) throw new Error(`analyticalContribs: no report for linear layer ${step.layer}`);
      const [a, b] = step.slots;
      if (!bound.has(a)) throw new Error(`analyticalContribs: slot ${a} has no bound (dangling input?)`);
      const bA = bound.get(a);
      stepInBounds.set(si, [bA]);
      const deq = deqByIndex.get(step.layer);
      // Sound envelope over pre- and post-quantization operator norms so upward
      // quantization rounding (|W_q| > |W|) never underestimates multi-layer or mul bounds.
      const wNormPre = operatorNormInf(rep.weights, rep.inDim, rep.outDim);
      const wNormPost = deq ? operatorNormInf(deq, rep.inDim, rep.outDim) : wNormPre;
      const wNorm = Math.max(wNormPre, wNormPost);
      const bNormPre = biasNormInf(rep.weights, rep.inDim, rep.outDim, rep.bias);
      const bNormPost = deq ? biasNormInf(deq, rep.inDim, rep.outDim, rep.bias) : bNormPre;
      const bNorm = Math.max(bNormPre, bNormPost);
      bound.set(b, wNorm * bA + bNorm);
      linSteps.set(si, { rep, deq, inSlot: a, outSlot: b, wNorm, inBound: bA });
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(b, outSh);
    } else if (step.kind === 'unary') {
      const [a, b] = step.slots;
      if (!bound.has(a)) throw new Error(`analyticalContribs: slot ${a} has no bound`);
      const inB = bound.get(a);
      const L = unaryLipschitz(decl, spec, step.layer, inB);
      if (L === null) {
        throw new Error(
          `analyticalContribs: unsupported step '${ctor}' (${step.kind}) — bound mode requires known Lipschitz constant`,
        );
      }
      stepInBounds.set(si, [inB]);
      if (NORM_CONSTRUCTORS.has(ctor)) {
        bound.set(b, normOutputBound(decl, inB, spec, step.layer));
      } else if (CONV_CONSTRUCTORS.has(ctor)) {
        bound.set(b, convOutputBound(decl, inB, spec, step.layer));
      } else {
        bound.set(b, unaryOutputBound(decl, inB, spec, step.layer, shapeOf.get(a), L));
      }
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(b, outSh);
    } else if (step.kind === 'binary' && (ctor === 'add' || ctor === 'sub')) {
      const [a, b, c] = step.slots;
      if (!bound.has(a) || !bound.has(b)) throw new Error(`analyticalContribs: ${ctor} slot missing bound`);
      stepInBounds.set(si, [bound.get(a), bound.get(b)]);
      bound.set(c, bound.get(a) + bound.get(b));
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(c, outSh);
    } else if (step.kind === 'binary' && ctor === 'mul') {
      const [a, b, c] = step.slots;
      if (!bound.has(a) || !bound.has(b)) throw new Error('analyticalContribs: mul slot missing bound');
      const bA = bound.get(a), bB = bound.get(b);
      stepInBounds.set(si, [bA, bB]);
      bound.set(c, bA * bB);
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(c, outSh);
    } else if (step.kind === 'binary' && ctor === 'concat') {
      const [a, b, c] = step.slots;
      if (!bound.has(a) || !bound.has(b)) throw new Error('analyticalContribs: concat slot missing bound');
      stepInBounds.set(si, [bound.get(a), bound.get(b)]);
      bound.set(c, Math.max(bound.get(a), bound.get(b)));
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(c, outSh);
    } else if (step.kind === 'binary' && ctor === 'matmul') {
      // C = A*B, [B,C,M,K]x[B,C,K,N]: |C| <= K*||A||_oo*||B||_oo, and the
      // sensitivities are K*||other||_oo (see the backward pass). K is the
      // contraction dim from the tracked shapes.
      const [a, b, c] = step.slots;
      if (!bound.has(a) || !bound.has(b)) throw new Error('analyticalContribs: matmul slot missing bound');
      const bA = bound.get(a), bB = bound.get(b);
      const aSh = shapeOf.get(a);
      if (!aSh) throw new Error('analyticalContribs: matmul input shape unknown (shape tracking gap)');
      const K = aSh[3];
      stepInBounds.set(si, [bA, bB, K]);
      bound.set(c, K * bA * bB);
      const outSh = stepOutputShape(decl, step, shapeOf);
      if (outSh) shapeOf.set(c, outSh);
    } else {
      throw new Error(
        `analyticalContribs: unsupported step '${ctor}' (${step.kind}) — bound mode handles linear + Lipschitz unary + add/sub/mul/concat + matmul + conv + embedding`,
      );
    }
  }

  // Backward: accumulate sensitivity of each slot toward the output slot (supports shared layers, slot compaction & DAG fan-out).
  const sens = new Map([[spec.outputSlot, 1]]);
  const get = (m, k) => m.get(k) ?? 0;
  const contribByLayer = new Map();

  for (let i = steps.length - 1; i >= 0; i--) {
    const step = steps[i];
    const decl = spec.layers[step.layer];
    const ctor = decl?.constructor;
    if (step.kind === 'unary' && ctor === 'linear') {
      const [a, b] = step.slots;
      const ls = linSteps.get(i);
      const sb = get(sens, b);
      sens.delete(b);
      sens.set(a, get(sens, a) + sb * ls.wNorm);
      // |Δy_i| <= Σ_j|ΔW_ij|·|x_j| + |Δb_i| <= ||ΔW||_oo·||x||_oo + ||Δb||_oo.
      // ΔW/Δb are EXACT (dequantized via the engine), so this stays rigorous
      // while being ~6x tighter than the report's bound_W·inDim on outliers.
      const deq = ls.deq ?? dequantizedWeights(rt, ls.rep, scheme);
      const nW = ls.rep.inDim * ls.rep.outDim;
      const errW = operatorNormInf(
        ls.rep.weights.slice(0, nW).map((w, idx) => w - deq[idx]),
        ls.rep.inDim, ls.rep.outDim,
      );
      const errB = ls.rep.bias
        ? Math.max(...ls.rep.weights.slice(nW).map((bv, idx) => Math.abs(bv - deq[nW + idx])))
        : 0;
      const pert = errW * ls.inBound + errB;
      contribByLayer.set(step.layer, (contribByLayer.get(step.layer) ?? 0) + pert * sb);
    } else if (step.kind === 'unary') {
      const [a, b] = step.slots;
      const inB = stepInBounds.get(i)?.[0];
      const L = unaryLipschitz(decl, spec, step.layer, inB ?? null) ?? 1;
      const sb = get(sens, b);
      sens.delete(b);
      sens.set(a, get(sens, a) + sb * L);
    } else if (step.kind === 'binary' && (ctor === 'add' || ctor === 'sub' || ctor === 'concat')) {
      const [a, b, c] = step.slots;
      const sc = get(sens, c);
      sens.delete(c);
      sens.set(a, get(sens, a) + sc);
      sens.set(b, get(sens, b) + sc);
    } else if (step.kind === 'binary' && ctor === 'mul') {
      const [a, b, c] = step.slots;
      const sc = get(sens, c);
      sens.delete(c);
      const [bA, bB] = stepInBounds.get(i) ?? [1, 1];
      sens.set(a, get(sens, a) + sc * bB);
      sens.set(b, get(sens, b) + sc * bA);
    } else if (step.kind === 'binary' && ctor === 'matmul') {
      // |dC| <= K*||dA||_oo*||B||_oo and symmetrically for B.
      const [a, b, c] = step.slots;
      const sc = get(sens, c);
      sens.delete(c);
      const [bA, bB, K] = stepInBounds.get(i) ?? [1, 1, 1];
      sens.set(a, get(sens, a) + sc * K * bB);
      sens.set(b, get(sens, b) + sc * K * bA);
    }
  }

  const contribs = [...contribByLayer.entries()]
    .map(([index, contrib]) => ({ index, contrib }))
    .sort((x, y) => x.index - y.index);
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
      if (!bounds.has(slot) || m > bounds.get(slot)) bounds.set(slot, m);
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
      let t = null;
      try {
        t = g.run(reg, bundle);
        outs.push(Array.from(t.to_array()));
      } finally {
        if (t?.free) t.free();
        if (bundle?.free) bundle.free();
      }
    }
  } finally {
    if (g?.free) g.free();
    if (plan?.free) plan.free();
    if (reg?.free) reg.free();
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
