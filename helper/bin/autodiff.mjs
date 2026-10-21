// burn-math #12: autodiff as a MAP (not a framework).
//
// A declarative table: forward constructor -> backward (VJP) rule.
// Each rule is a PURE function over flat f64 arrays: given the saved
// forward values and the upstream gradient dL/dOut, it returns dL/dIn
// (and dL/dParams for weighted layers). Rules are DATA — inspectable,
// auditable, and each one is PROVEN against two independent oracles:
//   (1) a hand-computed exact case, and
//   (2) central finite differences of the engine's own forward pass.
// A rule that passes only self-consistent checks is not proven (the
// attention_head softmax-axis lesson). The finite-difference oracle
// (bin/gradcheck.mjs) stays the permanent regression guard.
//
// Coverage v1: 15 constructors (the differentiable core behind MLP and
// attention blocks). Everything else is an explicit, classified gap in
// GAPS — a missing rule is an honest error, never a silent wrong answer.
//
// Conventions (engine-faithful, audited):
// - Tensors are flat row-major f64 arrays with shape [d0,d1,d2,d3].
// - linear(id,in_dim,out_dim,bias): input MUST be [B,F,1,1] (the engine
//   rejects anything else); W is column-major: y[b,i] = sum_j
//   w[i+j*O]*x[b,j] + b[i].
// - matmul(id): batched last-two-axis: C[b0,b1,i,j] = sum_k
//   A[b0,b1,i,k]*B[b0,b1,k,j]; engine validates b0/b1 match and a_k==b_k.
// - All elementwise ops require exact shape match (engine-validated).

export const MAP_VERSION = 'autodiff-map.v1';

// ---------- tensor helpers ----------
export const numel = (s) => s.reduce((a, b) => a * b, 1);
const coords = (shape, flat) => {
  const c = [0, 0, 0, 0];
  let r = flat;
  for (let d = 3; d >= 0; d--) { c[d] = r % shape[d]; r = Math.floor(r / shape[d]); }
  return c;
};
const flat = (shape, c) => ((c[0] * shape[1] + c[1]) * shape[2] + c[2]) * shape[3] + c[3];

export function transposeDims(values, shape, d0, d1) {
  const ns = [...shape];
  [ns[d0], ns[d1]] = [ns[d1], ns[d0]];
  const out = new Array(values.length);
  for (let f = 0; f < values.length; f++) {
    const c = coords(ns, f);
    [c[d0], c[d1]] = [c[d1], c[d0]];
    out[f] = values[flat(shape, c)];
  }
  return { values: out, shape: ns };
}

// Sum over `dim`, keeping the dim (size 1) so the result broadcasts.
export function reduceSumKeepdim(values, shape, dim) {
  const ns = [...shape]; ns[dim] = 1;
  const out = new Array(numel(ns)).fill(0);
  for (let f = 0; f < values.length; f++) {
    const c = coords(shape, f); c[dim] = 0;
    out[flat(ns, c)] += values[f];
  }
  return { values: out, shape: ns };
}

function broadcastOf(keepdim, shape) {
  // keepdim has one axis at size 1; expand it to `shape`.
  const out = new Array(numel(shape));
  for (let f = 0; f < out.length; f++) {
    const c = coords(shape, f);
    const kc = c.map((v, d) => (keepdim.shape[d] === 1 ? 0 : v));
    out[f] = keepdim.values[flat(keepdim.shape, kc)];
  }
  return out;
}

const zip3 = (a, b, c, f) => a.map((x, i) => f(x, b[i], c[i]));

// Standard normal PDF and CDF (A&S 7.1.26, ~1e-7; the engine's gelu is
// the exact erf form — probed 2026-10-22: |engine-exact| ~ 2e-8).
function phi(x) { return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI); }
function erfA(x) {
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const e = 1 - poly * Math.exp(-ax * ax);
  return x < 0 ? -e : e;
}
function Phi(x) { return 0.5 * (1 + erfA(x / Math.SQRT2)); }

// ---------- VJP rules ----------
// Each rule: ({inputs:[{values,shape}], output:{values,shape}, upstream:[...], args})
//         -> {inputGrads:[[...],...], paramGrads:{w:[...],b:[...]}|null}
// `inputs` are the SAVED forward values the rule needs (never recomputed).

const ew = (fn) => ({ inputs, upstream: up }) => ({
  inputGrads: [inputs[0].values.map((x, i) => fn(x, up.values[i], i))],
  paramGrads: null,
});

export const RULES = {
  linear: {
    saved: ['input'],
    doc: 'y[b,i]=sum_j w[i+j*O]*x[b,j]+b[i]; W column-major (audited). Input must be [B,F,1,1] (engine-enforced).',
    vjp({ inputs, upstream: up, args, weights }) {
      const [, inD, outD, bias] = args;
      const x = inputs[0], sh = x.shape, B = sh[0], F = sh[1];
      if (F !== inD) throw new Error(`linear vjp: in_dim ${inD} != input axis1 ${F}`);
      if (sh[2] !== 1 || sh[3] !== 1) throw new Error('linear vjp: engine requires [B,F,1,1]');
      const w = weights.w, G = up.values;
      const dx = new Array(x.values.length).fill(0);
      const dw = new Array(inD * outD).fill(0);
      const db = new Array(outD).fill(0);
      for (let b = 0; b < B; b++) for (let i = 0; i < outD; i++) {
        const g = G[(b * outD + i)];
        db[i] += g;
        for (let j = 0; j < inD; j++) {
          dx[(b * inD + j)] += g * w[i + j * outD];
          dw[i + j * outD] += g * x.values[(b * inD + j)];
        }
      }
      return { inputGrads: [dx], paramGrads: { w: dw, b: bias ? db : null } };
    },
  },
  matmul: {
    saved: ['input', 'input'],
    doc: 'C[b0,b1,i,j]=sum_k A[b0,b1,i,k]*B[b0,b1,k,j]. dA=G@B^T, dB=A^T@G over last two axes.',
    vjp({ inputs, upstream: up }) {
      const [A, B] = inputs, G = up.values;
      const [b0, b1, m, k] = A.shape, n = B.shape[3];
      const dA = new Array(A.values.length).fill(0);
      const dB = new Array(B.values.length).fill(0);
      for (let a = 0; a < b0; a++) for (let b = 0; b < b1; b++)
        for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
          const g = G[(((a * b1 + b) * m + i) * n + j)];
          for (let kk = 0; kk < k; kk++) {
            dA[(((a * b1 + b) * m + i) * k + kk)] += g * B.values[(((a * b1 + b) * k + kk) * n + j)];
            dB[(((a * b1 + b) * k + kk) * n + j)] += g * A.values[(((a * b1 + b) * m + i) * k + kk)];
          }
        }
      return { inputGrads: [dA, dB], paramGrads: null };
    },
  },
  transpose: {
    saved: [],
    doc: 'dX = transpose(dL/dY) with the same axes: the VJP of a swap is the swap.',
    vjp({ upstream: up, args, outShape }) {
      const [, d0, d1] = args;
      return { inputGrads: [transposeDims(up.values, outShape, d0, d1).values], paramGrads: null };
    },
  },
  add: {
    saved: [],
    doc: 'dX=dL/dY, dY=dL/dY (fan-out).',
    vjp({ upstream: up }) {
      return { inputGrads: [[...up.values], [...up.values]], paramGrads: null };
    },
  },
  sub: {
    saved: [],
    doc: 'dX=G, dY=-G.',
    vjp({ upstream: up }) {
      return { inputGrads: [[...up.values], up.values.map((g) => -g)], paramGrads: null };
    },
  },
  mul: {
    saved: ['input', 'input'],
    doc: 'dX=G*y, dY=G*x (elementwise).',
    vjp({ inputs, upstream: up }) {
      const [x, y] = inputs, G = up.values;
      return {
        inputGrads: [
          x.values.map((xv, i) => G[i] * y.values[i]),
          y.values.map((yv, i) => G[i] * x.values[i]),
        ],
        paramGrads: null,
      };
    },
  },
  relu: {
    saved: ['input'],
    doc: 'dX=G*(x>0). Subgradient at x=0 is 0 (documented choice).',
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => (x > 0 ? up.values[i] : 0))],
        paramGrads: null,
      };
    },
  },
  leakyRelu: {
    saved: ['input'],
    doc: 'dX=G*(x>0?1:negative_slope). Subgradient at 0 uses slope 1.',
    vjp({ inputs, upstream: up, args }) {
      const alpha = args[1];
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] * (x > 0 ? 1 : alpha))],
        paramGrads: null,
      };
    },
  },
  sigmoid: {
    saved: ['output'],
    doc: 'dX=G*s*(1-s), s=saved output (avoids recomputing the exp).',
    vjp({ output, upstream: up }) {
      return {
        inputGrads: [output.values.map((s, i) => up.values[i] * s * (1 - s))],
        paramGrads: null,
      };
    },
  },
  tanh: {
    saved: ['output'],
    doc: 'dX=G*(1-t^2), t=saved output.',
    vjp({ output, upstream: up }) {
      return {
        inputGrads: [output.values.map((t, i) => up.values[i] * (1 - t * t))],
        paramGrads: null,
      };
    },
  },
  gelu: {
    saved: ['input'],
    doc: "Exact gelu(x)=x*Phi(x) (engine-probed 2026-10-22). dX=G*(Phi(x)+x*phi(x)).",
    vjp({ inputs, upstream: up }) {
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] * (Phi(x) + x * phi(x)))],
        paramGrads: null,
      };
    },
  },
  softplus: {
    saved: ['input'],
    doc: 'softplus(x;beta)=ln(1+e^(beta*x))/beta; d/dx=sigmoid(beta*x). beta=args[1].',
    vjp({ inputs, upstream: up, args }) {
      const beta = args[1];
      return {
        inputGrads: [inputs[0].values.map((x, i) => up.values[i] / (1 + Math.exp(-beta * x)))],
        paramGrads: null,
      };
    },
  },
  softmax: {
    saved: ['output'],
    doc: 'dX=s*(G-sum_dim(G*s)); s=saved output. dim is the constructor arg.',
    vjp({ output, upstream: up, args }) {
      const dim = args[1], s = output.values, sh = output.shape, G = up.values;
      const gs = s.map((sv, i) => G[i] * sv);
      const summed = reduceSumKeepdim(gs, sh, dim);
      const bcast = broadcastOf({ values: summed.values, shape: summed.shape }, sh);
      return { inputGrads: [s.map((sv, i) => sv * (G[i] - bcast[i]))], paramGrads: null };
    },
  },
  logSoftmax: {
    saved: ['output'],
    doc: 'dX=G-s*sum_dim(G); s=exp(saved log-output)=softmax.',
    vjp({ output, upstream: up, args }) {
      const dim = args[1], sh = output.shape, G = up.values;
      const s = output.values.map(Math.exp);
      const summed = reduceSumKeepdim(G, sh, dim);
      const bcast = broadcastOf({ values: summed.values, shape: summed.shape }, sh);
      return { inputGrads: [G.map((g, i) => g - s[i] * bcast[i])], paramGrads: null };
    },
  },
  concat: {
    saved: [],
    doc: 'dX_i = slice of G along dim per the forward input shapes.',
    vjp({ upstream: up, args, inShapes }) {
      const dim = args[1];
      const grads = [];
      let off = 0;
      for (const sh of inShapes) {
        const n = sh[dim], g = new Array(numel(sh));
        for (let f = 0; f < g.length; f++) {
          const c = coords(sh, f); c[dim] += off;
          // upstream shape: same as sh but dim extended
          const us = [...sh]; us[dim] = up.shape[dim];
          g[f] = up.values[flat(us, c)];
        }
        grads.push(g);
        off += n;
      }
      return { inputGrads: grads, paramGrads: null };
    },
  },
};

// ---------- classified gaps (honest, not silent) ----------
export const GAPS = {
  // Non-differentiable: no VJP exists (or only a subgradient needing engine support).
  maxPool1d: 'non-differentiable here: subgradient routes via argmax mask, engine does not expose indices',
  maxPool2d: 'non-differentiable here: subgradient routes via argmax mask, engine does not expose indices',
  embedding: 'non-differentiable w.r.t. integer indices (differentiable w.r.t. weight rows; deferred)',
  // Differentiable but deferred: rule not yet written/proven.
  conv1d: 'deferred: im2col-style VJP not yet proven',
  conv2d: 'deferred: VJP not yet proven',
  convTranspose2d: 'deferred: VJP not yet proven',
  batchNorm: 'deferred: VJP needs saved batch stats; not yet proven',
  layerNorm: 'deferred: VJP not yet proven',
  groupNorm: 'deferred: VJP not yet proven',
  instanceNorm: 'deferred: VJP not yet proven',
  rmsNorm: 'deferred: VJP not yet proven',
  featureNorm: 'deferred: VJP not yet proven',
  avgPool1d: 'deferred: VJP not yet proven',
  avgPool2d: 'deferred: VJP not yet proven',
  adaptiveAvgPool2d: 'deferred: VJP not yet proven',
  ghost: 'deferred: VJP not yet proven',
  seBlock: 'deferred: VJP not yet proven',
  glu: 'deferred: VJP not yet proven',
  swiGlu: 'deferred: VJP not yet proven',
  hardSwish: 'deferred: VJP not yet proven',
  hardSigmoid: 'deferred: VJP not yet proven',
  mish: 'deferred: VJP not yet proven',
  prelu: 'deferred: VJP not yet proven',
  shiftUp: 'deferred: VJP is inverse shift; not yet proven',
  shiftDown: 'deferred: VJP is inverse shift; not yet proven',
  shiftLeft: 'deferred: VJP is inverse shift; not yet proven',
  shiftRight: 'deferred: VJP is inverse shift; not yet proven',
};

export function ruleFor(constructor) {
  const r = RULES[constructor];
  if (!r) {
    const gap = GAPS[constructor];
    throw new Error(`no backward rule for '${constructor}'` + (gap ? `: ${gap}` : ' (unknown constructor)'));
  }
  return r;
}

export function ruleNames() { return Object.keys(RULES); }
export function gapNames() { return Object.keys(GAPS); }

// ---------- shape inference (engine-faithful; supplements lib/shapes.mjs) ----------
// lib/shapes.mjs lacks matmul/transpose and its linear rule does not enforce
// the engine's [B,F,1,1] requirement (probed 2026-10-22). These rules mirror
// the engine validators exactly.
function shapeTranspose(inShapes, args) {
  const [, d0, d1] = args;
  if (!(d0 >= 0 && d0 <= 3 && d1 >= 0 && d1 <= 3)) throw new Error(`transpose dims out of range`);
  const s = [...inShapes[0]];
  [s[d0], s[d1]] = [s[d1], s[d0]];
  return s;
}
function shapeMatmul(inShapes) {
  const [a, b] = inShapes;
  if (a[0] !== b[0] || a[1] !== b[1]) throw new Error(`matmul batch mismatch`);
  if (a[3] !== b[2]) throw new Error(`matmul k mismatch ${a[3]} vs ${b[2]}`);
  return [a[0], a[1], a[2], b[3]];
}
function shapeLinearStrict(inShapes, args) {
  const s = inShapes[0], [, inD, outD] = args;
  if (s[2] !== 1 || s[3] !== 1) throw new Error(`linear: engine requires [B,F,1,1], got [${s}]`);
  if (s[1] !== inD) throw new Error(`linear in_dim ${inD} != ${s[1]}`);
  return [s[0], outD, 1, 1];
}
const SHAPE_RULES_EXTRA = { transpose: shapeTranspose, matmul: shapeMatmul, linear: shapeLinearStrict };

export function inferShapes(spec) {
  // Returns Map slot -> shape. Throws an honest error on the first gap.
  const shapes = new Map(spec.ports.map((p) => [p.slot, [...p.shape]]));
  for (const step of spec.steps) {
    const layer = spec.layers[step.layer];
    const inShapes = step.slots.slice(0, step.kind === 'binary' ? 2 : 1).map((s) => {
      const sh = shapes.get(s);
      if (!sh) throw new Error(`shape inference: slot ${s} has no known shape`);
      return sh;
    });
    let out;
    const extra = SHAPE_RULES_EXTRA[layer.constructor];
    if (extra) {
      out = extra(inShapes, layer.args);
    } else {
      // fall back to lib/shapes.mjs for the rest (activations preserve, concat, ...)
      out = shapeRuleFromLib(layer.constructor, inShapes, layer.args);
    }
    const outSlot = step.slots[step.kind === 'binary' ? 2 : 1];
    shapes.set(outSlot, out);
    step._outShape = out; step._inShapes = inShapes;
  }
  return shapes;
}

import { ruleFor as libShapeRule } from './shapes.mjs';
function shapeRuleFromLib(constructor, inShapes, args) {
  const r = libShapeRule(constructor);
  if (!r) throw new Error(`no shape rule for '${constructor}' (add one or extend SHAPE_RULES_EXTRA)`);
  return r(inShapes, args);
}


// ---------- reverse-mode builder (pure; engine only supplies `saved`) ----------
// saved: Map slot -> {values, shape} (forward values each rule needs).
// weightOf: (layerIdx) -> {w:[...], b:[...]|null} for linear layers.
// doutSeed: {slot: [...]} (default: ones at outputSlot).
export function backwardWithSaved(spec, shapes, saved, doutSeed, weightOf) {
  const grads = new Map();
  const paramGrads = {};
  const outSlot = spec.outputSlot;
  const outShape = shapes.get(outSlot);
  const seed = doutSeed?.[outSlot] ?? new Array(numel(outShape)).fill(1);
  grads.set(outSlot, [...seed]);
  for (let si = spec.steps.length - 1; si >= 0; si--) {
    const step = spec.steps[si];
    const layer = spec.layers[step.layer];
    const rule = ruleFor(layer.constructor);
    const isBin = step.kind === 'binary';
    const inSlots = step.slots.slice(0, isBin ? 2 : 1);
    const os = step.slots[isBin ? 2 : 1];
    const oShape = shapes.get(os);
    const upstream = { values: grads.get(os) ?? new Array(numel(oShape)).fill(0), shape: oShape };
    const ctx = {
      inputs: inSlots.map((sl) => {
        const sv = saved.get(sl);
        if (!sv) throw new Error(`rule '${layer.constructor}' needs saved input but slot ${sl} was not captured`);
        return sv;
      }),
      output: (() => {
        const sv = saved.get(os);
        if (rule.saved.includes('output') && !sv)
          throw new Error(`rule '${layer.constructor}' needs saved output of slot ${os}`);
        return sv ?? { values: [], shape: oShape };
      })(),
      upstream,
      args: layer.args,
      inShapes: inSlots.map((sl) => shapes.get(sl)),
      outShape: oShape,
      weights: layer.constructor === 'linear' ? weightOf(step.layer, layer.args) : null,
    };
    const { inputGrads, paramGrads: pg } = rule.vjp(ctx);
    inSlots.forEach((sl, k) => {
      const cur = grads.get(sl) ?? new Array(numel(shapes.get(sl))).fill(0);
      grads.set(sl, cur.map((v, i) => v + inputGrads[k][i]));
    });
    if (pg) paramGrads[String(step.layer)] = pg;
  }
  const inputGrads = {};
  for (const po of spec.ports) inputGrads[String(po.slot)] = grads.get(po.slot) ?? null;
  return { inputGrads, paramGrads };
}

export function makeMap() {
  return {
    version: MAP_VERSION,
    rules: Object.entries(RULES).map(([name, r]) => ({
      name, saved: r.saved, doc: r.doc, status: 'proven: hand-computed + finite-difference oracle',
    })),
    gaps: { ...GAPS },
    note: 'A missing rule is an honest error, never a silent wrong answer. The finite-difference oracle (bin/gradcheck.mjs) is the permanent regression guard.',
  };
}

export { zip3 as _zip3 }; // (test hook)
