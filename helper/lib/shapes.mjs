// Shared shape-propagation rules for burn-math specs (used by shapecheck.mjs,
// and later by rewrite.mjs to validate transformed graphs).
//
// Shapes are 4-D [N, C, H, W]. Each rule is fn(inShapes, args) -> outShape,
// throwing ShapeError on a provable mismatch. Constructors without a reliable
// rule are absent from RULES: callers must treat them as unchecked (warning),
// never as an error — a wrong rule is worse than no rule.
import { createRequire } from 'node:module';
import { catalogPath } from './engine.mjs';

const require = createRequire(import.meta.url);
const CATALOG = require(catalogPath()).constructors;

export class ShapeError extends Error {}

export function arityOf(constructor) {
  // hasOwn: without it, CATALOG['toString'] returns the inherited
  // Object.prototype function and an unknown constructor looks known.
  if (typeof constructor !== 'string' || !Object.hasOwn(CATALOG, constructor)) return null;
  return CATALOG[constructor].arity ?? null; // null = unknown constructor
}

const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const shapeStr = (s) => `[${s.join(',')}]`;

// Elementwise unary: shape preserved.
function preserve(inShapes) {
  return [...inShapes[0]];
}

// WASM initAgentLayer rejects dim=0 ("dimension must be positive") and
// forward rejects dim>=4 for softmax/logSoftmax/glu/concat — valid axis
// range is 1..3 (axis 0 is the batch dim). Ported from pack v2.5.
function softmaxRule(name) {
  return (inShapes, args) => {
    const dim = args[1];
    if (!(dim >= 1 && dim <= 3)) throw new ShapeError(`${name} dim ${dim} out of range 1..3 (axis 0 batch dim is not allowed)`);
    return [...inShapes[0]];
  };
}

function binaryElementwise(name) {
  return (inShapes) => {
    if (!same(inShapes[0], inShapes[1]))
      throw new ShapeError(`${name} shape mismatch ${shapeStr(inShapes[0])} vs ${shapeStr(inShapes[1])}`);
    return [...inShapes[0]];
  };
}

function checkEpsilon(name, eps) {
  if (eps !== undefined && eps !== null && !(eps > 0)) {
    throw new ShapeError(`${name} epsilon must be > 0, got ${eps}`);
  }
}

const RULES = {
  // ---- activations (shape-preserving) ----
  relu: preserve, gelu: preserve, sigmoid: preserve, tanh: preserve,
  hardSwish: preserve, hardSigmoid: preserve, mish: preserve,
  // leakyRelu/softplus/prelu below: engine-validated bounds (port v2.5).
  // The engine rejects negative slope, non-positive beta, and a
  // num_parameters that is neither 1 nor the channel dim at create time.
  shiftUp: preserve, shiftDown: preserve, shiftLeft: preserve, shiftRight: preserve,
  seBlock: preserve, featureNorm: preserve,
  softmax: softmaxRule('softmax'), logSoftmax: softmaxRule('logSoftmax'),

  leakyRelu: (inShapes, args) => {
    if (!(args[1] >= 0)) throw new ShapeError(`leakyRelu negative_slope must be >= 0, got ${args[1]}`);
    return [...inShapes[0]];
  },
  softplus: (inShapes, args) => {
    if (!(args[1] > 0)) throw new ShapeError(`softplus beta must be > 0, got ${args[1]}`);
    return [...inShapes[0]];
  },
  prelu: (inShapes, args) => {
    const numParams = args[1];
    const c = inShapes[0][1];
    if (!(numParams > 0)) throw new ShapeError(`prelu num_parameters must be > 0, got ${numParams}`);
    if (numParams !== 1 && numParams !== c) {
      throw new ShapeError(`prelu num_parameters ${numParams} must be 1 or match channel dim ${c}`);
    }
    return [...inShapes[0]];
  },
  swiGlu: (inShapes, args) => {
    const [, dIn, dOut] = args;
    if (!(dIn > 0 && dOut > 0)) throw new ShapeError(`swiGlu dims [${dIn},${dOut}] must be > 0`);
    const [n, c, h, w] = inShapes[0];
    if (w !== dIn) throw new ShapeError(`swiGlu d_input ${dIn} != last-axis feature dim ${w}`);
    return [n, c, h, dOut];
  },

  glu: (inShapes, args) => {
    const dim = args[1];
    if (!(dim >= 1 && dim <= 3)) throw new ShapeError(`glu dim ${dim} out of range 1..3 (axis 0 batch dim is not allowed)`);
    const s = [...inShapes[0]];
    if (s[dim] % 2 !== 0) throw new ShapeError(`glu dim ${dim}: size ${s[dim]} not even, cannot halve`);
    s[dim] /= 2;
    return s;
  },

  transpose: (inShapes, args) => {
    const dim0 = args[1], dim1 = args[2];
    if (!(dim0 >= 0 && dim0 <= 3) || !(dim1 >= 0 && dim1 <= 3))
      throw new ShapeError(`transpose dims [${dim0},${dim1}] out of range 0..3`);
    const out = [...inShapes[0]];
    const tmp = out[dim0];
    out[dim0] = out[dim1];
    out[dim1] = tmp;
    return out;
  },

  // ---- norms (shape-preserving, with declared-size checks) ----
  // Port v2.5: the engine validates size args and epsilon (> 0) at
  // create time — probed for batchNorm (num_features 0, epsilon 0),
  // groupNorm (num_groups 0, num_channels 0), instanceNorm, layerNorm,
  // rmsNorm (size 0). The old rules only checked the channel match.
  batchNorm: (inShapes, args) => {
    const [n, c] = inShapes[0];
    if (!(args[1] > 0)) throw new ShapeError(`batchNorm num_features must be > 0, got ${args[1]}`);
    checkEpsilon('batchNorm', args[2]);
    if (c !== args[1]) throw new ShapeError(`batchNorm num_features ${args[1]} != channel dim ${c}`);
    return [...inShapes[0]];
  },
  groupNorm: (inShapes, args) => {
    const c = inShapes[0][1];
    if (!(args[1] > 0)) throw new ShapeError(`groupNorm num_groups must be > 0, got ${args[1]}`);
    if (!(args[2] > 0)) throw new ShapeError(`groupNorm num_channels must be > 0, got ${args[2]}`);
    checkEpsilon('groupNorm', args[3]);
    if (c !== args[2]) throw new ShapeError(`groupNorm num_channels ${args[2]} != channel dim ${c}`);
    if (c % args[1] !== 0) throw new ShapeError(`groupNorm channels ${c} not divisible by num_groups ${args[1]}`);
    return [...inShapes[0]];
  },
  instanceNorm: (inShapes, args) => {
    if (!(args[1] > 0)) throw new ShapeError(`instanceNorm num_channels must be > 0, got ${args[1]}`);
    checkEpsilon('instanceNorm', args[2]);
    if (inShapes[0][1] !== args[1]) throw new ShapeError(`instanceNorm num_channels ${args[1]} != channel dim ${inShapes[0][1]}`);
    return [...inShapes[0]];
  },
  // Ported from pack v2.4 (Audit_7 E1): layerNorm/rmsNorm normalize over
  // the LAST axis (feature_last [B,*,*,F], feature_axis 3) — the engine
  // rejects axis-3 mismatches at run time. The old rules checked axis 1
  // (channel), so shapecheck and run disagreed in both directions.
  // batch/group/instanceNorm are channel_first and correctly keep axis 1.
  layerNorm: (inShapes, args) => {
    if (!(args[1] > 0)) throw new ShapeError(`layerNorm size must be > 0, got ${args[1]}`);
    checkEpsilon('layerNorm', args[2]);
    if (inShapes[0][3] !== args[1]) throw new ShapeError(`layerNorm size ${args[1]} != feature dim ${inShapes[0][3]}`);
    return [...inShapes[0]];
  },
  rmsNorm: (inShapes, args) => {
    if (!(args[1] > 0)) throw new ShapeError(`rmsNorm size must be > 0, got ${args[1]}`);
    checkEpsilon('rmsNorm', args[2]);
    if (inShapes[0][3] !== args[1]) throw new ShapeError(`rmsNorm size ${args[1]} != feature dim ${inShapes[0][3]}`);
    return [...inShapes[0]];
  },

  // ---- linear ----
  // Port v2.5: the engine requires [B,F,1,1] (probed 2026-10-22, also
  // stated in the autodiff linear VJP doc). The old rule preserved H,W,
  // so shapecheck passed specs the engine rejects.
  linear: (inShapes, args) => {
    const s = inShapes[0];
    if (!(args[1] > 0 && args[2] > 0)) throw new ShapeError(`linear dims [${args[1]},${args[2]}] must be > 0`);
    if (s[1] !== args[1]) throw new ShapeError(`linear in_dim ${args[1]} != channel dim ${s[1]}`);
    if (s[2] !== 1 || s[3] !== 1) throw new ShapeError(`linear requires [B,F,1,1] (axes 2 and 3 must be 1), got ${shapeStr(s)}`);
    return [s[0], args[2], 1, 1];
  },

  // ---- embedding ----
  embedding: (inShapes, args) => {
    const [, vocabSize, dModel] = args;
    if (!(vocabSize > 0 && dModel > 0)) throw new ShapeError(`embedding dims [${vocabSize},${dModel}] must be > 0`);
    const s = inShapes[0];
    if (s[2] !== 1 || s[3] !== 1) throw new ShapeError(`embedding requires [B,S,1,1] (axes 2 and 3 must be 1), got ${shapeStr(s)}`);
    return [s[0], s[1], dModel, 1];
  },

  // ---- binary elementwise ----
  add: (inShapes) => {
    if (!same(inShapes[0], inShapes[1]))
      throw new ShapeError(`add shape mismatch ${shapeStr(inShapes[0])} vs ${shapeStr(inShapes[1])}`);
    return [...inShapes[0]];
  },
  sub: (inShapes) => {
    if (!same(inShapes[0], inShapes[1]))
      throw new ShapeError(`sub shape mismatch ${shapeStr(inShapes[0])} vs ${shapeStr(inShapes[1])}`);
    return [...inShapes[0]];
  },
  mul: (inShapes) => {
    if (!same(inShapes[0], inShapes[1]))
      throw new ShapeError(`mul shape mismatch ${shapeStr(inShapes[0])} vs ${shapeStr(inShapes[1])}`);
    return [...inShapes[0]];
  },
  intAdd: binaryElementwise('intAdd'),
  intSub: binaryElementwise('intSub'),
  intMul: binaryElementwise('intMul'),
  intDiv: binaryElementwise('intDiv'),
  intRem: binaryElementwise('intRem'),
  // Casts (ported from the agent pack for REG-01 completeness): shape
  // preserved, element type changes (f32 <-> i64). Unary, like preserve.
  intToFloat: preserve, intFromFloat: preserve,

  matmul: (inShapes) => {
    const [a, b] = inShapes;
    if (a[0] !== b[0] || a[1] !== b[1] || a[3] !== b[2]) {
      throw new ShapeError(`matmul shape mismatch ${shapeStr(a)} @ ${shapeStr(b)} (requires [B,C,M,K] @ [B,C,K,N])`);
    }
    return [a[0], a[1], a[2], b[3]];
  },
  intMatmul: (inShapes) => {
    const [a, b] = inShapes;
    if (a[0] !== b[0] || a[1] !== b[1] || a[3] !== b[2]) {
      throw new ShapeError(`intMatmul shape mismatch ${shapeStr(a)} @ ${shapeStr(b)} (requires [B,C,M,K] @ [B,C,K,N])`);
    }
    return [a[0], a[1], a[2], b[3]];
  },

  concat: (inShapes, args) => {
    const dim = args[1];
    if (!(dim >= 1 && dim <= 3)) throw new ShapeError(`concat dim ${dim} out of range 1..3 (axis 0 batch dim is not allowed)`);
    const base = inShapes[0];
    let sum = 0;
    for (const s of inShapes) {
      for (let d = 0; d < 4; d++) {
        if (d !== dim && s[d] !== base[d])
          throw new ShapeError(`concat axis ${d} mismatch ${shapeStr(s)} vs ${shapeStr(base)}`);
      }
      sum += s[dim];
    }
    const out = [...base];
    out[dim] = sum;
    return out;
  },

  // ---- conv / pool (standard formulas) ----
  // Ported from pack v2.4 (Audit_7 E4): Option params arrive as null
  // from JSON (there is no undefined in JSON). Destructuring defaults
  // only fire on undefined, so null strides/padding used to flow into
  // arithmetic. Normalize with ?? so null means "default", like None.
  conv2d: (inShapes, args) => {
    const [, inCh, outCh, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sh = sh0 ?? 1, sw = sw0 ?? 1, ph = ph0 ?? 0, pw = pw0 ?? 0;
    if (!(inCh > 0 && outCh > 0)) throw new ShapeError(`conv2d channels [${inCh},${outCh}] must be > 0`);
    if (!(kh > 0 && kw > 0)) throw new ShapeError(`conv2d kernel [${kh},${kw}] must be > 0`);
    if (!(sh > 0 && sw > 0)) throw new ShapeError(`conv2d stride [${sh},${sw}] must be > 0`);
    const [n, c, h, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`conv2d in_ch ${inCh} != channel dim ${c}`);
    const oh = Math.floor((h + 2 * ph - kh) / sh) + 1;
    const ow = Math.floor((w + 2 * pw - kw) / sw) + 1;
    if (!Number.isFinite(oh) || !Number.isFinite(ow) || oh <= 0 || ow <= 0)
      throw new ShapeError(`conv2d output spatial dims non-positive [${oh},${ow}]`);
    return [n, outCh, oh, ow];
  },
  convTranspose2d: (inShapes, args) => {
    const [, inCh, outCh, kh, kw, sh0, sw0, ph0, pw0] = args;
    const sh = sh0 ?? 1, sw = sw0 ?? 1, ph = ph0 ?? 0, pw = pw0 ?? 0;
    if (!(inCh > 0 && outCh > 0)) throw new ShapeError(`convTranspose2d channels [${inCh},${outCh}] must be > 0`);
    if (!(kh > 0 && kw > 0)) throw new ShapeError(`convTranspose2d kernel [${kh},${kw}] must be > 0`);
    if (!(sh > 0 && sw > 0)) throw new ShapeError(`convTranspose2d stride [${sh},${sw}] must be > 0`);
    const [n, c, h, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`convTranspose2d in_ch ${inCh} != channel dim ${c}`);
    const oh = (h - 1) * sh - 2 * ph + kh;
    const ow = (w - 1) * sw - 2 * pw + kw;
    if (!Number.isFinite(oh) || !Number.isFinite(ow) || oh <= 0 || ow <= 0)
      throw new ShapeError(`convTranspose2d output spatial dims non-positive [${oh},${ow}]`);
    return [n, outCh, oh, ow];
  },
  // Port v2.5: conv1d/pool1d/ghost rules, engine-probed (conv1d [1,2,4,1]
  // -> [1,3,3,1]; ghost out 4 ratio 2 -> [1,4,2,2]; avgPool1d k2 s2 ->
  // [1,2,2,1] with exact window means).
  conv1d: (inShapes, args) => {
    const [, inCh, outCh, k, s0, p0] = args;
    const s = s0 ?? 1, p = p0 ?? 0;
    if (!(inCh > 0 && outCh > 0)) throw new ShapeError(`conv1d channels [${inCh},${outCh}] must be > 0`);
    if (!(k > 0)) throw new ShapeError(`conv1d kernel ${k} must be > 0`);
    if (!(s > 0)) throw new ShapeError(`conv1d stride ${s} must be > 0`);
    const [n, c, l, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`conv1d in_ch ${inCh} != channel dim ${c}`);
    if (w !== 1) throw new ShapeError(`conv1d requires axis 3 to be 1 ([B,C,L,1]), got ${shapeStr(inShapes[0])}`);
    const ol = Math.floor((l + 2 * p - k) / s) + 1;
    if (!Number.isFinite(ol) || ol <= 0) throw new ShapeError(`conv1d output length non-positive [${ol}]`);
    return [n, outCh, ol, 1];
  },
  ghost: (inShapes, args) => {
    const [, inCh, outCh, kh, kw, ratio, sh0, sw0, ph0, pw0] = args;
    const sh = sh0 ?? 1, sw = sw0 ?? 1, ph = ph0 ?? 0, pw = pw0 ?? 0;
    if (!(inCh > 0 && outCh > 0)) throw new ShapeError(`ghost channels [${inCh},${outCh}] must be > 0`);
    if (!(kh > 0 && kw > 0)) throw new ShapeError(`ghost kernel [${kh},${kw}] must be > 0`);
    if (!(ratio > 0)) throw new ShapeError(`ghost ratio ${ratio} must be > 0`);
    if (!(sh > 0 && sw > 0)) throw new ShapeError(`ghost stride [${sh},${sw}] must be > 0`);
    const [n, c, h, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`ghost in_ch ${inCh} != channel dim ${c}`);
    const oh = Math.floor((h + 2 * ph - kh) / sh) + 1;
    const ow = Math.floor((w + 2 * pw - kw) / sw) + 1;
    if (!Number.isFinite(oh) || !Number.isFinite(ow) || oh <= 0 || ow <= 0)
      throw new ShapeError(`ghost output spatial dims non-positive [${oh},${ow}]`);
    return [n, outCh, oh, ow];
  },
  maxPool1d: (inShapes, args) => pool1d('maxPool1d', inShapes, args),
  avgPool1d: (inShapes, args) => pool1d('avgPool1d', inShapes, args),
  maxPool2d: (inShapes, args) => pool2d('maxPool2d', inShapes, args),
  avgPool2d: (inShapes, args) => pool2d('avgPool2d', inShapes, args),
  adaptiveAvgPool2d: (inShapes, args) => {
    const [n, c] = inShapes[0];
    if (!(args[1] > 0 && args[2] > 0))
      throw new ShapeError(`adaptiveAvgPool2d output dims [${args[1]},${args[2]}] must be > 0`);
    return [n, c, args[1], args[2]];
  },
};

function pool1d(name, inShapes, args) {
  const k = args[1];
  const s = args[2] ?? k, p = args[3] ?? 0;
  if (!(k > 0)) throw new ShapeError(`${name} kernel ${k} must be > 0`);
  if (!(s > 0)) throw new ShapeError(`${name} stride ${s} must be > 0`);
  const [n, c, l, w] = inShapes[0];
  if (w !== 1) throw new ShapeError(`${name} requires axis 3 to be 1 ([B,C,L,1]), got ${shapeStr(inShapes[0])}`);
  const ol = Math.floor((l + 2 * p - k) / s) + 1;
  if (!Number.isFinite(ol) || ol <= 0) throw new ShapeError(`${name} output length non-positive [${ol}]`);
  return [n, c, ol, 1];
}

function pool2d(name, inShapes, args) {
  const kh = args[1], kw = args[2];
  const sh = args[3] ?? kh, sw = args[4] ?? kw, ph = args[5] ?? 0, pw = args[6] ?? 0;
  if (!(kh > 0 && kw > 0)) throw new ShapeError(`${name} kernel [${kh},${kw}] must be > 0`);
  if (!(sh > 0 && sw > 0)) throw new ShapeError(`${name} stride [${sh},${sw}] must be > 0`);
  const [n, c, h, w] = inShapes[0];
  const oh = Math.floor((h + 2 * ph - kh) / sh) + 1;
  const ow = Math.floor((w + 2 * pw - kw) / sw) + 1;
  if (!Number.isFinite(oh) || !Number.isFinite(ow) || oh <= 0 || ow <= 0)
    throw new ShapeError(`${name} output spatial dims non-positive [${oh},${ow}]`);
  return [n, c, oh, ow];
}

// Port v2.5: every catalog constructor now has a rule (the old "absent
// on purpose" list — matmul, embedding, conv1d, pool1d, ghost, swiGlu —
// was closed by engine-probed rules). hasOwn: without it,
// RULES['toString'] returns the inherited Object.prototype function and
// a bogus constructor looks like it has a rule.
export function ruleFor(constructor) {
  if (typeof constructor !== 'string' || !Object.hasOwn(RULES, constructor)) return null;
  return RULES[constructor];
}

export { CATALOG, shapeStr };
