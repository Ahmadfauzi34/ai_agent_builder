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
  const c = CATALOG[constructor];
  return c ? c.arity : null; // null = unknown constructor
}

const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const shapeStr = (s) => `[${s.join(',')}]`;

// Elementwise unary: shape preserved.
function preserve(inShapes) {
  return [...inShapes[0]];
}

const RULES = {
  // ---- activations (shape-preserving) ----
  relu: preserve, gelu: preserve, sigmoid: preserve, tanh: preserve,
  hardSwish: preserve, hardSigmoid: preserve, softplus: preserve, mish: preserve,
  leakyRelu: preserve, prelu: preserve,
  shiftUp: preserve, shiftDown: preserve, shiftLeft: preserve, shiftRight: preserve,
  seBlock: preserve, featureNorm: preserve,
  softmax: preserve, logSoftmax: preserve,

  glu: (inShapes, args) => {
    const dim = args[1];
    if (!(dim >= 0 && dim <= 3)) throw new ShapeError(`glu dim ${dim} out of range 0..3`);
    const s = [...inShapes[0]];
    if (s[dim] % 2 !== 0) throw new ShapeError(`glu dim ${dim}: size ${s[dim]} not even, cannot halve`);
    s[dim] /= 2;
    return s;
  },

  // ---- norms (shape-preserving, with declared-size checks) ----
  batchNorm: (inShapes, args) => {
    const [n, c] = inShapes[0];
    if (c !== args[1]) throw new ShapeError(`batchNorm num_features ${args[1]} != channel dim ${c}`);
    return [...inShapes[0]];
  },
  groupNorm: (inShapes, args) => {
    const c = inShapes[0][1];
    if (c !== args[2]) throw new ShapeError(`groupNorm num_channels ${args[2]} != channel dim ${c}`);
    if (c % args[1] !== 0) throw new ShapeError(`groupNorm channels ${c} not divisible by num_groups ${args[1]}`);
    return [...inShapes[0]];
  },
  instanceNorm: (inShapes, args) => {
    if (inShapes[0][1] !== args[1]) throw new ShapeError(`instanceNorm num_channels ${args[1]} != channel dim ${inShapes[0][1]}`);
    return [...inShapes[0]];
  },
  layerNorm: (inShapes, args) => {
    if (inShapes[0][1] !== args[1]) throw new ShapeError(`layerNorm size ${args[1]} != channel dim ${inShapes[0][1]}`);
    return [...inShapes[0]];
  },
  rmsNorm: (inShapes, args) => {
    if (inShapes[0][1] !== args[1]) throw new ShapeError(`rmsNorm size ${args[1]} != channel dim ${inShapes[0][1]}`);
    return [...inShapes[0]];
  },

  // ---- linear ----
  linear: (inShapes, args) => {
    const s = inShapes[0];
    if (s[1] !== args[1]) throw new ShapeError(`linear in_dim ${args[1]} != channel dim ${s[1]}`);
    return [s[0], args[2], s[2], s[3]];
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

  concat: (inShapes, args) => {
    const dim = args[1];
    if (!(dim >= 0 && dim <= 3)) throw new ShapeError(`concat dim ${dim} out of range 0..3`);
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
  conv2d: (inShapes, args) => {
    const [, inCh, outCh, kh, kw, sh = 1, sw = 1, ph = 0, pw = 0] = args;
    const [n, c, h, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`conv2d in_ch ${inCh} != channel dim ${c}`);
    const oh = Math.floor((h + 2 * ph - kh) / sh) + 1;
    const ow = Math.floor((w + 2 * pw - kw) / sw) + 1;
    if (oh <= 0 || ow <= 0) throw new ShapeError(`conv2d output spatial dims non-positive [${oh},${ow}]`);
    return [n, outCh, oh, ow];
  },
  convTranspose2d: (inShapes, args) => {
    const [, inCh, outCh, kh, kw, sh = 1, sw = 1, ph = 0, pw = 0] = args;
    const [n, c, h, w] = inShapes[0];
    if (c !== inCh) throw new ShapeError(`convTranspose2d in_ch ${inCh} != channel dim ${c}`);
    return [n, outCh, (h - 1) * sh - 2 * ph + kh, (w - 1) * sw - 2 * pw + kw];
  },
  maxPool2d: (inShapes, args) => pool2d('maxPool2d', inShapes, args),
  avgPool2d: (inShapes, args) => pool2d('avgPool2d', inShapes, args),
  adaptiveAvgPool2d: (inShapes, args) => {
    const [n, c] = inShapes[0];
    return [n, c, args[1], args[2]];
  },
};

function pool2d(name, inShapes, args) {
  const kh = args[1], kw = args[2];
  const sh = args[3] ?? kh, sw = args[4] ?? kw, ph = args[5] ?? 0, pw = args[6] ?? 0;
  const [n, c, h, w] = inShapes[0];
  const oh = Math.floor((h + 2 * ph - kh) / sh) + 1;
  const ow = Math.floor((w + 2 * pw - kw) / sw) + 1;
  if (oh <= 0 || ow <= 0) throw new ShapeError(`${name} output spatial dims non-positive [${oh},${ow}]`);
  return [n, c, oh, ow];
}

// Constructors with no reliable static rule: matmul (ambiguous 4D semantics),
// embedding (index semantics), conv1d/maxPool1d/avgPool1d (ambiguous with 4D),
// ghost, swiGlu (uncertain split semantics). Absent from RULES on purpose.
export function ruleFor(constructor) {
  return RULES[constructor] ?? null;
}

export { CATALOG, shapeStr };
