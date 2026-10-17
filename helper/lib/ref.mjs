// Independent f64 reference interpreter for burn-math specs (used by diff_ref.mjs).
//
// This is a SEPARATE code path from the WASM engine: plain JS, float64,
// written directly from the documented op semantics (spec-schema.md documents
// linear as column-major: y_i = sum_j w[i + j*outDim] * x_j). Disagreement
// with the engine beyond f32 rounding is a signal, not a verdict —
// investigate which side is wrong (each op below was probed against the
// engine during development; see SUPPORTED).
import { portBySlot } from './spec.mjs';

// Every op receives one context object:
//   { xs: [values, ...], shapes: [[N,C,H,W], ...], args, weights }
// and returns { values, shape? } (shape defaults to the first input's shape).
function linearF64(x, w, inDim, outDim, bias) {
  const y = new Array(outDim).fill(0);
  for (let i = 0; i < outDim; i++) {
    let acc = 0;
    for (let j = 0; j < inDim; j++) acc += w[i + j * outDim] * x[j];
    y[i] = acc;
  }
  if (bias) {
    const b = w.slice(inDim * outDim, inDim * outDim + outDim);
    for (let i = 0; i < outDim; i++) y[i] += b[i];
  }
  return y;
}

// Softmax over one axis of a 4-D [N,C,H,W] tensor.
function softmaxAxis(x, shape, dim) {
  const dims = [...shape];
  const strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
  const y = new Array(x.length);
  const off = (idx) => idx[0] * strides[0] + idx[1] * strides[1] + idx[2] * strides[2] + idx[3];
  const idx = [0, 0, 0, 0];
  const walk = (d, fn) => {
    if (d === 4) { fn(); return; }
    if (d === dim) { walk(d + 1, fn); return; }
    for (let k = 0; k < dims[d]; k++) { idx[d] = k; walk(d + 1, fn); }
  };
  walk(0, () => {
    let m = -Infinity;
    for (let k = 0; k < dims[dim]; k++) { idx[dim] = k; const v = x[off(idx)]; if (v > m) m = v; }
    let s = 0;
    for (let k = 0; k < dims[dim]; k++) { idx[dim] = k; s += Math.exp(x[off(idx)] - m); }
    for (let k = 0; k < dims[dim]; k++) { idx[dim] = k; const o = off(idx); y[o] = Math.exp(x[o] - m) / s; }
  });
  return y;
}

const OPS = {
  linear: ({ xs, shapes, args, weights }) => {
    const [x] = xs;
    const [, inDim, outDim, bias] = args;
    if (x.length !== inDim) throw new Error(`ref linear: input len ${x.length} != in_dim ${inDim}`);
    if (!weights) throw new Error('ref linear: missing weights for layer');
    return { values: linearF64(x, weights, inDim, outDim, bias), shape: [shapes[0][0], outDim, shapes[0][2], shapes[0][3]] };
  },
  relu: ({ xs }) => ({ values: xs[0].map((v) => Math.max(0, v)) }),
  sigmoid: ({ xs }) => ({ values: xs[0].map((v) => 1 / (1 + Math.exp(-v))) }),
  tanh: ({ xs }) => ({ values: xs[0].map((v) => Math.tanh(v)) }),
  add: ({ xs }) => ({ values: xs[0].map((v, i) => v + xs[1][i]) }),
  sub: ({ xs }) => ({ values: xs[0].map((v, i) => v - xs[1][i]) }),
  mul: ({ xs }) => ({ values: xs[0].map((v, i) => v * xs[1][i]) }),
  softmax: ({ xs, shapes, args }) => ({ values: softmaxAxis(xs[0], shapes[0], args[1] ?? 1) }),
};

export const SUPPORTED = Object.keys(OPS);

// Run a spec's dataflow in f64. slots: Map slot -> {values, shape}.
// weights: {layerIndex -> number[]} (column-major, as in the spec).
export function runReference(spec, inputs) {
  const slots = new Map();
  for (const p of spec.ports) {
    const v = inputs[String(p.slot)];
    if (v === undefined) throw new Error(`ref: no input for port slot ${p.slot}`);
    if (v.length !== p.shape.reduce((a, b) => a * b, 1))
      throw new Error(`ref: input len ${v.length} != port shape ${JSON.stringify(p.shape)}`);
    slots.set(p.slot, { values: v.map(Number), shape: [...p.shape] });
  }
  for (const st of spec.steps) {
    const layerDef = spec.layers[st.layer];
    if (!layerDef) throw new Error(`ref: step references missing layer ${st.layer}`);
    const op = OPS[layerDef.constructor];
    if (!op) throw new Error(`ref: unsupported constructor '${layerDef.constructor}' (supported: ${SUPPORTED.join(', ')})`);
    const inSlots = st.kind === 'unary' ? [st.slots[0]] : [st.slots[0], st.slots[1]];
    const outSlot = st.slots[st.slots.length - 1];
    const ins = inSlots.map((s) => {
      const r = slots.get(s);
      if (!r) throw new Error(`ref: slot ${s} read before written`);
      return r;
    });
    if (ins.length === 2 && ins[0].values.length !== ins[1].values.length)
      throw new Error(`ref: binary length mismatch ${ins[0].values.length} vs ${ins[1].values.length}`);
    const r = op({
      xs: ins.map((r) => r.values),
      shapes: ins.map((r) => r.shape),
      args: layerDef.args,
      weights: spec.weights?.[String(st.layer)],
    });
    slots.set(outSlot, { values: r.values, shape: r.shape ?? [...ins[0].shape] });
  }
  const out = slots.get(spec.outputSlot);
  if (!out) throw new Error('ref: outputSlot never written');
  return out.values;
}
