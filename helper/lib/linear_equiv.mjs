// burn-math #11: exact equivalence for linear subgraphs.
//
// A pure-linear spec (a chain of unary `linear` steps) denotes an exact
// affine map y = Wx + b, computed here host-side in f64 by symbolic
// composition. Two specs are declared ACCEPT iff their effective (W, b)
// are bit-identical elementwise — no tolerance involved.
//
// Scope (deliberate, stated): single input slot -> outputSlot, unary
// `linear` steps only. Anything else (activations, binary ops, branches)
// -> REJECT with a reason. This is a proof about the SPEC's symbolic form,
// not about engine numerics: ACCEPT means the two specs denote the same
// affine map exactly, so any faithful executor must agree on every input.
//
// Weight layout (audited): kernel column-major flat[k*out_dim + i],
// bias (if the layer declares it) appended after the kernel.
import { portBySlot } from './spec.mjs';

// Row-major nested arrays in JS; layout conversion happens at the boundary.
function kernelToRows(flat, inDim, outDim) {
  const W = [];
  for (let i = 0; i < outDim; i++) {
    const row = [];
    for (let k = 0; k < inDim; k++) row.push(flat[k * outDim + i]);
    W.push(row);
  }
  return W;
}

function matMulRows(A, B) {
  // A: m x n, B: n x p -> m x p
  const m = A.length, n = B.length, p = B[0].length;
  const C = Array.from({ length: m }, () => new Array(p).fill(0));
  for (let i = 0; i < m; i++)
    for (let k = 0; k < n; k++) {
      const aik = A[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < p; j++) C[i][j] += aik * B[k][j];
    }
  return C;
}

function matVecRows(A, v) {
  return A.map((row) => row.reduce((s, a, k) => s + a * v[k], 0));
}

function addVec(a, b) {
  return a.map((x, i) => x + b[i]);
}

// Compose B-after-A: y = B.W (A.W x + A.b) + B.b. General (non-square) dims.
export function composeAffineGeneral(A, B) {
  if (A.W[0].length !== B.W.length)
    throw new Error(`composeAffineGeneral: inner dim mismatch ${A.W[0].length} vs ${B.W.length}`);
  const W = matMulRows(B.W, A.W);
  const b = addVec(matVecRows(B.W, A.b), B.b);
  return { W, b };
}

// Effective affine map of a spec: {W, b, inDim, outDim, chain: [layerIds]}.
// Throws an object {code:'REJECT', reason} for out-of-scope specs.
export function effectiveAffine(spec) {
  const reject = (reason) => { throw { code: 'REJECT', reason }; };
  const layers = spec.layers ?? [];
  const byId = new Map(); // layer index -> {inDim, outDim, W, b}
  for (const [li, decl] of layers.entries()) {
    if (decl.constructor !== 'linear')
      reject(`layer ${li} is '${decl.constructor}', not linear`);
    const [id, inDim, outDim, bias] = decl.args;
    const flat = (spec.weights ?? {})[String(li)];
    if (!flat) reject(`linear layer ${li} (id ${id}) has no weights in spec`);
    const need = inDim * outDim + (bias ? outDim : 0);
    if (flat.length < need) reject(`linear layer ${li}: weights length ${flat.length} < ${need}`);
    const W = kernelToRows(flat, inDim, outDim);
    const b = bias ? flat.slice(inDim * outDim, inDim * outDim + outDim) : new Array(outDim).fill(0);
    byId.set(li, { inDim, outDim, W, b });
  }
  const ports = portBySlot(spec);
  // slot -> affine {W, b} | 'input' marker
  const aff = new Map();
  const chain = [];
  for (const step of spec.steps ?? []) {
    if (step.kind !== 'unary') reject(`non-unary step (kind '${step.kind}')`);
    const L = byId.get(step.layer);
    if (!L) reject(`step references non-linear layer ${step.layer}`);
    const [inSlot, outSlot] = step.slots;
    let prev = aff.get(inSlot);
    if (prev === undefined) {
      // Fresh slot: it must be a declared input port, else the chain is broken.
      if (!ports.has(inSlot)) reject(`linear chain reads slot ${inSlot} with no affine and no input port`);
      const shape = ports.get(inSlot).shape;
      const inDim = shape[1];
      if (inDim !== L.inDim) reject(`input port dim ${inDim} != layer in_dim ${L.inDim}`);
      prev = { W: identityRows(inDim), b: new Array(inDim).fill(0) };
    }
    if (prev.W[0].length !== L.inDim)
      reject(`dim mismatch into linear layer ${step.layer}: ${prev.W[0].length} vs ${L.inDim}`);
    aff.set(outSlot, composeAffineGeneral(prev, L));
    chain.push(step.layer);
  }
  const out = aff.get(spec.outputSlot);
  if (!out) reject(`outputSlot ${spec.outputSlot} is not the end of a linear chain`);
  return { W: out.W, b: out.b, inDim: out.W[0].length, outDim: out.W.length, chain };
}

function identityRows(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

// Exact comparison: ACCEPT iff same dims and every element === (bit-identical).
export function equivLinear(specA, specB) {
  let A, B;
  try { A = effectiveAffine(specA); }
  catch (e) { return e?.code === 'REJECT' ? { verdict: 'REJECT', reason: `specA: ${e.reason}` } : (() => { throw e; })(); }
  try { B = effectiveAffine(specB); }
  catch (e) { return e?.code === 'REJECT' ? { verdict: 'REJECT', reason: `specB: ${e.reason}` } : (() => { throw e; })(); }
  if (A.inDim !== B.inDim || A.outDim !== B.outDim)
    return { verdict: 'REJECT', reason: `dim mismatch A(${A.inDim}->${A.outDim}) vs B(${B.inDim}->${B.outDim})` };
  for (let i = 0; i < A.outDim; i++) {
    for (let k = 0; k < A.inDim; k++) {
      if (A.W[i][k] !== B.W[i][k])
        return { verdict: 'REJECT', reason: `W[${i}][${k}] differs`, firstDiff: { kind: 'W', i, k, a: A.W[i][k], b: B.W[i][k] } };
    }
    if (A.b[i] !== B.b[i])
      return { verdict: 'REJECT', reason: `b[${i}] differs`, firstDiff: { kind: 'b', i, a: A.b[i], b: B.b[i] } };
  }
  return {
    verdict: 'ACCEPT',
    reason: `bit-identical affine map ${A.inDim}->${A.outDim} over ${A.chain.length + B.chain.length} linear layers`,
    dims: { inDim: A.inDim, outDim: A.outDim },
    chainA: A.chain, chainB: B.chain,
  };
}
