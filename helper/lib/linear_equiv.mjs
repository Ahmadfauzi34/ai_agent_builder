// burn-math #11: exact algebraic equivalence for linear, residual, branched, and recursive affine subgraphs.
//
// Upgraded in v3.9 (Honesty & Fail-Closed Audit):
// - Supports unary `linear`, binary `add`, binary `sub`, and binary `concat` (dim=1)
// - Tracks multi-port affine maps y = sum_p W_p x_p + b across declared ports
// - Traces live steps feeding `outputSlot` so dead port-drain steps do not cause false rejections
// - Strictly forbids folding active non-context input ports (`role !== 'context'`) into bias unless W_p == 0,
//   preventing false ACCEPT verdicts on multi-port graphs with different active input dependencies.
//
// Restored from main (A), scope updated for v3.9: an affine spec denotes an
// exact affine map, computed here host-side in f64 by symbolic composition.
// Two specs are declared ACCEPT iff their effective per-port maps match
// within tol (default tol = 0, i.e. bit-identical elementwise).
//
// This is a proof about the SPEC's symbolic form, not about engine
// numerics: ACCEPT means the two specs denote the same affine map
// exactly, so any faithful executor must agree on every input (a
// conditional ACCEPT additionally records the fixed context ports it is
// conditioned on; see conditional_on_context).
//
// Weight layout (audited): kernel column-major flat[k*out_dim + i],
// bias (if the layer declares it) appended after the kernel.
//
// Concat rule: concat(a, b) along the feature axis (dim=1) of two slots
// that are affine maps of the SAME input is again affine: W rows stack,
// b concatenates. This is what makes a sharded linear (output-row blocks
// rejoined by a concat chain) provably equal to the original.
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

function subVec(a, b) {
  return a.map((x, i) => x - b[i]);
}

function addMat(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

function subMat(A, B) {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

function zeroMat(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function isZeroMat(W, tol = 0) {
  return W.every((row) => row.every((v) => Math.abs(v) <= tol));
}

function identityRows(n) {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
}

// Compose B-after-A: y = B.W (A.W x + A.b) + B.b. General (non-square) dims.
export function composeAffineGeneral(A, B) {
  if (A.W.length !== B.W[0].length)
    throw new Error(`composeAffineGeneral: inner dim mismatch ${A.W.length} vs ${B.W[0].length}`);
  const W = matMulRows(B.W, A.W);
  const b = addVec(matVecRows(B.W, A.b), B.b);
  return { W, b };
}

function computeLiveStepIndices(spec) {
  const steps = spec.steps ?? [];
  const liveSlots = new Set([spec.outputSlot]);
  const liveSteps = new Set();
  for (let si = steps.length - 1; si >= 0; si--) {
    const st = steps[si];
    if (!st || !Array.isArray(st.slots)) continue;
    const outSlot = st.slots[st.slots.length - 1];
    if (liveSlots.has(outSlot)) {
      liveSteps.add(si);
      const inCount = st.kind === 'binary' ? 2 : 1;
      for (let k = 0; k < inCount; k++) liveSlots.add(st.slots[k]);
    }
  }
  return liveSteps;
}

// Effective affine map of a spec:
// { W, b, inDim, outDim, chain: [layerIndices], byPort: Map<portSlot, W_p> }.
// Throws {code:'REJECT', reason} for out-of-scope specs.
export function effectiveAffine(spec) {
  const reject = (reason) => { throw { code: 'REJECT', reason }; };
  const layers = spec.layers ?? [];
  // Linear layer descriptors, built lazily: only layers referenced by
  // unary steps must be linear; add/sub/concat layers are handled on binary steps.
  const linOf = (li) => {
    const decl = layers[li];
    if (!decl || decl.constructor !== 'linear')
      reject(`unary step references non-linear layer ${li} ('${decl?.constructor}')`);
    const [id, inDim, outDim, bias] = decl.args;
    const flat = (spec.weights ?? {})[String(li)];
    if (!flat) reject(`linear layer ${li} (id ${id}) has no weights in spec`);
    const need = inDim * outDim + (bias ? outDim : 0);
    if (flat.length < need) reject(`linear layer ${li}: weights length ${flat.length} < ${need}`);
    return {
      inDim,
      outDim,
      W: kernelToRows(flat, inDim, outDim),
      b: bias ? flat.slice(inDim * outDim, inDim * outDim + outDim) : new Array(outDim).fill(0),
    };
  };

  const ports = portBySlot(spec);
  const liveSteps = computeLiveStepIndices(spec);
  // slot -> { outDim, byPort: Map<portSlot, rows[][]>, b: number[], primaryOrigin: number }
  // primaryOrigin generalizes A's `origin`: the input slot of the primary dependency; concat requires equal primaryOrigins.
  const aff = new Map();
  const chain = [];

  const getOrInitPortAffine = (slot, expectedDim = null) => {
    if (aff.has(slot)) return aff.get(slot);
    // Fresh slot: it must be a declared input port, else the chain is broken.
    if (!ports.has(slot)) reject(`linear chain reads slot ${slot} with no affine and no input port`);
    const shape = ports.get(slot).shape;
    const inDim = shape[1];
    if (expectedDim !== null && inDim !== expectedDim) {
      reject(`input port dim ${inDim} != layer in_dim ${expectedDim}`);
    }
    const byPort = new Map([[slot, identityRows(inDim)]]);
    return { outDim: inDim, byPort, b: new Array(inDim).fill(0), primaryOrigin: slot };
  };

  const steps = spec.steps ?? [];
  for (let si = 0; si < steps.length; si++) {
    if (!liveSteps.has(si)) continue;
    const step = steps[si];
    if (step.kind === 'unary') {
      const L = linOf(step.layer);
      const [inSlot, outSlot] = step.slots;
      const prev = getOrInitPortAffine(inSlot, L.inDim);
      if (prev.outDim !== L.inDim) {
        reject(`dim mismatch into linear layer ${step.layer}: ${prev.outDim} vs ${L.inDim}`);
      }
      const nextByPort = new Map();
      for (const [pSlot, Wp] of prev.byPort.entries()) {
        nextByPort.set(pSlot, matMulRows(L.W, Wp));
      }
      const nextB = addVec(matVecRows(L.W, prev.b), L.b);
      aff.set(outSlot, {
        outDim: L.outDim,
        byPort: nextByPort,
        b: nextB,
        primaryOrigin: prev.primaryOrigin,
      });
      chain.push(step.layer);
      continue;
    }

    if (step.kind === 'binary') {
      const decl = layers[step.layer];
      const ctor = decl?.constructor;
      const [aSlot, bSlot, outSlot] = step.slots;

      if (ctor === 'add' || ctor === 'sub') {
        const A = getOrInitPortAffine(aSlot, null);
        const B = getOrInitPortAffine(bSlot, null);
        if (A.outDim !== B.outDim) {
          reject(`${ctor} dim mismatch ${A.outDim} vs ${B.outDim}`);
        }
        const outDim = A.outDim;
        const allPorts = new Set([...A.byPort.keys(), ...B.byPort.keys()]);
        const nextByPort = new Map();
        for (const pSlot of allPorts) {
          const pDim = ports.get(pSlot)?.shape?.[1] ?? (A.byPort.get(pSlot)?.[0]?.length ?? B.byPort.get(pSlot)?.[0]?.length);
          const WA = A.byPort.get(pSlot) ?? zeroMat(outDim, pDim);
          const WB = B.byPort.get(pSlot) ?? zeroMat(outDim, pDim);
          nextByPort.set(pSlot, ctor === 'add' ? addMat(WA, WB) : subMat(WA, WB));
        }
        const nextB = ctor === 'add' ? addVec(A.b, B.b) : subVec(A.b, B.b);
        aff.set(outSlot, {
          outDim,
          byPort: nextByPort,
          b: nextB,
          primaryOrigin: A.primaryOrigin ?? B.primaryOrigin,
        });
        chain.push(step.layer);
        continue;
      }

      if (ctor === 'concat') {
        if (decl.args[1] !== 1) {
          reject(`concat dim ${decl.args[1]} unsupported: only dim=1 (feature axis) preserves affinity`);
        }
        const A = aff.get(aSlot);
        const B = aff.get(bSlot);
        if (!A || !B) reject('concat reads slot(s) with no affine map');
        if (A.primaryOrigin !== B.primaryOrigin) {
          reject(`concat of affines from different inputs (slots ${A.primaryOrigin} vs ${B.primaryOrigin})`);
        }
        const allPorts = new Set([...A.byPort.keys(), ...B.byPort.keys()]);
        const nextByPort = new Map();
        for (const pSlot of allPorts) {
          const pDim = ports.get(pSlot)?.shape?.[1] ?? (A.byPort.get(pSlot)?.[0]?.length ?? B.byPort.get(pSlot)?.[0]?.length);
          const WA = A.byPort.get(pSlot) ?? zeroMat(A.outDim, pDim);
          const WB = B.byPort.get(pSlot) ?? zeroMat(B.outDim, pDim);
          if (WA[0].length !== WB[0].length) {
            reject(`concat input dim mismatch ${WA[0].length} vs ${WB[0].length}`);
          }
          nextByPort.set(pSlot, [...WA, ...WB]);
        }
        aff.set(outSlot, {
          outDim: A.outDim + B.outDim,
          byPort: nextByPort,
          b: [...A.b, ...B.b],
          primaryOrigin: A.primaryOrigin,
        });
        chain.push(step.layer);
        continue;
      }

      reject(`binary step on layer ${step.layer} is '${ctor}', only add, sub, and concat are supported`);
    }

    reject(`step kind '${step.kind}' unsupported`);
  }

  const out = aff.get(spec.outputSlot);
  if (!out) reject(`outputSlot ${spec.outputSlot} is not the end of a linear chain`);
  const primaryW = out.byPort.get(out.primaryOrigin) ?? [...out.byPort.values()][0];
  return {
    W: primaryW,
    b: out.b,
    inDim: primaryW[0].length,
    outDim: out.outDim,
    chain,
    byPort: out.byPort,
    primaryOrigin: out.primaryOrigin,
  };
}

// Fold non-shared ports ONLY if:
// (a) Their effective matrix W_p is identically zero (algebraically dead dependency), OR
// (b) The port is explicitly declared with `role === 'context'` AND has matching context inputs.
// Active `observation` or `state` ports with non-zero W_p are NEVER folded into constant bias.
function foldExtraPorts(aff, spec, otherSpec, sharedPorts, tol = 0) {
  let b = [...aff.b];
  const foldedPorts = [];
  const ports = portBySlot(spec);
  const otherPorts = portBySlot(otherSpec);

  for (const [pSlot, Wp] of aff.byPort.entries()) {
    if (sharedPorts.has(pSlot)) continue;
    if (isZeroMat(Wp, tol)) continue;

    const pDef = ports.get(pSlot);
    const otherDef = otherPorts.get(pSlot);
    const isContextRole = pDef?.role === 'context' && (!otherDef || otherDef.role === 'context');
    if (!isContextRole) {
      return {
        ok: false,
        reason: `port ${pSlot} (role '${pDef?.role ?? 'unknown'}') has non-zero linear dependency in one graph but is absent in the other (only role='context' ports may be folded into constant bias)`,
      };
    }

    const inVals = spec.inputs?.[String(pSlot)];
    if (!Array.isArray(inVals) || inVals.length !== Wp[0].length) {
      return {
        ok: false,
        reason: `context port ${pSlot} has no valid inputs array of length ${Wp[0].length} to fold`,
      };
    }
    const otherVals = otherSpec.inputs?.[String(pSlot)];
    if (Array.isArray(otherVals) && otherVals.length === inVals.length) {
      for (let i = 0; i < inVals.length; i++) {
        if (Math.abs(inVals[i] - otherVals[i]) > tol) {
          return {
            ok: false,
            reason: `context port ${pSlot} input value at index ${i} differs between specA (${inVals[i]}) and specB (${otherVals[i]})`,
          };
        }
      }
    }
    b = addVec(b, matVecRows(Wp, inVals));
    foldedPorts.push(pSlot);
  }
  return { ok: true, b, foldedPorts };
}

// Exact comparison: ACCEPT iff same dims and every element matches within tol (default 0 = bit-identical).
export function equivLinear(specA, specB, tol = 0) {
  let A, B;
  try { A = effectiveAffine(specA); }
  catch (e) { return e?.code === 'REJECT' ? { verdict: 'REJECT', reason: `specA: ${e.reason}` } : (() => { throw e; })(); }
  try { B = effectiveAffine(specB); }
  catch (e) { return e?.code === 'REJECT' ? { verdict: 'REJECT', reason: `specB: ${e.reason}` } : (() => { throw e; })(); }

  if (A.outDim !== B.outDim) {
    return { verdict: 'REJECT', reason: `dim mismatch A(${A.inDim}->${A.outDim}) vs B(${B.inDim}->${B.outDim})` };
  }

  const portsA = [...A.byPort.keys()].sort((x, y) => x - y);
  const portsB = [...B.byPort.keys()].sort((x, y) => x - y);
  const sharedPorts = new Set(portsA.filter((p) => B.byPort.has(p)));

  let bA = A.b;
  let bB = B.b;
  let foldedInfo = null;

  if (portsA.join(',') !== portsB.join(',')) {
    if (sharedPorts.size === 0) {
      return { verdict: 'REJECT', reason: `disjoint input ports A([${portsA}]) vs B([${portsB}])` };
    }
    const fA = foldExtraPorts(A, specA, specB, sharedPorts, tol);
    if (!fA.ok) {
      return { verdict: 'REJECT', reason: `active port set mismatch A([${portsA}]) vs B([${portsB}]): ${fA.reason}` };
    }
    const fB = foldExtraPorts(B, specB, specA, sharedPorts, tol);
    if (!fB.ok) {
      return { verdict: 'REJECT', reason: `active port set mismatch A([${portsA}]) vs B([${portsB}]): ${fB.reason}` };
    }
    bA = fA.b;
    bB = fB.b;
    if (fA.foldedPorts.length > 0 || fB.foldedPorts.length > 0) {
      foldedInfo = { folded_in_A: fA.foldedPorts, folded_in_B: fB.foldedPorts };
    }
  }

  for (const pSlot of sharedPorts) {
    const WA = A.byPort.get(pSlot);
    const WB = B.byPort.get(pSlot);
    if (WA[0].length !== WB[0].length) {
      return { verdict: 'REJECT', reason: `dim mismatch on port ${pSlot}: A(${WA[0].length}->${A.outDim}) vs B(${WB[0].length}->${B.outDim})` };
    }
    for (let i = 0; i < A.outDim; i++) {
      for (let k = 0; k < WA[0].length; k++) {
        if (Math.abs(WA[i][k] - WB[i][k]) > tol) {
          return {
            verdict: 'REJECT',
            reason: `W[${i}][${k}] (port ${pSlot}) differs`,
            firstDiff: { kind: 'W', port: pSlot, i, k, a: WA[i][k], b: WB[i][k] },
          };
        }
      }
    }
  }

  for (let i = 0; i < A.outDim; i++) {
    if (Math.abs(bA[i] - bB[i]) > tol) {
      return {
        verdict: 'REJECT',
        reason: `b[${i}] differs`,
        firstDiff: { kind: 'b', i, a: bA[i], b: bB[i] },
      };
    }
  }

  const foldedNote = foldedInfo
    ? ` (conditioned on fixed context port(s) A=[${foldedInfo.folded_in_A.join(',')}], B=[${foldedInfo.folded_in_B.join(',')}])`
    : '';

  return {
    verdict: 'ACCEPT',
    reason: `bit-identical affine map ${A.inDim}->${A.outDim} over ${A.chain.length + B.chain.length} linear/affine steps${foldedNote}`,
    dims: { inDim: A.inDim, outDim: A.outDim },
    active_ports: [...sharedPorts],
    ...(foldedInfo ? { conditional_on_context: true, context_ports_folded: foldedInfo } : {}),
    chainA: A.chain,
    chainB: B.chain,
  };
}
