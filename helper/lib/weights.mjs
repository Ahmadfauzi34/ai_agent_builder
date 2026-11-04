// lib/weights.mjs — explicit helpers for the MANUAL WEIGHTING domain pack (v3.9: rectangular m x n + recursive weight expression trees + 2-step oscillatory-safe spectralRadius + general N x N det_positive).
//
// Supports both square and rectangular (outDim x inDim) matrices in toColumnMajor,
// affineSpec, composeAffine, and evaluateWeightExpr, while keeping spectralRadius/det
// for square matrices.
//
// The engine does not care where weights come from (training vs hand-derived:
// y = Wx either way). This module is the agent-facing workbench for the
// hand-derived side: authoring weight sets in math notation (row-major),
// checking the properties that make a hand-set weight SAFE (stability,
// orthonormality, orientation), blending weight sets for adaptation, and
// fingerprinting the live set. Execution itself goes through the engine
// (bin/weight_lab.mjs); property checks are f64 host-side.
//
// Conventions: W is row-major [[..],[..]] in every function here. The
// column-major engine layout is applied ONLY at the spec boundary
// (toColumnMajor / affineSpec) — never hand-write engine layout.
import { createHash } from 'node:crypto';
import { mulberry32 } from './prng.mjs';

export function matVecMul(W, v) {
  if (W[0].length !== v.length) throw new Error(`matVecMul: W cols ${W[0].length} != v len ${v.length}`);
  return W.map((row) => row.reduce((s, w, j) => s + w * v[j], 0));
}

export function matMul(A, B) {
  const n = A.length, m = B[0].length, k = B.length;
  if (A[0].length !== k) throw new Error('matMul: inner dims mismatch');
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: m }, (_, j) => {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      return s;
    }));
}

// Row-major math notation (outDim rows x inDim cols) -> engine column-major flat array.
export function toColumnMajor(W) {
  const outDim = W.length;
  if (outDim === 0 || !Array.isArray(W[0]) || W[0].length === 0) {
    throw new Error('toColumnMajor: W must be a non-empty 2D matrix');
  }
  const inDim = W[0].length;
  if (!W.every((r) => Array.isArray(r) && r.length === inDim)) {
    throw new Error('toColumnMajor: all rows of W must have equal length');
  }
  const out = [];
  for (let j = 0; j < inDim; j++) for (let i = 0; i < outDim; i++) out.push(W[i][j]);
  return out;
}

// One-shot spec for v_out = W·v + b, supporting both square and rectangular (outDim x inDim) W.
export function affineSpec(W, b, v) {
  const outDim = W.length;
  const inDim = W[0]?.length ?? 0;
  if (!W.every((r) => Array.isArray(r) && r.length === inDim)) {
    throw new Error('affineSpec: all rows of W must have equal length');
  }
  if (b.length !== outDim || v.length !== inDim) {
    throw new Error(`affineSpec: b length (${b.length}) must match outDim (${outDim}) and v length (${v.length}) must match inDim (${inDim})`);
  }
  return {
    layers: [{ constructor: 'linear', args: [100, inDim, outDim, false] }, { constructor: 'add', args: [200] }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    numSlots: 4,
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, inDim, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, outDim, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { 0: toColumnMajor(W) },
    inputs: { 0: v.slice(), 1: b.slice() },
  };
}

// Dominant eigenvalue magnitude via power iteration (f64, host-side).
// For the 'dynamics' kind: rho < 1  =>  the damped/adaptive loop contracts.
// Honest limit: iterative approximation; slow on defective matrices.
// Returns { rho, iters }.
export function spectralRadius(W, iters = 500) {
  const n = W.length;
  if (!W.every((r) => r.length === n)) throw new Error('spectralRadius: W must be square');
  // Probe standard basis vectors plus a deterministic pseudo-random vector so
  // matrices whose dominant eigenvector is orthogonal to [1,...,1] never return a false 0.
  const rng = mulberry32(0x5eed1234);
  const seeds = [
    Array.from({ length: n }, () => 1 / Math.sqrt(n)),
    ...Array.from({ length: Math.min(n, 4) }, (_, k) => Array.from({ length: n }, (__, j) => (j === k ? 1 : 0))),
    (() => {
      const rv = Array.from({ length: n }, () => rng() * 2 - 1);
      const rn = Math.hypot(...rv) || 1;
      return rv.map((x) => x / rn);
    })(),
  ];
  let bestRho = 0;
  let bestIters = 0;
  for (const initV of seeds) {
    let v = initV.slice();
    let rho = 0;
    let usedIters = iters;
    for (let it = 0; it < iters; it++) {
      const w = matVecMul(W, v);
      const norm = Math.hypot(...w);
      if (!(norm > 0)) {
        rho = 0;
        usedIters = it;
        break;
      }
      v = w.map((x) => x / norm);
      rho = norm;
    }
    if (rho > 0) {
      // Evaluate 2-step geometric norm sqrt(||W^2 v||_2) so anti-diagonal / 2-cycle
      // oscillatory matrices (e.g. [[0, 5], [0.05, 0]] with true eigenvalues +-0.5)
      // return their true spectral radius (0.5) instead of a single-step norm spike (5.0).
      const w1 = matVecMul(W, v);
      const w2 = matVecMul(W, w1);
      const norm2 = Math.hypot(...w2);
      rho = norm2 > 0 ? Math.sqrt(norm2) : 0;
    }
    if (rho > bestRho) {
      bestRho = rho;
      bestIters = usedIters;
    }
  }
  return { rho: bestRho, iters: bestIters };
}

export function det2(W) {
  if (W.length !== 2 || W[0].length !== 2) throw new Error('det2: needs 2x2');
  return W[0][0] * W[1][1] - W[0][1] * W[1][0];
}

// Max |row_i·row_j − δij|: 0 for a true rotation/reflection matrix.
export function orthonormalityError(W) {
  const n = W.length;
  const m = W[0]?.length ?? 0;
  if (n === 0 || n !== m || !W.every((r) => r.length === n)) return Infinity;
  let e = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const d = W[i].reduce((s, w, k) => s + w * W[j][k], 0) - (i === j ? 1 : 0);
    e = Math.max(e, Math.abs(d));
  }
  return e;
}

// Adaptation by interpolation: W(a) = (1−a)·A + a·B.
export function blendWeights(A, B, a) {
  if (A.length !== B.length || A[0].length !== B[0].length) throw new Error('blendWeights: dim mismatch');
  return A.map((row, i) => row.map((w, j) => (1 - a) * w + a * B[i][j]));
}

export function blendBias(a, b, t) {
  return a.map((x, i) => (1 - t) * x + t * b[i]);
}

// Which weights are live? sha256 over the f32 bytes of W+b.
// Record alongside every certified set; re-check at deploy/runtime.
export function weightFingerprint(W, b) {
  const h = createHash('sha256');
  h.update(Buffer.from(Float32Array.from([...W.flat(), ...b]).buffer));
  return h.digest('hex');
}

// ---------------------------------------------------------------------------
// Property COMPOSITION — the semantic step: certified properties that survive
// function composition, so a repertoire of certified sets becomes a
// composable vocabulary instead of a bag of isolated sets.
//
// v -> B(A(v)) with A(v) = W_A v + b_A, B(v) = W_B v + b_B fuses to
//   W_C = W_B W_A,  b_C = W_B b_A + b_B   (affine composition, exact).
// The question is which SAFETY properties compose provably — no re-measurement
// of the product needed. Three rules, all provable:
//
//  Rule 1 — norm-bound contraction ('dynamics'):
//    ||W||_F < 1  =>  strict contraction, because
//    ||Wx||_2 <= ||W||_2 ||x||_2 <= ||W||_F ||x||_2 < ||x||_2.
//    And ||W_B W_A||_F <= ||W_A||_F ||W_B||_F (submultiplicativity).
//    So if ||W_A||_F * ||W_B||_F < 1, the composed set is PROVEN a
//    contraction without measuring the product.
//    HONEST LIMIT (documented trap): rho(W_B W_A) <= rho(W_A) rho(W_B) is
//    FALSE in general (counterexample: A=[[0,2],[0,0]], B=[[0,0],[0.4,0]]
//    gives rho(AB)=0.8 > 0*0). The certificate MUST use operator norms,
//    never the spectral-radius product. This is why the bound is written
//    out explicitly instead of "feeling right".
//
//  Rule 2 — orthonormality + orientation ('calibration'):
//    (W_B W_A)(W_B W_A)^T = W_B W_A W_A^T W_B^T = W_B W_B^T = I,
//    so the product of orthonormal matrices is orthonormal; and
//    det(W_C) = det(W_A) det(W_B), so det-positivity composes.
//
//  Rule 3 — measured fallback: when neither rule fires (e.g. rho<1 but
//    ||.||_F >= 1, or mixed kinds), the product is measured DIRECTLY
//    (power-iteration rho / orthonormality) instead of inherited.
//    Refusal of a bound is NOT a rejection — it is "not provable by
//    composition; measure". The certificate says which path was taken.
// (v3.9 gate: Rules 1-2 apply only for square matrices; see isSquare in composeProperties.)
// Exact Frobenius norm.
export function frobeniusNorm(W) {
  return Math.sqrt(W.flat().reduce((s, x) => s + x * x, 0));
}

// det via Gaussian elimination with partial pivoting (f64). Sign is what
// matters for orientation; magnitude degrades with n (fine for 2x2..4x4).
export function det(W) {
  const n = W.length;
  if (!W.every((r) => r.length === n)) throw new Error('det: W must be square');
  const A = W.map((r) => r.slice());
  let d = 1;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
    if (Math.abs(A[p][i]) < 1e-300) return 0;
    if (p !== i) { [A[p], A[i]] = [A[i], A[p]]; d = -d; }
    d *= A[i][i];
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
    }
  }
  return d;
}

// Affine composition B-after-A: y = B.W (A.W x + A.b) + B.b.
// Supports both square and rectangular matrices where outDim(A) == inDim(B).
export function composeAffine(A, B) {
  const outA = A.W.length;
  const inB = B.W[0]?.length ?? 0;
  if (outA !== inB) {
    throw new Error(`composeAffine: inner dim mismatch (A outDim ${outA} != B inDim ${inB})`);
  }
  const W = matMul(B.W, A.W);
  const b = matVecMul(B.W, A.b).map((x, i) => x + B.b[i]);
  return { W, b };
}

// Compose two weight sets with a property certificate.
// Returns { kind, W, b, verdict: 'COMPOSED'|'REJECTED', method,
//           certificate: {rule, bound?, measured?, checks: [...]}, note }.
// method: 'norm_bound' | 'orthonormal' | 'measured' — always stated.
export function composeProperties(A, B) {
  const { W, b } = composeAffine(A, B);
  const isSquare = W.length === W[0].length && A.W.length === A.W[0].length && B.W.length === B.W[0].length;
  const ka = A.kind ?? 'map', kb = B.kind ?? 'map';
  const cert = { checks: [] };
  const push = (name, ok, extra = {}) => cert.checks.push({ name, ok, ...extra });

  // Rule 2: calibration x calibration — orthonormality and det compose.
  if (isSquare && ka === 'calibration' && kb === 'calibration') {
    const tol = Math.min(A.requires?.orthonormal_tol ?? 1e-6, B.requires?.orthonormal_tol ?? 1e-6);
    const eA = orthonormalityError(A.W), eB = orthonormalityError(B.W);
    const preOk = eA <= tol && eB <= tol;
    push('factors_orthonormal', preOk, { err_A: eA, err_B: eB, tol });
    if (!preOk) return { kind: 'map', W, b, verdict: 'REJECTED', method: 'orthonormal', certificate: cert, note: 'factors not orthonormal within tol; no composition claim' };
    const eC = orthonormalityError(W);
    push('product_orthonormal', eC <= tol, { err_product: eC, tol });
    const detPos = (A.requires?.det_positive ?? false) && (B.requires?.det_positive ?? false);
    let dC = null;
    if (detPos) { dC = det(W); push('det_positive_composed', dC > 0, { det_product: dC }); }
    const ok = cert.checks.every((c) => c.ok);
    return {
      kind: 'calibration', W, b,
      verdict: ok ? 'COMPOSED' : 'REJECTED', method: 'orthonormal',
      certificate: { ...cert, rule: '(WB WA)(WB WA)^T = I; det(WB WA) = det(WB)det(WA)', det_product: dC },
      note: 'orthonormality proven by the product identity, not by luck of the numbers',
    };
  }

  // Rule 1: dynamics x dynamics — norm-bound contraction certificate.
  if (isSquare && ka === 'dynamics' && kb === 'dynamics') {
    const nA = frobeniusNorm(A.W), nB = frobeniusNorm(B.W);
    const bound = nA * nB;
    const rhoBound = Math.min(A.requires?.spectral_radius_lt ?? 1, B.requires?.spectral_radius_lt ?? 1);
    const refused = bound >= rhoBound;
    push('norm_product_bound', !refused, { norm_A: nA, norm_B: nB, bound, rho_bound: rhoBound, refused });
    if (!refused) {
      // Consistency positive control: the measured chain must hold —
      // rho(W_C) <= ||W_C||_F <= bound.
      const nC = frobeniusNorm(W);
      const { rho } = spectralRadius(W);
      push('consistency_rho_le_frob_le_bound', rho <= nC + 1e-9 && nC <= bound + 1e-9, { rho_product: rho, frob_product: nC, bound });
      const ok = cert.checks.every((c) => c.ok);
      return {
        kind: 'dynamics', W, b,
        verdict: ok ? 'COMPOSED' : 'REJECTED', method: 'norm_bound',
        certificate: { ...cert, rule: '||WB WA||_F <= ||WA||_F ||WB||_F < 1 => strict contraction' },
        note: 'contraction certified by the norm bound; the product was never power-iterated for the verdict',
      };
    }
    // Rule 3: bound refused -> measure the product directly (honest fallback).
    // The refused bound is recorded but does not sink the verdict.
    const { rho } = spectralRadius(W);
    push('measured_rho', rho < rhoBound, { rho_product: rho, rho_bound: rhoBound });
    const ok = cert.checks.filter((c) => !c.refused).every((c) => c.ok);
    return {
      kind: ok ? 'dynamics' : 'map', W, b,
      verdict: ok ? 'COMPOSED' : 'REJECTED', method: 'measured',
      certificate: { ...cert, rule: 'norm bound refused (>= 1); product measured directly' },
      note: 'bound refusal is not a rejection — the product was measured instead',
    };
  }

  return {
  // Mixed / other kinds: no semantic composition claim; measure or 'map'.
    kind: 'map', W, b, verdict: 'COMPOSED', method: 'measured',
    certificate: { ...cert, rule: 'none — kinds do not compose semantically; affine fusion only' },
    note: 'affine fusion is exact (W_C = W_B W_A, b_C = W_B b_A + b_B); safety must come from probes + gate',
  };
}

// Property checks for a weight set {kind, W, requires}.
// kind 'dynamics'    — adaptive loop weights: need contraction (rho < 1).
// kind 'calibration' — sensor calibration: rotation part must be orthonormal
//                      (and orientation-preserving when requires.det_positive).
// kind 'map'         — general v_out = W·v + b: no intrinsic properties;
//                      safety comes from probes + the gate's jump bound.
export function checkProperties(set) {
  const { kind = 'map', W, requires = {} } = set;
  const out = [];
  if (kind === 'dynamics') {
    const bound = requires.spectral_radius_lt ?? 1;
    const { rho } = spectralRadius(W);
    out.push({ name: 'spectral_radius_lt', ok: rho < bound, value: rho, bound });
  } else if (kind === 'calibration') {
    const tol = requires.orthonormal_tol ?? 1e-6;
    const e = orthonormalityError(W);
    out.push({ name: 'orthonormal', ok: e <= tol, value: e, bound: tol });
    if (requires.det_positive) {
      const isSq = W.length > 0 && W.every((r) => r.length === W.length);
      const d = isSq ? (W.length === 2 ? det2(W) : det(W)) : NaN;
      out.push({ name: 'det_positive', ok: d > 0, value: d, bound: 0 });
    }
  }
  return out;
}

// Recursive Weight Expression Tree Evaluator:
// Evaluates nested composition / blending / scaling expressions over weight sets:
//   - { op: 'ref', name: 'elastic' } (looked up in repertoireMap)
//   - { op: 'literal', W: [...], b: [...], kind?: 'map'|'dynamics'|'calibration' }
//   - { op: 'compose', steps: [expr1, expr2, ...] } or { op: 'compose', first, second }
//   - { op: 'blend', a: exprA, b: exprB, alpha: 0.3 }
//   - { op: 'repeat', expr: innerExpr, depth: N } (or steps: N)
export function evaluateWeightExpr(expr, repertoireMap = new Map()) {
  if (!expr || typeof expr !== 'object') throw new Error('evaluateWeightExpr: expr must be an object');
  if (Array.isArray(expr.W) && Array.isArray(expr.b) && !expr.op) {
    return { ...expr, name: expr.name ?? 'literal', kind: expr.kind ?? 'map', verdict: 'LITERAL' };
  }
  if (expr.op === 'ref') {
    const entry = repertoireMap instanceof Map ? repertoireMap.get(expr.name) : repertoireMap[expr.name];
    if (!entry) throw new Error(`evaluateWeightExpr: unknown weight set '${expr.name}'`);
    return { ...entry, name: entry.name ?? expr.name, kind: entry.kind ?? 'map', verdict: 'CERTIFIED_REF' };
  }
  if (expr.op === 'literal') {
    return { name: expr.name ?? 'literal', kind: expr.kind ?? 'map', W: expr.W, b: expr.b, requires: expr.requires ?? {}, verdict: 'LITERAL' };
  }
  if (expr.op === 'blend') {
    const A = evaluateWeightExpr(expr.a ?? expr.first, repertoireMap);
    const B = evaluateWeightExpr(expr.b ?? expr.second, repertoireMap);
    const alpha = Number(expr.alpha ?? 0.5);
    const W = blendWeights(A.W, B.W, alpha);
    const b = blendBias(A.b, B.b, alpha);
    const kind = A.kind === B.kind ? A.kind : 'map';
    const requires = A.requires ?? B.requires ?? {};
    const props = kind !== 'map' ? checkProperties({ kind, W, requires }) : [];
    const ok = props.every((p) => p.ok);
    return {
      name: expr.name ?? `blend(${A.name ?? 'A'},${B.name ?? 'B'})`,
      kind: ok ? kind : 'map',
      W,
      b,
      verdict: ok ? 'BLENDED' : 'REJECTED',
      properties: props,
    };
  }
  if (expr.op === 'repeat') {
    const depth = Number(expr.depth ?? expr.steps ?? 1);
    if (!Number.isInteger(depth) || depth < 1 || depth > 64) {
      throw new Error(`evaluateWeightExpr: repeat depth must be 1..64, got ${expr.depth ?? expr.steps}`);
    }
    const base = evaluateWeightExpr(expr.expr, repertoireMap);
    let acc = base;
    for (let d = 1; d < depth; d++) {
      acc = composeProperties(acc, base);
      if (acc.verdict === 'REJECTED') break;
    }
    return { ...acc, name: expr.name ?? `repeat(${base.name ?? 'expr'},${depth})` };
  }
  if (expr.op === 'compose') {
    const list = expr.steps ?? expr.chain ?? (expr.first && expr.second ? [expr.first, expr.second] : []);
    if (!Array.isArray(list) || list.length === 0) {
      throw new Error('evaluateWeightExpr: compose requires non-empty steps array (or first/second)');
    }
    let acc = evaluateWeightExpr(list[0], repertoireMap);
    for (let i = 1; i < list.length; i++) {
      const next = evaluateWeightExpr(list[i], repertoireMap);
      acc = composeProperties(acc, next);
      if (acc.verdict === 'REJECTED') break;
    }
    return { ...acc, name: expr.name ?? 'composed_expr' };
  }
  throw new Error(`evaluateWeightExpr: unknown op '${expr.op}'`);
}
