#!/usr/bin/env node
// burn-math arith: deterministic f32 engineering arithmetic & linear algebra
// executed directly through the WASM engine's native numeric kernels
// (WasmNumericKernel, WasmLinearAlgebra, WasmStatistics, WasmReduction, WasmProbability).
//
// Binary vector ops (length-1 broadcasts):
//   node bin/arith.mjs <add|sub|mul|div|dot|dist|cos|xent|kl> <a1,a2,...> <b1,b2,...>
// Unary vector & statistical ops:
//   node bin/arith.mjs <sqrt|exp|log|abs|norm|normalize|entropy|sum|mean|min|max|var|std> <a1,a2,...>
// Parameterized clamp:
//   node bin/arith.mjs clamp <a1,a2,...> <min>,<max>
// Matrix ops (rows separated by ';'):
//   node bin/arith.mjs matmul <a11,a12;...> <b11,b12;...>
//   node bin/arith.mjs <sum_axis|mean_axis|min_axis|max_axis> <a11,a12;...> <0|1>
// Batch mode (one engine load for N ops):
//   node bin/arith.mjs --batch <ops.json>
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm } from '../lib/wasm_direct.mjs';
import { isFiniteF32 } from '../lib/spec.mjs';
import {
  assertAllFiniteOutput,
  engineProvenance,
  matrixQuantizationWarnings,
  parseMatrix,
  parseMatrixValues,
  parseVector,
  parseVectorValues,
  quantizationWarnings,
} from '../lib/cli.mjs';
import fs from 'node:fs';

const BINARY_OPS = new Set(['add', 'sub', 'mul', 'div', 'dot', 'dist', 'cos', 'xent', 'kl']);
const UNARY_OPS = new Set([
  'sqrt', 'exp', 'log', 'abs',
  'norm', 'normalize', 'entropy',
  'sum', 'mean', 'min', 'max', 'var', 'std',
]);
const AXIS_OPS = new Set(['sum_axis', 'mean_axis', 'min_axis', 'max_axis']);

const USAGE =
  'usage: node bin/arith.mjs <add|sub|mul|div|dot|dist|cos|xent|kl> <a1,a2,...> <b1,b2,...> ' +
  'OR <sqrt|exp|log|abs|norm|normalize|entropy|sum|mean|min|max|var|std> <a1,a2,...> ' +
  'OR clamp <a1,a2,...> <min>,<max> ' +
  'OR matmul <a11,a12;...> <b11,b12;...> ' +
  'OR <sum_axis|mean_axis|min_axis|max_axis> <a11,a12;...> <0|1> ' +
  'OR --batch <ops.json>';

function prepareBinary(op, a, b) {
  let aEff = a, bEff = b, broadcast = null;
  if (a.length !== b.length) {
    if (a.length === 1) { aEff = new Array(b.length).fill(a[0]); broadcast = 'a'; }
    else if (b.length === 1) { bEff = new Array(a.length).fill(b[0]); broadcast = 'b'; }
    else throw new Error(`length mismatch: a has ${a.length} values, b has ${b.length} (binary ops require exactly matching shapes)`);
  }
  if (op === 'div') {
    for (let i = 0; i < bEff.length; i++) {
      const v = bEff[i];
      if (v === 0) throw new Error(`division by zero at index ${i} (b[${i}] is 0)`);
      if (Math.fround(v) === 0) {
        throw new Error(`divisor underflow at index ${i} (b[${i}] is ${v}: its reciprocal is not finite; in f32 it is zero)`);
      }
    }
  }
  return { aEff, bEff, broadcast };
}

function execBinaryKernel(rt, op, aEff, bEff) {
  const n = aEff.length;
  const shape = Uint32Array.from([1, n, 1, 1]);
  const tA = new rt.WasmTensor(Float32Array.from(aEff), shape);
  const tB = new rt.WasmTensor(Float32Array.from(bEff), shape);
  let nk = null, la = null, pb = null, out = null;
  try {
    if (op === 'add' || op === 'sub' || op === 'mul' || op === 'div') {
      nk = new rt.WasmNumericKernel();
      out = nk[op](tA, tB);
    } else if (op === 'dot') {
      la = new rt.WasmLinearAlgebra();
      out = la.dot(tA, tB);
    } else if (op === 'dist') {
      la = new rt.WasmLinearAlgebra();
      out = la.l2Distance(tA, tB);
    } else if (op === 'cos') {
      la = new rt.WasmLinearAlgebra();
      out = la.cosineSimilarity(tA, tB, 1e-8);
    } else if (op === 'xent') {
      pb = new rt.WasmProbability();
      out = pb.crossEntropy(tA, tB);
    } else if (op === 'kl') {
      pb = new rt.WasmProbability();
      out = pb.klDivergence(tA, tB);
    } else {
      throw new Error(`unknown binary op: ${op}`);
    }
    return assertAllFiniteOutput(op, Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { nk?.free?.(); } catch { /* ignore */ }
    try { la?.free?.(); } catch { /* ignore */ }
    try { pb?.free?.(); } catch { /* ignore */ }
    try { tB.free(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function execUnaryKernel(rt, op, a, extra = {}) {
  const n = a.length;
  const shape = Uint32Array.from([1, n, 1, 1]);
  const tA = new rt.WasmTensor(Float32Array.from(a), shape);
  let nk = null, la = null, st = null, pb = null, out = null;
  try {
    if (op === 'sqrt' || op === 'exp' || op === 'log' || op === 'abs') {
      nk = new rt.WasmNumericKernel();
      out = nk[op](tA);
    } else if (op === 'clamp') {
      const { min, max } = extra;
      if (!isFiniteF32(min) || !isFiniteF32(max) || min > max) {
        throw new Error(`clamp requires finite f32 min <= max, got [${min}, ${max}]`);
      }
      nk = new rt.WasmNumericKernel();
      out = nk.clamp(tA, min, max);
    } else if (op === 'norm') {
      la = new rt.WasmLinearAlgebra();
      out = la.l2Norm(tA);
    } else if (op === 'normalize') {
      pb = new rt.WasmProbability();
      out = pb.normalize(tA);
    } else if (op === 'entropy') {
      pb = new rt.WasmProbability();
      out = pb.entropy(tA);
    } else if (op === 'sum' || op === 'mean' || op === 'min' || op === 'max') {
      st = new rt.WasmStatistics();
      out = st[op](tA);
    } else if (op === 'var') {
      st = new rt.WasmStatistics();
      out = st.variancePopulation(tA);
    } else if (op === 'std') {
      st = new rt.WasmStatistics();
      out = st.stdPopulation(tA);
    } else {
      throw new Error(`unknown unary op: ${op}`);
    }
    return assertAllFiniteOutput(op, Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { nk?.free?.(); } catch { /* ignore */ }
    try { la?.free?.(); } catch { /* ignore */ }
    try { st?.free?.(); } catch { /* ignore */ }
    try { pb?.free?.(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function prepareMatmul(A, B) {
  const M = A.length, K = A[0].length, K2 = B.length, N = B[0].length;
  if (K !== K2) throw new Error(`matmul shape mismatch: a is ${M}x${K}, b is ${K2}x${N} (inner dimensions must match)`);
  return { M, K, N };
}

function execMatmulKernel(rt, A, B, M, K, N) {
  const tA = new rt.WasmTensor(Float32Array.from(A.flat()), Uint32Array.from([1, 1, M, K]));
  const tB = new rt.WasmTensor(Float32Array.from(B.flat()), Uint32Array.from([1, 1, K, N]));
  let la = null, out = null;
  try {
    la = new rt.WasmLinearAlgebra();
    out = la.matmul(tA, tB);
    return assertAllFiniteOutput('matmul', Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { la?.free?.(); } catch { /* ignore */ }
    try { tB.free(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function execAxisKernel(rt, op, A, axis2d) {
  if (axis2d !== 0 && axis2d !== 1) {
    throw new Error(`${op}: matrix axis must be 0 (reduce rows) or 1 (reduce cols), got ${JSON.stringify(axis2d)}`);
  }
  const M = A.length, N = A[0].length;
  const wasmAxis = axis2d + 2; // [1, 1, M, N] -> axis 2 is rows (M), axis 3 is cols (N)
  const tA = new rt.WasmTensor(Float32Array.from(A.flat()), Uint32Array.from([1, 1, M, N]));
  let rd = null, out = null;
  try {
    rd = new rt.WasmReduction();
    const methodMap = {
      sum_axis: 'sumAxis',
      mean_axis: 'meanAxis',
      min_axis: 'minAxis',
      max_axis: 'maxAxis',
    };
    out = rd[methodMap[op]](tA, wasmAxis);
    const values = assertAllFiniteOutput(op, Array.from(out.to_array()));
    const outShape = axis2d === 0 ? [1, N] : [M, 1];
    return { values, shape: outShape };
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { rd?.free?.(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

async function runBatch(file) {
  let text;
  try { text = fs.readFileSync(file, 'utf8'); }
  catch (e) { throw new Error(`cannot read batch file ${file}: ${e.message}`); }
  let doc;
  try { doc = JSON.parse(text); }
  catch (e) { throw new Error(`invalid batch JSON in ${file}: ${e.message}`); }
  if (doc === null || typeof doc !== 'object' || Array.isArray(doc)) throw new Error('batch must be an object with an ops array');
  if (!Array.isArray(doc.ops)) throw new Error('batch.ops must be an array');

  const rt = await loadWasm();
  const engine = engineProvenance();
  const results = [];
  for (let i = 0; i < doc.ops.length; i++) {
    const entry = doc.ops[i];
    try {
      if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new Error(`op entry ${i} must be an object with op, a, ...`);
      }
      const { op } = entry;
      if (op === 'matmul') {
        const A = parseMatrixValues(entry.a, 'a');
        const B = parseMatrixValues(entry.b, 'b');
        const { M, K, N } = prepareMatmul(A, B);
        const values = execMatmulKernel(rt, A, B, M, K, N);
        results.push({ index: i, ok: true, op, a: A, b: B, shape: [M, N], broadcast: null, values, precision_warnings: [...matrixQuantizationWarnings(A, 'a'), ...matrixQuantizationWarnings(B, 'b')] });
      } else if (AXIS_OPS.has(op)) {
        const A = parseMatrixValues(entry.a, 'a');
        const axis = Number(entry.axis ?? entry.b);
        const { values, shape } = execAxisKernel(rt, op, A, axis);
        results.push({ index: i, ok: true, op, a: A, axis, shape, values, precision_warnings: matrixQuantizationWarnings(A, 'a') });
      } else if (BINARY_OPS.has(op)) {
        const a = parseVectorValues(entry.a, 'a');
        const b = parseVectorValues(entry.b, 'b', op === 'div');
        const { aEff, bEff, broadcast } = prepareBinary(op, a, b);
        const values = execBinaryKernel(rt, op, aEff, bEff);
        results.push({ index: i, ok: true, op, a: aEff, b: bEff, broadcast, values, precision_warnings: [...quantizationWarnings(aEff, 'a'), ...quantizationWarnings(bEff, 'b')] });
      } else if (op === 'clamp') {
        const a = parseVectorValues(entry.a, 'a');
        const min = entry.min ?? (Array.isArray(entry.b) ? entry.b[0] : undefined);
        const max = entry.max ?? (Array.isArray(entry.b) ? entry.b[1] : undefined);
        const values = execUnaryKernel(rt, 'clamp', a, { min: Number(min), max: Number(max) });
        results.push({ index: i, ok: true, op, a, min: Number(min), max: Number(max), values, precision_warnings: quantizationWarnings(a, 'a') });
      } else if (UNARY_OPS.has(op)) {
        const a = parseVectorValues(entry.a, 'a');
        const values = execUnaryKernel(rt, op, a);
        results.push({ index: i, ok: true, op, a, values, precision_warnings: quantizationWarnings(a, 'a') });
      } else {
        throw new Error(`unknown op: ${JSON.stringify(op)}`);
      }
    } catch (e) {
      results.push({ index: i, ok: false, op: entry?.op ?? null, error: e instanceof Error ? e.message : String(e) });
    }
  }
  const allOk = results.every((r) => r.ok);
  printJsonAndExit({ ok: allOk, engine, results }, allOk ? 0 : 2);
}

async function main() {
  const argv = process.argv.slice(2);
  if (!argv.length || argv.includes('--help') || argv.includes('-h')) {
    console.error(USAGE);
    process.exit(2);
  }
  const [op, aText, bText, extra] = argv;
  if (op === '--batch') {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    return runBatch(aText);
  }
  if (op === 'matmul') {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    const A = parseMatrix(aText, 'a', USAGE);
    const B = parseMatrix(bText, 'b', USAGE);
    const { M, K, N } = prepareMatmul(A, B);
    const rt = await loadWasm();
    const values = execMatmulKernel(rt, A, B, M, K, N);
    printJsonAndExit({ ok: true, op: 'matmul', a: A, b: B, shape: [M, N], broadcast: null, engine: engineProvenance(), values, precision_warnings: [...matrixQuantizationWarnings(A, 'a'), ...matrixQuantizationWarnings(B, 'b')] }, 0);
    return;
  }
  if (AXIS_OPS.has(op)) {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    const A = parseMatrix(aText, 'a', USAGE);
    const axis = Number(bText);
    const rt = await loadWasm();
    const { values, shape } = execAxisKernel(rt, op, A, axis);
    printJsonAndExit({ ok: true, op, a: A, axis, shape, engine: engineProvenance(), values, precision_warnings: matrixQuantizationWarnings(A, 'a') }, 0);
    return;
  }
  if (op === 'clamp') {
    if (aText === undefined || bText === undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const bounds = extra !== undefined ? [Number(bText), Number(extra)] : parseVector(bText, 'bounds', USAGE);
    if (bounds.length !== 2) throw new Error('clamp bounds must provide min,max');
    const rt = await loadWasm();
    const values = execUnaryKernel(rt, 'clamp', a, { min: bounds[0], max: bounds[1] });
    printJsonAndExit({ ok: true, op: 'clamp', a, min: bounds[0], max: bounds[1], engine: engineProvenance(), values, precision_warnings: quantizationWarnings(a, 'a') }, 0);
    return;
  }
  if (UNARY_OPS.has(op)) {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const rt = await loadWasm();
    const values = execUnaryKernel(rt, op, a);
    printJsonAndExit({ ok: true, op, a, engine: engineProvenance(), values, precision_warnings: quantizationWarnings(a, 'a') }, 0);
    return;
  }
  if (BINARY_OPS.has(op)) {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const b = parseVector(bText, 'b', USAGE, op === 'div');
    const { aEff, bEff, broadcast } = prepareBinary(op, a, b);
    const rt = await loadWasm();
    const values = execBinaryKernel(rt, op, aEff, bEff);
    printJsonAndExit({ ok: true, op, a: aEff, b: bEff, broadcast, engine: engineProvenance(), values, precision_warnings: [...quantizationWarnings(aEff, 'a'), ...quantizationWarnings(bEff, 'b')] }, 0);
    return;
  }
  throw new Error(USAGE);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
