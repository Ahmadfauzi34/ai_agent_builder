#!/usr/bin/env node
// burn-math kernel: the Lane E WasmNumericKernel elementwise kernels,
// exposed as a CLI tool with --batch and $prev / $<index> register chaining.
//
//   node bin/kernel.mjs <sin|cos|tan|floor|ceil|round> <a1,a2,...>
//   node bin/kernel.mjs <atan2|powf|hypot> <a1,a2,...> <b1,b2,...>
//   node bin/kernel.mjs fma <a1,a2,...> <b1,b2,...> <c1,c2,...>
//   node bin/kernel.mjs --batch <ops.json>
//
// Argument order is the engine's, and it is NOT symmetric where it
// matters — read it before calling:
//   atan2 <y...> <x...>:  y FIRST, like math atan2(y, x). atan2(1, 0)
//     is pi/2; atan2(0, 1) is 0. Swapping them silently gives a wrong
//     angle, so the order is pinned by a regression case, not just docs.
//   powf <base...> <exp...>, hypot <x...> <y...>,
//   fma <a...> <b...> <c...>:  a*b+c per element via f32::mul_add.
//     On wasm32 without a scalar fma instruction this lowers to mul+add
//     (two roundings) — fused execution is NOT claimed.
//
// Traps an agent must not fall into (each pinned by a regression case):
//   - kernel cos is the ELEMENTWISE cosine (radians). arith's `cos` is
//     the cosine SIMILARITY of two vectors — a different operation with
//     the same short name. This tool never computes similarity.
//   - round is Burn's round-half-to-EVEN (2.5 -> 2, 3.5 -> 4), NOT
//     JS Math.round (half up) and NOT Rust f32::round (half away).
//   - NO broadcast: every input vector must have exactly the same
//     length. arith broadcasts length-1 vectors; this tool does not,
//     because the kernels themselves require same-shape inputs and a
//     silent broadcast here would mask a shape error.
//   - Domain failures fail honestly through the engine (exit 2):
//     powf with a negative base and fractional exponent -> NaN,
//     0 to a negative power -> inf, tan poles and overflowing results
//     -> inf; all fail output validation instead of returning a value.
//     The CLI adds no domain check of its own beyond finite-f32 input
//     parsing, so the engine's message is the error, unedited.
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm } from '../lib/wasm_direct.mjs';
import {
  assertAllFiniteOutput,
  engineProvenance,
  parseVector,
  parseVectorValues,
  quantizationWarnings,
} from '../lib/cli.mjs';
import fs from 'node:fs';

const UNARY_OPS = new Set(['sin', 'cos', 'tan', 'floor', 'ceil', 'round']);
const BINARY_OPS = new Set(['atan2', 'powf', 'hypot']);
const TERNARY_OPS = new Set(['fma']);

const USAGE =
  'usage: node bin/kernel.mjs <sin|cos|tan|floor|ceil|round> <a1,a2,...> ' +
  'OR <atan2|powf|hypot> <a1,a2,...> <b1,b2,...> ' +
  '(atan2 takes y first, then x; powf takes base, then exp) ' +
  'OR fma <a1,a2,...> <b1,b2,...> <c1,c2,...> (a*b+c per element) ' +
  'OR --batch <ops.json>. ' +
  'All input vectors must have exactly the same length (no broadcast).';

function vectorTensor(rt, values) {
  return new rt.WasmTensor(
    Float32Array.from(values),
    Uint32Array.from([1, values.length, 1, 1]),
  );
}

function execUnaryKernel(rt, op, a) {
  const tA = vectorTensor(rt, a);
  let nk = null, out = null;
  try {
    nk = new rt.WasmNumericKernel();
    out = nk[op](tA);
    return assertAllFiniteOutput(op, Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { nk?.free?.(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function assertSameLength(op, names, vectors) {
  const n = vectors[0].length;
  for (let i = 1; i < vectors.length; i++) {
    if (vectors[i].length !== n) {
      throw new Error(
        `length mismatch: ${names[0]} has ${n} values, ${names[i]} has ${vectors[i].length} values ` +
        `(kernel ops require exactly matching shapes; no broadcast)`,
      );
    }
  }
}

function execBinaryKernel(rt, op, a, b) {
  assertSameLength(op, ['a', 'b'], [a, b]);
  const tA = vectorTensor(rt, a);
  const tB = vectorTensor(rt, b);
  let nk = null, out = null;
  try {
    nk = new rt.WasmNumericKernel();
    out = nk[op](tA, tB);
    return assertAllFiniteOutput(op, Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { nk?.free?.(); } catch { /* ignore */ }
    try { tB.free(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function execTernaryKernel(rt, op, a, b, c) {
  assertSameLength(op, ['a', 'b', 'c'], [a, b, c]);
  const tA = vectorTensor(rt, a);
  const tB = vectorTensor(rt, b);
  const tC = vectorTensor(rt, c);
  let nk = null, out = null;
  try {
    nk = new rt.WasmNumericKernel();
    out = nk[op](tA, tB, tC);
    return assertAllFiniteOutput(op, Array.from(out.to_array()));
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { nk?.free?.(); } catch { /* ignore */ }
    try { tC.free(); } catch { /* ignore */ }
    try { tB.free(); } catch { /* ignore */ }
    try { tA.free(); } catch { /* ignore */ }
  }
}

function resolveBatchRef(argVal, curIndex, results) {
  if (typeof argVal !== 'string' || !argVal.startsWith('$')) return argVal;
  let refIdx = -1;
  if (argVal === '$prev') refIdx = curIndex - 1;
  else if (/^\$(?:0|[1-9]\d*)$/.test(argVal)) refIdx = Number(argVal.slice(1));
  else throw new Error(`invalid batch register reference ${JSON.stringify(argVal)} (use "$prev" or "$<stepIndex>")`);

  if (refIdx < 0 || refIdx >= curIndex) {
    throw new Error(`batch reference ${argVal} at step ${curIndex} out of bounds (must reference an earlier step 0..${curIndex - 1})`);
  }
  const prev = results[refIdx];
  if (!prev || !prev.ok || !Array.isArray(prev.values)) {
    throw new Error(`batch reference ${argVal} at step ${curIndex} points to failed step ${refIdx}`);
  }
  return prev.values;
}

async function runBatch(file) {
  let text;
  try { text = fs.readFileSync(file, 'utf8'); }
  catch (e) { throw new Error(`cannot read batch file ${file}: ${e.message}`); }
  let doc;
  try { doc = JSON.parse(text); }
  catch (e) { throw new Error(`invalid batch JSON in ${file}: ${e.message}`); }
  if (doc === null || typeof doc !== 'object' || Array.isArray(doc) || !Array.isArray(doc.ops)) {
    throw new Error('batch must be an object with an ops array');
  }
  const rt = await loadWasm();
  const engine = engineProvenance();
  const results = [];
  for (let i = 0; i < doc.ops.length; i++) {
    const entry = doc.ops[i];
    try {
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
        throw new Error(`op entry ${i} must be an object`);
      }
      const { op } = entry;
      if (UNARY_OPS.has(op)) {
        const a = parseVectorValues(resolveBatchRef(entry.a, i, results), 'a');
        const values = execUnaryKernel(rt, op, a);
        results.push({ index: i, ok: true, op, a, broadcast: null, values, precision_warnings: quantizationWarnings(a, 'a') });
      } else if (BINARY_OPS.has(op)) {
        const a = parseVectorValues(resolveBatchRef(entry.a, i, results), 'a');
        const b = parseVectorValues(resolveBatchRef(entry.b, i, results), 'b');
        const values = execBinaryKernel(rt, op, a, b);
        results.push({ index: i, ok: true, op, a, b, broadcast: null, values, precision_warnings: [...quantizationWarnings(a, 'a'), ...quantizationWarnings(b, 'b')] });
      } else if (TERNARY_OPS.has(op)) {
        const a = parseVectorValues(resolveBatchRef(entry.a, i, results), 'a');
        const b = parseVectorValues(resolveBatchRef(entry.b, i, results), 'b');
        const c = parseVectorValues(resolveBatchRef(entry.c, i, results), 'c');
        const values = execTernaryKernel(rt, op, a, b, c);
        results.push({ index: i, ok: true, op, a, b, c, broadcast: null, values, precision_warnings: [...quantizationWarnings(a, 'a'), ...quantizationWarnings(b, 'b'), ...quantizationWarnings(c, 'c')] });
      } else {
        throw new Error(`unknown kernel op: ${JSON.stringify(op)}`);
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
  const [op, aText, bText, cText, extra] = argv;
  if (op === '--batch') {
    if (aText === undefined || bText !== undefined || cText !== undefined || extra !== undefined) throw new Error(USAGE);
    return runBatch(aText);
  }
  if (UNARY_OPS.has(op)) {
    if (aText === undefined || bText !== undefined || cText !== undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const rt = await loadWasm();
    const values = execUnaryKernel(rt, op, a);
    printJsonAndExit({ ok: true, op, a, broadcast: null, engine: engineProvenance(), values, precision_warnings: quantizationWarnings(a, 'a') }, 0);
    return;
  }
  if (BINARY_OPS.has(op)) {
    if (aText === undefined || bText === undefined || cText !== undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const b = parseVector(bText, 'b', USAGE);
    const rt = await loadWasm();
    const values = execBinaryKernel(rt, op, a, b);
    printJsonAndExit({ ok: true, op, a, b, broadcast: null, engine: engineProvenance(), values, precision_warnings: [...quantizationWarnings(a, 'a'), ...quantizationWarnings(b, 'b')] }, 0);
    return;
  }
  if (TERNARY_OPS.has(op)) {
    if (aText === undefined || bText === undefined || cText === undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a', USAGE);
    const b = parseVector(bText, 'b', USAGE);
    const c = parseVector(cText, 'c', USAGE);
    const rt = await loadWasm();
    const values = execTernaryKernel(rt, op, a, b, c);
    printJsonAndExit({ ok: true, op, a, b, c, broadcast: null, engine: engineProvenance(), values, precision_warnings: [...quantizationWarnings(a, 'a'), ...quantizationWarnings(b, 'b'), ...quantizationWarnings(c, 'c')] }, 0);
    return;
  }
  throw new Error(USAGE);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
