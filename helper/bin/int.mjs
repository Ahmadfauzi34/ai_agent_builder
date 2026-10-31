#!/usr/bin/env node
// burn-math int: exact i64 arithmetic & casts through the engine's integer world,
// with fail-closed i64 overflow detection (or explicit --wrap for two's-complement).
//
//   node bin/int.mjs add 16777217,100 1,-30   -> {"ok":true,"op":"add","values":["16777218","70"]}
//   node bin/int.mjs sub 10,-5 3,7            -> {"ok":true,"op":"sub","values":["7","-12"]}
//   node bin/int.mjs mul 6,-4 7,8             -> {"ok":true,"op":"mul","values":["42","-32"]}
//   node bin/int.mjs matmul "1,2;3,4" "5,6;7,8"
//        -> {"ok":true,"op":"matmul","shape":[2,2],"values":["19","22","43","50"]}
//   node bin/int.mjs to-f32 100,-200          -> {"ok":true,"op":"to-f32","values":[100,-200]}
//   node bin/int.mjs from-f32 3.9,-4.7,0      -> {"ok":true,"op":"from-f32","values":["3","-4","0"]}
//   node bin/int.mjs --batch ops.json         -> batch execution in one engine load
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm } from '../lib/wasm_direct.mjs';
import { isFiniteF32 } from '../lib/spec.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

function engineProvenance() {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'MANIFEST.json'), 'utf8'));
    const full = m?.sha256?.['burn_research_bg.wasm'] ?? m?.engine?.wasm_sha256 ?? null;
    return { wasm_sha256_short: full ? full.slice(0, 16) : null, mode: 'direct' };
  } catch { return { wasm_sha256_short: null, mode: 'direct' }; }
}

const OPS = { add: 'intAdd', sub: 'intSub', mul: 'intMul' };
const USAGE =
  'usage: node bin/int.mjs <add|sub|mul> <a1,a2,...> <b1,b2,...> [--wrap] ' +
  'OR matmul <a11,a12;...a21,...> <b11,b12;...b21,...> [--wrap] ' +
  'OR to-f32 <a1,a2,...> [--allow-inexact] ' +
  'OR from-f32 <f1,f2,...> ' +
  'OR --batch <ops.json>';

const I64_MIN = -(2n ** 63n);
const I64_MAX = 2n ** 63n - 1n;
const F32_EXACT_INT_MAX = 16777216n; // 2^24
const INT_RE = /^-?\d+$/;

function parseBigIntToken(t, name) {
  const s = String(t).trim();
  if (!INT_RE.test(s)) throw new Error(`${name}: not an integer: ${JSON.stringify(t)}`);
  return BigInt(s);
}

function checkI64Range(values, name) {
  for (let i = 0; i < values.length; i++) {
    if (values[i] < I64_MIN || values[i] > I64_MAX) {
      throw new Error(`${name} value out of i64 range at index ${i}: ${values[i]}`);
    }
  }
}

function parseVector(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const values = text.split(',').map((t) => parseBigIntToken(t, name));
  if (values.length === 0) throw new Error(`${name}: empty vector`);
  return values;
}

function parseFloatVector(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const values = text.split(',').map((t) => {
    const s = t.trim();
    const v = Number(s);
    if (s === '' || !Number.isFinite(v) || !isFiniteF32(v)) {
      throw new Error(`${name}: not a finite f32 number: ${JSON.stringify(t)}`);
    }
    if (Math.trunc(v) < Number(I64_MIN) || Math.trunc(v) >= 9223372036854775808) {
      throw new Error(`${name}: float value ${v} exceeds i64 range after truncation`);
    }
    return v;
  });
  if (values.length === 0) throw new Error(`${name}: empty vector`);
  return values;
}

function parseMatrix(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const rows = text.split(';').map((row, ri) => {
    const vals = row.split(',').map((t) => parseBigIntToken(t, `${name} row ${ri}`));
    if (vals.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    return vals;
  });
  if (rows.length === 0) throw new Error(`${name}: empty matrix`);
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) {
    throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  }
  return rows;
}

function prepareElementwise(a, b) {
  let aEff = a, bEff = b, broadcast = null;
  if (a.length !== b.length) {
    if (a.length === 1) { aEff = new Array(b.length).fill(a[0]); broadcast = 'a'; }
    else if (b.length === 1) { bEff = new Array(a.length).fill(b[0]); broadcast = 'b'; }
    else throw new Error(`length mismatch: a has ${a.length} values, b has ${b.length} (binary ops require exactly matching shapes)`);
  }
  return { aEff, bEff, broadcast };
}

function prepareMatmul(A, B) {
  const M = A.length, K = A[0].length, K2 = B.length, N = B[0].length;
  if (K !== K2) throw new Error(`matmul shape mismatch: a is ${M}x${K}, b is ${K2}x${N} (inner dimensions must match)`);
  return { M, K, N };
}

function makeIntTensor(rt, values, shape) {
  return new rt.WasmIntTensor(new BigInt64Array(values), new Uint32Array(shape));
}

function verifyBinaryNoOverflow(op, aEff, bEff, allowWrap) {
  if (allowWrap) return;
  for (let i = 0; i < aEff.length; i++) {
    const x = aEff[i], y = bEff[i];
    const exact = op === 'add' ? x + y : op === 'sub' ? x - y : x * y;
    if (exact < I64_MIN || exact > I64_MAX) {
      throw new Error(`i64 overflow at index ${i}: exact ${op} result ${exact} is out of i64 range (pass --wrap for two's-complement wraparound)`);
    }
  }
}

function verifyMatmulNoOverflow(A, B, M, K, N, allowWrap) {
  if (allowWrap) return;
  for (let r = 0; r < M; r++) {
    for (let c = 0; c < N; c++) {
      let acc = 0n;
      for (let k = 0; k < K; k++) {
        const prod = A[r][k] * B[k][c];
        if (prod < I64_MIN || prod > I64_MAX) {
          throw new Error(`i64 overflow in matmul product at [${r},${c}] (k=${k}): ${prod} is out of i64 range`);
        }
        acc += prod;
        if (acc < I64_MIN || acc > I64_MAX) {
          throw new Error(`i64 overflow in matmul accumulator at [${r},${c}]: ${acc} is out of i64 range`);
        }
      }
    }
  }
}

function runIntBinary(rt, op, aEff, bEff, allowWrap = false) {
  verifyBinaryNoOverflow(op, aEff, bEff, allowWrap);
  const ctor = OPS[op];
  const n = aEff.length;
  const reg = new rt.LayerRegistry();
  let a = null, b = null, out = null;
  try {
    reg.initAgentLayer(rt.AgentLayerSpec[ctor](1));
    a = makeIntTensor(rt, aEff, [1, n, 1, 1]);
    b = makeIntTensor(rt, bEff, [1, n, 1, 1]);
    out = reg.forwardIntBinary(1, a, b);
    return Array.from(out.to_array(), String);
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { b?.free?.(); } catch { /* ignore */ }
    try { a?.free?.(); } catch { /* ignore */ }
    try { reg.free(); } catch { /* ignore */ }
  }
}

function runIntMatmul(rt, A, B, M, K, N, allowWrap = false) {
  verifyMatmulNoOverflow(A, B, M, K, N, allowWrap);
  const reg = new rt.LayerRegistry();
  let a = null, b = null, out = null;
  try {
    reg.initAgentLayer(rt.AgentLayerSpec.intMatmul(1));
    a = makeIntTensor(rt, A.flat(), [1, 1, M, K]);
    b = makeIntTensor(rt, B.flat(), [1, 1, K, N]);
    out = reg.forwardIntBinary(1, a, b);
    return Array.from(out.to_array(), String);
  } finally {
    try { out?.free?.(); } catch { /* ignore */ }
    try { b?.free?.(); } catch { /* ignore */ }
    try { a?.free?.(); } catch { /* ignore */ }
    try { reg.free(); } catch { /* ignore */ }
  }
}

function runIntToFloat(rt, a, allowInexact = false) {
  if (!allowInexact) {
    for (let i = 0; i < a.length; i++) {
      if (a[i] < -F32_EXACT_INT_MAX || a[i] > F32_EXACT_INT_MAX) {
        throw new Error(`to-f32 precision loss at index ${i}: ${a[i]} exceeds exact f32 integer range [-16777216, 16777216] (pass --allow-inexact to round)`);
      }
    }
  }
  const n = a.length;
  const reg = new rt.LayerRegistry();
  let ti = null, tf = null;
  try {
    reg.initAgentLayer(rt.AgentLayerSpec.intToFloat(1));
    ti = makeIntTensor(rt, a, [1, n, 1, 1]);
    tf = reg.forwardIntCastToFloat(1, ti);
    return Array.from(tf.to_array());
  } finally {
    try { tf?.free?.(); } catch { /* ignore */ }
    try { ti?.free?.(); } catch { /* ignore */ }
    try { reg.free(); } catch { /* ignore */ }
  }
}

function runIntFromFloat(rt, floats) {
  const n = floats.length;
  const reg = new rt.LayerRegistry();
  let tf = null, ti = null;
  try {
    reg.initAgentLayer(rt.AgentLayerSpec.intFromFloat(1));
    tf = new rt.WasmTensor(Float32Array.from(floats), Uint32Array.from([1, n, 1, 1]));
    ti = reg.forwardIntCastFromFloat(1, tf);
    return Array.from(ti.to_array(), String);
  } finally {
    try { ti?.free?.(); } catch { /* ignore */ }
    try { tf?.free?.(); } catch { /* ignore */ }
    try { reg.free(); } catch { /* ignore */ }
  }
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
      const { op, wrap = false, allowInexact = false } = entry;
      if (op === 'matmul') {
        const A = entry.a.map((r, ri) => r.map((v) => parseBigIntToken(v, `a row ${ri}`)));
        const B = entry.b.map((r, ri) => r.map((v) => parseBigIntToken(v, `b row ${ri}`)));
        checkI64Range(A.flat(), 'a');
        checkI64Range(B.flat(), 'b');
        const { M, K, N } = prepareMatmul(A, B);
        const values = runIntMatmul(rt, A, B, M, K, N, Boolean(wrap));
        results.push({ index: i, ok: true, op, shape: [M, N], values });
      } else if (Object.hasOwn(OPS, op)) {
        const a = entry.a.map((v) => parseBigIntToken(v, 'a'));
        const b = entry.b.map((v) => parseBigIntToken(v, 'b'));
        checkI64Range(a, 'a');
        checkI64Range(b, 'b');
        const { aEff, bEff, broadcast } = prepareElementwise(a, b);
        const values = runIntBinary(rt, op, aEff, bEff, Boolean(wrap));
        results.push({ index: i, ok: true, op, a: aEff.map(String), b: bEff.map(String), broadcast, values });
      } else if (op === 'to-f32') {
        const a = entry.a.map((v) => parseBigIntToken(v, 'a'));
        checkI64Range(a, 'a');
        const values = runIntToFloat(rt, a, Boolean(allowInexact));
        results.push({ index: i, ok: true, op, a: a.map(String), values });
      } else if (op === 'from-f32') {
        const a = parseFloatVector(entry.a.join(','), 'a');
        const values = runIntFromFloat(rt, a);
        results.push({ index: i, ok: true, op, a, values });
      } else {
        throw new Error(`unknown int op: ${JSON.stringify(op)}`);
      }
    } catch (e) {
      results.push({ index: i, ok: false, op: entry?.op ?? null, error: e instanceof Error ? e.message : String(e) });
    }
  }
  const allOk = results.every((r) => r.ok);
  printJsonAndExit({ ok: allOk, engine, results }, allOk ? 0 : 2);
}

async function main() {
  const rawArgs = process.argv.slice(2);
  if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
    console.error(USAGE);
    process.exit(2);
  }
  const allowWrap = rawArgs.includes('--wrap');
  const allowInexact = rawArgs.includes('--allow-inexact');
  const posArgs = rawArgs.filter((a) => a !== '--wrap' && a !== '--allow-inexact');
  const [op, aText, bText, extra] = posArgs;

  if (op === '--batch') {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    return runBatch(aText);
  }
  if (op === 'to-f32') {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseVector(aText, 'a');
    checkI64Range(a, 'a');
    const rt = await loadWasm();
    const values = runIntToFloat(rt, a, allowInexact);
    printJsonAndExit({ ok: true, op, a: a.map(String), engine: engineProvenance(), values }, 0);
    return;
  }
  if (op === 'from-f32') {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    const a = parseFloatVector(aText, 'a');
    const rt = await loadWasm();
    const values = runIntFromFloat(rt, a);
    printJsonAndExit({ ok: true, op, a, engine: engineProvenance(), values }, 0);
    return;
  }
  if (op === 'matmul') {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    const A = parseMatrix(aText, 'a');
    const B = parseMatrix(bText, 'b');
    checkI64Range(A.flat(), 'a');
    checkI64Range(B.flat(), 'b');
    const { M, K, N } = prepareMatmul(A, B);
    const rt = await loadWasm();
    const values = runIntMatmul(rt, A, B, M, K, N, allowWrap);
    printJsonAndExit({ ok: true, op: 'matmul', a: A.map((r) => r.map(String)), b: B.map((r) => r.map(String)), shape: [M, N], broadcast: null, engine: engineProvenance(), values }, 0);
    return;
  }
  if (!Object.hasOwn(OPS, op) || aText === undefined || bText === undefined || extra !== undefined) {
    throw new Error(USAGE);
  }
  const a = parseVector(aText, 'a');
  const b = parseVector(bText, 'b');
  checkI64Range(a, 'a');
  checkI64Range(b, 'b');
  const { aEff, bEff, broadcast } = prepareElementwise(a, b);
  const rt = await loadWasm();
  const values = runIntBinary(rt, op, aEff, bEff, allowWrap);
  printJsonAndExit({ ok: true, op, a: aEff.map(String), b: bEff.map(String), broadcast, engine: engineProvenance(), values }, 0);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
