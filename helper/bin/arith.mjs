#!/usr/bin/env node
// burn-math arith: arithmetic without hand-writing a spec.
//
//   node bin/arith.mjs add 3,4 5,6        -> {"ok":true,"op":"add","values":[8,10]}
//   node bin/arith.mjs sub 10,3 4,9        -> {"ok":true,"op":"sub","values":[6,-6]}
//   node bin/arith.mjs mul 3,4 5,6         -> {"ok":true,"op":"mul","values":[15,24]}
//   node bin/arith.mjs div 12,20 4,5       -> {"ok":true,"op":"div","values":[3,4]}
//   node bin/arith.mjs matmul "1,2,3;4,5,6" "7,8;9,10;11,12"
//        -> {"ok":true,"op":"matmul","shape":[2,2],"values":[58,64,139,154]}
//   node bin/arith.mjs --batch ops.json
//        -> {"ok":true,"engine":{...},"results":[{"index":0,"ok":true,...}]}
//   Batch file: {"ops":[{"op":"add","a":[...],"b":[...]},
//   {"op":"matmul","a":[[...],...],"b":[[...],...]}]}. One engine load
//   for all ops; a failing op is recorded with its index and the rest
//   still run (per-op continue); exit 0 iff every op succeeds, else 2.
//   Batch-level failures (unreadable file, bad JSON, ops not an array)
//   print {"ok":false,"error":...} with exit 2.
//
// Why this tool exists: the engine already does arithmetic (add, sub,
// mul are layer constructors), but using them meant hand-writing a
// full spec JSON (layers, steps, ports, shapes, layouts) for every
// computation. This tool builds that spec from two comma lists and
// runs it through the engine in-process (wasm-direct scope, same
// direct path as `run.mjs --direct`), so the result is engine output,
// not host-side JavaScript arithmetic.
//
// Honest limits (v1, deliberate):
// - Elementwise vectors only, except `matmul`: matrices are rows
//   separated by `;` (a bare comma list is one row), single-batch
//   `[1,1,M,K] @ [1,1,K,N] -> [1,1,M,N]` row-major with `any_rank4`
//   ports (contract probe 2026-10-31, VERDICT-A). No batch axes in
//   v1: the engine rejects mismatched batches, it never broadcasts
//   them. Output `values` are flat row-major MxN with `shape: [M,N]`;
//   `a`/`b` echo as row arrays.
// - No integer ops in THIS tool: intAdd/intSub/intMul/intMatmul live
//   in the i64 registry world (LayerRegistry.forwardIntBinary), not
//   the f32 graph path — see bin/int.mjs (CP5 correction 2026-10-31:
//   addBinary rejects int specs by design, it is not a missing
//   feature). This tool is f32.
// - No `div` constructor exists in the engine, so `div` is mul by
//   the host-computed reciprocal. Division by zero is a clean error
//   (exit 2), never Infinity: the reciprocal is checked finite
//   before the spec is built. A subnormal divisor whose reciprocal
//   overflows is NOT "division by zero": it errors as divisor
//   underflow (in f32 such a divisor is zero anyway, so erroring
//   is correct for this f32 tool — only the label used to be wrong).
//   Precision contract (measured 2026-10-31, n=200 random pairs,
//   max rel err 5.96e-8): div-by-reciprocal stays within 1 ulp of
//   f32(true division) for normal values; the regression suite pins
//   this with fixed stress pairs.
// - Vectors must have equal lengths, except a length-1 vector
//   broadcasts to the other's length (reported in `broadcast`).
// - Output is the flat value vector; shapes are [1, n, 1, 1]
//   (feature_axis1_singleton), the same vector convention the
//   specs use — except matmul, whose values are flat row-major MxN
//   with `shape: [M,N]` and `a`/`b` echoed as row arrays.
//   `a`/`b` echo the effective (post-broadcast,
//   pre-reciprocal) inputs; `engine` carries the wasm pin short
//   + mode so results are attributable to a build. Exit 0 on
//   success, 2 on usage/computation errors.
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm, withGraphScope } from '../lib/wasm_direct.mjs';
import { portBySlot } from '../lib/spec.mjs';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));

// Provenance without re-hashing: loadWasm() already proves the loaded
// engine matches the MANIFEST pin (it throws otherwise), so reporting
// the pin identifies the engine that computed the result.
function engineProvenance() {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'MANIFEST.json'), 'utf8'));
    const full = m?.sha256?.['burn_research_bg.wasm'] ?? m?.engine?.wasm_sha256 ?? null;
    return { wasm_sha256_short: full ? full.slice(0, 16) : null, mode: 'direct' };
  } catch { return { wasm_sha256_short: null, mode: 'direct' }; }
}

const OPS = { add: 'add', sub: 'sub', mul: 'mul', div: 'mul' };
const USAGE = 'usage: node bin/arith.mjs <add|sub|mul|div> <a1,a2,...> <b1,b2,...> OR matmul <a11,a12;...a21,...> <b11,b12;...b21,...> OR --batch <ops.json>';

function parseVector(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const values = text.split(',').map((t) => {
    const v = Number(t.trim());
    if (t.trim() === '' || !Number.isFinite(v)) throw new Error(`${name}: not a finite number: ${JSON.stringify(t)}`);
    return v;
  });
  if (values.length === 0) throw new Error(`${name}: empty vector`);
  return values;
}

async function runMatmul(aText, bText) {
  // Single-batch v1 (probe CP1): [1,1,M,K] @ [1,1,K,N], row-major,
  // any_rank4 ports. No batch broadcast: the engine rejects mismatched
  // batches instead of broadcasting them, and v1 never builds one.
  const A = parseMatrix(aText, 'a');
  const B = parseMatrix(bText, 'b');
  const { M, K, N } = prepareMatmul(A, B);
  const rt = await loadWasm();
  const values = await runMatmulSpec(rt, A, B, M, K, N);
  printJsonAndExit({ ok: true, op: 'matmul', a: A, b: B, shape: [M, N], broadcast: null, engine: engineProvenance(), values }, 0);
}

// Batch (CP4, Opsi A) helpers: same checks as the CLI path, factored
// so one engine load serves every op. prepare* is pure (throws before
// any engine work); run*Spec runs one already-loaded rt.
function parseVectorValues(arr, name) {
  if (!Array.isArray(arr)) throw new Error(`${name}: expected an array of finite numbers`);
  if (arr.length === 0) throw new Error(`${name}: empty vector`);
  for (const v of arr) {
    if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${name}: not a finite number: ${JSON.stringify(v)}`);
  }
  return arr.slice();
}

function parseMatrixValues(arr, name) {
  if (!Array.isArray(arr)) throw new Error(`${name}: expected an array of rows`);
  if (arr.length === 0) throw new Error(`${name}: empty matrix`);
  const rows = arr.map((row, ri) => {
    if (!Array.isArray(row)) throw new Error(`${name} row ${ri}: expected an array of finite numbers`);
    if (row.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    for (const v of row) {
      if (typeof v !== 'number' || !Number.isFinite(v)) throw new Error(`${name} row ${ri}: not a finite number: ${JSON.stringify(v)}`);
    }
    return row.slice();
  });
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  return rows;
}

function prepareElementwise(op, a, b) {
  // Length-1 broadcasts; anything else must match exactly (the engine
  // requires exactly matching shapes for binary ops).
  let aEff = a, bEff = b, broadcast = null;
  if (a.length !== b.length) {
    if (a.length === 1) { aEff = new Array(b.length).fill(a[0]); broadcast = 'a'; }
    else if (b.length === 1) { bEff = new Array(a.length).fill(b[0]); broadcast = 'b'; }
    else throw new Error(`length mismatch: a has ${a.length} values, b has ${b.length} (binary ops require exactly matching shapes)`);
  }
  // bUser: post-broadcast, pre-reciprocal — what the output echoes, so
  // values[i] = a[i] op b[i] always holds for the reader.
  let bUser = bEff;
  if (op === 'div') {
    bEff = bEff.map((v, i) => {
      if (v === 0) throw new Error(`division by zero at index ${i} (b[${i}] is 0)`);
      const r = 1 / v;
      if (!Number.isFinite(r)) throw new Error(`divisor underflow at index ${i} (b[${i}] is ${v}: its reciprocal is not finite; in f32 it is zero)`);
      return r;
    });
  }
  return { aEff, bUser, bEff, broadcast };
}

async function runElementwiseSpec(rt, op, aEff, bEff) {
  const n = aEff.length;
  const spec = {
    layers: [{ constructor: OPS[op], args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2, numSlots: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, n, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, n, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {}, inputs: { 0: aEff, 1: bEff },
  };
  return withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
    const bnd = bundle(spec.inputs, portBySlot(spec));
    const out = g.run(reg, bnd);
    const v = Array.from(out.to_array());
    out.free();
    return v;
  });
}

function prepareMatmul(A, B) {
  const M = A.length, K = A[0].length, K2 = B.length, N = B[0].length;
  if (K !== K2) throw new Error(`matmul shape mismatch: a is ${M}x${K}, b is ${K2}x${N} (inner dimensions must match)`);
  return { M, K, N };
}

async function runMatmulSpec(rt, A, B, M, K, N) {
  const spec = {
    layers: [{ constructor: 'matmul', args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2, numSlots: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, M, K], layout: 'any_rank4' },
      { slot: 1, role: 'observation', shape: [1, 1, K, N], layout: 'any_rank4' },
    ],
    weights: {}, inputs: { 0: A.flat(), 1: B.flat() },
  };
  return withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
    const bnd = bundle(spec.inputs, portBySlot(spec));
    const out = g.run(reg, bnd);
    const v = Array.from(out.to_array());
    out.free();
    return v;
  });
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
  // One engine load for the whole batch (loadWasm has no cache: every
  // call re-hashes and re-imports), provenance once at top level.
  const rt = await loadWasm();
  const engine = engineProvenance();
  const results = [];
  for (let i = 0; i < doc.ops.length; i++) {
    const entry = doc.ops[i];
    try {
      if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) throw new Error(`op entry ${i} must be an object with op, a, b`);
      const { op } = entry;
      if (op === 'matmul') {
        const A = parseMatrixValues(entry.a, 'a');
        const B = parseMatrixValues(entry.b, 'b');
        const { M, K, N } = prepareMatmul(A, B);
        const values = await runMatmulSpec(rt, A, B, M, K, N);
        results.push({ index: i, ok: true, op, a: A, b: B, shape: [M, N], broadcast: null, values });
      } else if (Object.hasOwn(OPS, op)) {
        const a = parseVectorValues(entry.a, 'a');
        const b = parseVectorValues(entry.b, 'b');
        const { aEff, bUser, bEff, broadcast } = prepareElementwise(op, a, b);
        const values = await runElementwiseSpec(rt, op, aEff, bEff);
        results.push({ index: i, ok: true, op, a: aEff, b: bUser, broadcast, values });
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

function parseMatrix(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const rows = text.split(';').map((row, ri) => {
    const vals = row.split(',').map((t) => {
      const v = Number(t.trim());
      if (t.trim() === '' || !Number.isFinite(v)) throw new Error(`${name} row ${ri}: not a finite number: ${JSON.stringify(t)}`);
      return v;
    });
    if (vals.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    return vals;
  });
  if (rows.length === 0) throw new Error(`${name}: empty matrix`);
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  return rows;
}

async function main() {
  const [op, aText, bText, extra] = process.argv.slice(2);
  if (op === '--batch') {
    if (aText === undefined || bText !== undefined || extra !== undefined) throw new Error(USAGE);
    return runBatch(aText);
  }
  if (op === 'matmul') {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    return runMatmul(aText, bText);
  }
  if (!Object.hasOwn(OPS, op) || aText === undefined || bText === undefined || extra !== undefined) {
    throw new Error(USAGE);
  }
  const a = parseVector(aText, 'a');
  const b = parseVector(bText, 'b');
  const { aEff, bUser, bEff, broadcast } = prepareElementwise(op, a, b);
  const rt = await loadWasm();
  const values = await runElementwiseSpec(rt, op, aEff, bEff);
  // Echo the EFFECTIVE inputs (post-broadcast, pre-reciprocal for div)
  // so values[i] = a[i] op b[i] always holds for the reader.
  printJsonAndExit({ ok: true, op, a: aEff, b: bUser, broadcast, engine: engineProvenance(), values }, 0);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
