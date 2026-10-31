#!/usr/bin/env node
// burn-math int: exact i64 arithmetic through the engine's integer world.
//
//   node bin/int.mjs add 16777217,100 1,-30   -> {"ok":true,"op":"add","values":["16777218","70"]}
//   node bin/int.mjs sub 10,-5 3,7            -> {"ok":true,"op":"sub","values":["7","-12"]}
//   node bin/int.mjs mul 6,-4 7,8             -> {"ok":true,"op":"mul","values":["42","-32"]}
//   node bin/int.mjs matmul "1,2;3,4" "5,6;7,8"
//        -> {"ok":true,"op":"matmul","shape":[2,2],"values":["19","22","43","50"]}
//
// Why a separate tool from arith: int does NOT go through the graph
// path. Graph slots are f32; an int layer id in a plan fails validation,
// and addBinary rejects int specs by design (int_binary 0x16 is not
// LAYER_BINARY). Int runs imperatively through the registry:
// LayerRegistry.initAgentLayer(spec) + forwardIntBinary(id, a, b)
// (repo docs/int-path-guide.md). Same engine, different world —
// which is why this is a separate tool, not an arith flag.
//
// Honest limits (v1, deliberate):
// - No intDiv constructor exists in the engine, so no div op.
// - No casts in v1 (intToFloat/intFromFloat are one follow-up away).
// - No batch JSON in v1: batch uses the graph path, int needs registry
//   dispatch per op — a later increment, not a gap to hide.
// - Elementwise vectors only, except `matmul`: matrices are rows
//   separated by `;` (a bare comma list is one row), single-batch
//   `[1,1,M,K] @ [1,1,K,N]` (parity with arith matmul v1). Output
//   `values` are flat row-major with `shape: [M,N]`; `a`/`b` echo as
//   row arrays of strings.
// - Inputs are INTEGER STRINGS (comma lists; matmul rows separated by
//   `;`): parsed as BigInt, so the full i64 range is reachable.
//   fromNumbers caps at 2^53; BigInt64Array does not. A token that is
//   not an integer (1.5, 1e3, "x") is a clean error, exit 2.
// - Outputs are DECIMAL STRINGS: BigInt64Array is not JSON-serializable
//   and Number() would silently round beyond 2^53. Strings keep the
//   exactness the integer world exists for.
// - Range: values outside i64 are a clean error, checked here BEFORE
//   the tensor is built — BigInt64Array would silently wrap them
//   (modulo 2^64), which must never reach the user as a result.
// - Overflow of the arithmetic itself: the release engine wraps
//   two's-complement silently (Rust native `+`/`*`); debug builds panic.
//   This tool does NOT range-check arithmetic results: "never wraps"
//   is the caller's invariant to enforce. Documented here, in AGENT.md,
//   and in the output contract — never silently changed.
// - Vectors must have equal lengths, except a length-1 vector
//   broadcasts to the other's length (reported in `broadcast`).
//   `a`/`b` echo the effective (post-broadcast) inputs as strings.
//   `engine` carries the wasm pin short + mode so results are
//   attributable to a build. Exit 0 on success, 2 on
//   usage/computation errors.
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm } from '../lib/wasm_direct.mjs';
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

const OPS = { add: 'intAdd', sub: 'intSub', mul: 'intMul' };
const USAGE = 'usage: node bin/int.mjs <add|sub|mul> <a1,a2,...> <b1,b2,...> OR matmul <a11,a12;...a21,...> <b11,b12;...b21,...>';

const I64_MIN = -(2n ** 63n);
const I64_MAX = 2n ** 63n - 1n;
const INT_RE = /^-?\d+$/;

function parseBigIntToken(t, name) {
  const s = t.trim();
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

function parseMatrix(text, name) {
  if (text === undefined || text === '') throw new Error(USAGE);
  const rows = text.split(';').map((row, ri) => {
    const vals = row.split(',').map((t) => parseBigIntToken(t, `${name} row ${ri}`));
    if (vals.length === 0) throw new Error(`${name} row ${ri}: empty row`);
    return vals;
  });
  if (rows.length === 0) throw new Error(`${name}: empty matrix`);
  const width = rows[0].length;
  if (!rows.every((r) => r.length === width)) throw new Error(`${name}: non-rectangular rows (row lengths ${rows.map((r) => r.length).join(',')})`);
  return rows;
}

function prepareElementwise(a, b) {
  // Length-1 broadcasts; anything else must match exactly (the engine
  // requires exactly matching shapes for binary ops).
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

function runIntBinary(rt, ctor, aEff, bEff) {
  const n = aEff.length;
  const reg = new rt.LayerRegistry();
  reg.initAgentLayer(rt.AgentLayerSpec[ctor](1));
  const a = makeIntTensor(rt, aEff, [1, n, 1, 1]);
  const b = makeIntTensor(rt, bEff, [1, n, 1, 1]);
  const out = reg.forwardIntBinary(1, a, b);
  const values = Array.from(out.to_array(), String);
  out.free(); a.free(); b.free(); reg.free();
  return values;
}

function runIntMatmul(rt, A, B, M, K, N) {
  const reg = new rt.LayerRegistry();
  reg.initAgentLayer(rt.AgentLayerSpec.intMatmul(1));
  const a = makeIntTensor(rt, A.flat(), [1, 1, M, K]);
  const b = makeIntTensor(rt, B.flat(), [1, 1, K, N]);
  const out = reg.forwardIntBinary(1, a, b);
  const values = Array.from(out.to_array(), String);
  out.free(); a.free(); b.free(); reg.free();
  return values;
}

async function main() {
  const [op, aText, bText, extra] = process.argv.slice(2);
  if (op === 'matmul') {
    if (aText === undefined || bText === undefined || extra !== undefined) throw new Error(USAGE);
    const A = parseMatrix(aText, 'a');
    const B = parseMatrix(bText, 'b');
    checkI64Range(A.flat(), 'a');
    checkI64Range(B.flat(), 'b');
    const { M, K, N } = prepareMatmul(A, B);
    const rt = await loadWasm();
    const values = runIntMatmul(rt, A, B, M, K, N);
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
  const values = runIntBinary(rt, OPS[op], aEff, bEff);
  printJsonAndExit({ ok: true, op, a: aEff.map(String), b: bEff.map(String), broadcast, engine: engineProvenance(), values }, 0);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
