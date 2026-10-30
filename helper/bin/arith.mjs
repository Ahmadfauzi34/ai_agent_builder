#!/usr/bin/env node
// burn-math arith: arithmetic without hand-writing a spec.
//
//   node bin/arith.mjs add 3,4 5,6        -> {"ok":true,"op":"add","values":[8,10]}
//   node bin/arith.mjs sub 10,3 4,9        -> {"ok":true,"op":"sub","values":[6,-6]}
//   node bin/arith.mjs mul 3,4 5,6         -> {"ok":true,"op":"mul","values":[15,24]}
//   node bin/arith.mjs div 12,20 4,5       -> {"ok":true,"op":"div","values":[3,4]}
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
// - Elementwise vectors only. `matmul` is NOT included: its layout
//   contract is dynamic ("last-two-axis batched matmul") and its
//   port-shape semantics are unprobed — including it unverified
//   would break the measured-not-claimed rule. It is increment 2,
//   after a contract probe.
// - No integer ops: intAdd/intSub/intMul live in the i64 graph
//   family and are rejected by the f32 run path ("spec is not
//   binary"). This tool is f32.
// - No `div` constructor exists in the engine, so `div` is mul by
//   the host-computed reciprocal. Division by zero is a clean error
//   (exit 2), never Infinity: the reciprocal is checked finite
//   before the spec is built.
// - Vectors must have equal lengths (the engine requires exactly
//   matching shapes for binary ops); a mismatch is a clean error.
// - Output is the flat value vector; shapes are [1, n, 1, 1]
//   (feature_axis1_singleton), the same vector convention the
//   specs use. Exit 0 on success, 2 on usage/computation errors.
import { printJsonAndExit } from '../lib/runner.mjs';
import { loadWasm, withGraphScope } from '../lib/wasm_direct.mjs';
import { portBySlot } from '../lib/spec.mjs';

const OPS = { add: 'add', sub: 'sub', mul: 'mul', div: 'mul' };

function parseVector(text, name) {
  if (text === undefined || text === '') throw new Error(`usage: node bin/arith.mjs <add|sub|mul|div> <a1,a2,...> <b1,b2,...>`);
  const values = text.split(',').map((t) => {
    const v = Number(t.trim());
    if (t.trim() === '' || !Number.isFinite(v)) throw new Error(`${name}: not a finite number: ${JSON.stringify(t)}`);
    return v;
  });
  if (values.length === 0) throw new Error(`${name}: empty vector`);
  return values;
}

async function main() {
  const [op, aText, bText, extra] = process.argv.slice(2);
  if (!Object.hasOwn(OPS, op) || aText === undefined || bText === undefined || extra !== undefined) {
    throw new Error(`usage: node bin/arith.mjs <add|sub|mul|div> <a1,a2,...> <b1,b2,...>`);
  }
  const a = parseVector(aText, 'a');
  const b = parseVector(bText, 'b');
  if (a.length !== b.length) {
    throw new Error(`length mismatch: a has ${a.length} values, b has ${b.length} (binary ops require exactly matching shapes)`);
  }
  let bEff = b;
  if (op === 'div') {
    bEff = b.map((v, i) => {
      if (v === 0) throw new Error(`division by zero at index ${i} (b[${i}] is 0)`);
      const r = 1 / v;
      if (!Number.isFinite(r)) throw new Error(`division by zero at index ${i} (1/${v} is not finite)`);
      return r;
    });
  }
  const n = a.length;
  const spec = {
    layers: [{ constructor: OPS[op], args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2, numSlots: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, n, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, n, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {}, inputs: { 0: a, 1: bEff },
  };
  const rt = await loadWasm();
  const values = await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
    const bnd = bundle(spec.inputs, portBySlot(spec));
    const out = g.run(reg, bnd);
    const v = Array.from(out.to_array());
    out.free();
    return v;
  });
  printJsonAndExit({ ok: true, op, a, b, values }, 0);
}

main().catch((e) => {
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
});
