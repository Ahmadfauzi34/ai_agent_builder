#!/usr/bin/env node
// burn-math regression gate (repo helper edition).
// Usage: node helper/bin/regression.mjs   (run from helper/)
// Exits 0 when every case passes, 1 otherwise, and prints a JSON summary.
// Every case is a permanent regression case for a failure that was once
// real: the Gemini Fase 1/2 notes (REG-01..15, EC1-EC5, F1-F6) plus the
// v2.5 fixes (F4 expectedWeightCount, F5 tolerance validation, F6
// drain-safe stdout). A case asserts the CLI behavior (exit code +
// parsed output), never just "the tool ran". Generated specs live in a
// temp dir that is removed at the end. See reference/regression.md (from helper/).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { printJsonAndExit } from '../lib/runner.mjs';
import { expectedWeightCount } from '../lib/spec.mjs';
import { ruleFor } from '../lib/shapes.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'burn-regression-'));
const WASM_SHA = '2fb8cc9034bca2f9947bc7ab416025c2b823ceebef35fac8b918c2fbbd899f33';
const CKPT_SHA = '8c209c9fabf256bcf6dda2fc66d1ca347b6d5ad04eaf1331a08b6ee46059b454';

function runTool(args, { cwd = ROOT } = {}) {
  const r = spawnSync('node', args, { cwd, encoding: 'utf8', timeout: 180000, maxBuffer: 256 * 1024 * 1024 });
  let json = null;
  try { json = JSON.parse(r.stdout); } catch { /* non-JSON stdout (e.g. --list) */ }
  return { code: r.status, out: r.stdout, err: r.stderr, json };
}
function writeSpec(name, obj) {
  const f = path.join(TMP, name);
  fs.writeFileSync(f, JSON.stringify(obj));
  return f;
}
function clone(o) { return JSON.parse(JSON.stringify(o)); }
function assert(cond, msg) { if (!cond) throw new Error(msg); }
function assertValidFalse(r, code) {
  assert(r.code === 2, `expected exit 2, got ${r.code}: ${r.out.slice(0, 200)}`);
  assert(r.json && r.json.valid === false, `expected valid:false, got ${r.out.slice(0, 200)}`);
  if (code) assert(r.json.code === code, `expected code ${code}, got ${r.json.code}`);
}

const basePorts = () => [
  { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
  { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
];
const oneLayerSpec = (layer, step, ports = basePorts(), outputSlot = 1) => ({
  layers: [layer], steps: [step], outputSlot, numSlots: 5, ports,
});

const cases = [];
const test = (name, fn) => cases.push({ name, fn });

test('builder: 48/169/48 + wasm sha', () => {
  // The repo builder (scripts/build_agent_pack.mjs) assembles the pack
  // from this directory; the regression gate proves the builder's output
  // counts and engine sha, not just the tools in place.
  // In a built pack there is no repo builder to test (the pack IS its
  // output, and pin enforcement is still covered by every verify /
  // checkpoint case via wasm_direct), so the case passes vacuously
  // there instead of failing on a missing repo script.
  if (!fs.existsSync(path.join(ROOT, '..', 'scripts', 'build_agent_pack.mjs'))) return;
  const r = runTool(['../scripts/build_agent_pack.mjs', '--out', path.join(TMP, 'built-pack')]);
  assert(r.code === 0, `builder exit ${r.code}: ${r.out.slice(0, 200)} ${r.err.slice(0, 200)}`);
  assert(r.json.ok === true && r.json.catalog_entries === 48 && r.json.surface_entries === 169 && r.json.layout_entries === 48, `counts: ${r.out.slice(0, 300)}`);
  assert(r.json.wasm_sha256 === WASM_SHA, `wasm sha: ${r.json.wasm_sha256}`);
});
test('run: linear-relu [4.5, 10.5]', () => {
  const r = runTool(['bin/run.mjs', 'examples/linear-relu.json']);
  assert(r.code === 0 && r.json.ok === true, `run: ${r.out.slice(0, 200)}`);
  assert(JSON.stringify(r.json.outputs.values) === JSON.stringify([4.5, 10.5]), `values: ${JSON.stringify(r.json.outputs.values)}`);
});
test('validate: linear-relu valid', () => {
  const r = runTool(['bin/validate.mjs', 'examples/linear-relu.json']);
  assert(r.code === 0 && r.json.valid === true, `validate: ${r.out.slice(0, 200)}`);
});
test('verify: identical ACCEPT, mutated REJECT', () => {
  const a = runTool(['bin/verify.mjs', 'examples/linear-relu-bias.json', 'examples/linear-relu-bias.json']);
  assert(a.code === 0 && a.json.verdict === 'ACCEPT', `identical: ${a.out.slice(0, 200)}`);
  const b = runTool(['bin/verify.mjs', 'examples/linear-relu-bias.json', 'examples/linear-relu-mutated.json']);
  assert(b.code === 2 && b.json.verdict === 'REJECT', `mutated: ${b.out.slice(0, 200)}`);
});
test('checkpoint: export sha + verify ACCEPT + restore [4.5, 10.5]', () => {
  const e = runTool(['bin/checkpoint.mjs', 'export', 'examples/linear-relu.json']);
  assert(e.code === 0 && e.json.sha256 === CKPT_SHA, `export sha: ${e.json.sha256}`);
  const b64 = path.join(TMP, 'ck.b64');
  fs.writeFileSync(b64, e.json.bundle_base64);
  const v = runTool(['bin/checkpoint.mjs', 'verify', b64, CKPT_SHA]);
  assert(v.code === 0 && v.json.verdict === 'ACCEPT', `verify: ${v.out.slice(0, 200)}`);
  const r = runTool(['bin/checkpoint.mjs', 'restore', b64, 'examples/linear-relu.json', CKPT_SHA]);
  assert(r.code === 0 && r.json.verdict === 'ACCEPT', `restore: ${r.out.slice(0, 200)}`);
  assert(JSON.stringify(r.json.outputs.values) === JSON.stringify([4.5, 10.5]), `restored values: ${JSON.stringify(r.json.outputs.values)}`);
});
test('REG-10: registry id 2^32 rejected', () => {
  const s = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8')));
  s.layers[0].args[0] = 4294967296;
  const r = runTool(['bin/validate.mjs', writeSpec('id2pow32.json', s)]);
  assertValidFalse(r, 'invalid_argument');
});
test('REG-11: weights 1e39 rejected (f32 overflow)', () => {
  const s = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8')));
  s.weights['0'] = [1e39, 0.5, -0.25, 2, 0.5, -1];
  const r = runTool(['bin/validate.mjs', writeSpec('w1e39.json', s)]);
  assertValidFalse(r, 'invalid_argument');
});
test('REG-12: weights key "" rejected', () => {
  const s = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8')));
  s.weights = { '': [1, 0.5, -0.25, 2] };
  const r = runTool(['bin/validate.mjs', writeSpec('wemptykey.json', s)]);
  assertValidFalse(r, 'invalid_argument');
});
test('F2: fractional slots rejected', () => {
  const s = {
    inputs: { '0.5': [1, -2], '1.5': [0, 0] },
    layers: [{ args: [0, 2, 2, false], constructor: 'linear' }, { args: [1], constructor: 'relu' }, { args: [2], constructor: 'add' }],
    numSlots: 5, outputSlot: 4,
    ports: [
      { layout: 'feature_axis1_singleton', role: 'observation', shape: [1, 2, 1, 1], slot: 0.5 },
      { layout: 'feature_axis1_singleton', role: 'observation', shape: [1, 2, 1, 1], slot: 1.5 },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0.5, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'binary', layer: 2, slots: [3, 1.5, 4] },
    ],
    weights: { 0: [1, 0.5, -0.25, 2] },
  };
  const r = runTool(['bin/validate.mjs', writeSpec('frac.json', s)]);
  assertValidFalse(r, 'invalid_argument');
});
test('F2: collision slots 0 + 0.5 rejected', () => {
  const s = {
    inputs: { 0: [1, -2], '0.5': [0, 0] },
    layers: [{ args: [0, 2, 2, false], constructor: 'linear' }, { args: [1], constructor: 'relu' }, { args: [2], constructor: 'add' }],
    numSlots: 5, outputSlot: 4,
    ports: [
      { layout: 'feature_axis1_singleton', role: 'observation', shape: [1, 2, 1, 1], slot: 0 },
      { layout: 'feature_axis1_singleton', role: 'observation', shape: [1, 2, 1, 1], slot: 0.5 },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'binary', layer: 2, slots: [3, 0.5, 4] },
    ],
    weights: { 0: [1, 0.5, -0.25, 2] },
  };
  const r = runTool(['bin/validate.mjs', writeSpec('coll.json', s)]);
  assertValidFalse(r, 'invalid_argument');
});
test('REG-03/F3: dim-0 shape rejected', () => {
  const s = oneLayerSpec({ args: [0], constructor: 'relu' }, { kind: 'unary', layer: 0, slots: [0, 1] });
  s.ports[0].shape = [1, 0, 1, 1];
  const r = runTool(['bin/validate.mjs', writeSpec('dim0.json', s)]);
  assertValidFalse(r);
});
test('REG-04: conv stride 0 rejected', () => {
  const s = oneLayerSpec(
    { args: [0, 2, 1, 1, 1, 0, 1, 0, 0], constructor: 'conv2d' },
    { kind: 'unary', layer: 0, slots: [0, 1] },
    [
      { slot: 0, role: 'observation', shape: [1, 2, 3, 3], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 1, 3, 3], layout: 'feature_axis1_singleton' },
    ],
  );
  const r = runTool(['bin/validate.mjs', writeSpec('convsh0.json', s)]);
  assertValidFalse(r, 'static_check_failed');
  assert(JSON.stringify(r.json.mismatches).includes('stride'), `stride message: ${JSON.stringify(r.json.mismatches).slice(0, 200)}`);
});
test('REG-13: softmax dim 0 rejected by validate and by engine', () => {
  // Two layers so both ports are consumed by the plan (a one-layer spec
  // fails plan_unused_ports in run before reaching the engine).
  const s = {
    layers: [{ args: [0, 0], constructor: 'softmax' }, { args: [1], constructor: 'add' }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    outputSlot: 3, numSlots: 5, ports: basePorts(),
  };
  const f = writeSpec('softmax0.json', s);
  const v = runTool(['bin/validate.mjs', f]);
  assertValidFalse(v, 'static_check_failed');
  const r = runTool(['bin/run.mjs', f]);
  assert(r.code === 1 && r.json.ok === false, `run softmax dim 0: exit ${r.code} ${r.out.slice(0, 200)}`);
});
test('REG-13/14: leakyRelu slope, prelu count, softplus beta rejected', () => {
  const leaky = oneLayerSpec({ args: [0, -0.1], constructor: 'leakyRelu' }, { kind: 'unary', layer: 0, slots: [0, 1] });
  const r1 = runTool(['bin/validate.mjs', writeSpec('leaky.json', leaky)]);
  assertValidFalse(r1, 'static_check_failed');
  assert(JSON.stringify(r1.json.mismatches).includes('negative_slope'), `leaky message: ${JSON.stringify(r1.json.mismatches).slice(0, 200)}`);
  const prelu = oneLayerSpec(
    { args: [0, 2, 0.25], constructor: 'prelu' }, { kind: 'unary', layer: 0, slots: [0, 1] },
    [
      { slot: 0, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton' },
    ],
  );
  const r2 = runTool(['bin/validate.mjs', writeSpec('prelu.json', prelu)]);
  assertValidFalse(r2, 'static_check_failed');
  assert(JSON.stringify(r2.json.mismatches).includes('num_parameters'), `prelu message: ${JSON.stringify(r2.json.mismatches).slice(0, 200)}`);
  const soft = oneLayerSpec({ args: [0, 0], constructor: 'softplus' }, { kind: 'unary', layer: 0, slots: [0, 1] });
  const r3 = runTool(['bin/validate.mjs', writeSpec('softplus.json', soft)]);
  assertValidFalse(r3, 'static_check_failed');
  assert(JSON.stringify(r3.json.mismatches).includes('beta'), `softplus message: ${JSON.stringify(r3.json.mismatches).slice(0, 200)}`);
});
test('REG-09: prototype pollution names rejected', () => {
  for (const tool of ['find-op', 'layout']) {
    for (const name of ['toString', '__proto__', 'constructor']) {
      const r = runTool([`bin/${tool}.mjs`, name]);
      assert(r.code === 2, `${tool} ${name}: exit ${r.code}`);
    }
  }
});
test('REG-08: --help in second position is usage, not a run', () => {
  const r = runTool(['bin/run.mjs', 'examples/linear-relu.json', '--help']);
  assert(r.code === 2, `run --help pos2: exit ${r.code}`);
  assert(!r.json || r.json.ok !== true, 'must not run the spec');
});
test('REG-06: verify with missing cases file exits 1', () => {
  const r = runTool(['bin/verify.mjs', 'examples/linear-relu.json', 'examples/linear-relu.json', 'no-such-cases.json']);
  assert(r.code === 1, `missing cases: exit ${r.code} ${r.out.slice(0, 200)}`);
});
test('REG-02: checkpoint restore rejects dirty and padding-bit base64', () => {
  const e = runTool(['bin/checkpoint.mjs', 'export', 'examples/linear-relu-bias.json']);
  assert(e.code === 0, `export: ${e.out.slice(0, 200)}`);
  const dirty = path.join(TMP, 'ck-dirty.b64');
  fs.writeFileSync(dirty, e.json.bundle_base64 + '@@@');
  const r1 = runTool(['bin/checkpoint.mjs', 'restore', dirty, 'examples/linear-relu-bias.json', e.json.sha256]);
  assert(r1.code === 1, `dirty restore: exit ${r1.code}`);
  // Padding-bit flip: bundle ends with AAA= (single '='), so the last
  // data char's low 2 bits are unused. A->B changes an unused bit only:
  // decoded bytes are identical, canonical base64 must still reject it.
  const b64 = e.json.bundle_base64;
  assert(b64.endsWith('='), `expected padded bundle, tail ${b64.slice(-4)}`);
  const GRP = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  const last = b64[b64.length - 2];
  const idx = GRP.indexOf(last);
  let nc = GRP[(idx & ~1) | ((idx + 1) & 1)];
  if (nc === last) nc = GRP[idx & ~1];
  const flipped = b64.slice(0, -2) + nc + '=';
  assert(Buffer.from(flipped, 'base64').equals(Buffer.from(b64, 'base64')), 'flipped bundle must decode to identical bytes');
  const pad = path.join(TMP, 'ck-pad.b64');
  fs.writeFileSync(pad, flipped);
  const r2 = runTool(['bin/checkpoint.mjs', 'restore', pad, 'examples/linear-relu-bias.json', e.json.sha256]);
  assert(r2.code === 1, `padding-bit restore: exit ${r2.code} (must reject, not ACCEPT)`);
});
test('REG-07: bad layout + bad role warn (not error), layout carries adapter', () => {
  const s = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8')));
  s.ports[0].layout = 'bogus_layout';
  s.ports[0].role = 'BAD ROLE';
  const r = runTool(['bin/validate.mjs', writeSpec('badlayout.json', s)]);
  assert(r.code === 0 && r.json.valid === true, `bad layout/role must warn, not fail: ${r.out.slice(0, 200)}`);
  const w = r.json.static_warnings ?? [];
  assert(w.some((x) => x.includes('layout')), `layout warning missing: ${JSON.stringify(w)}`);
  assert(w.some((x) => x.includes('role')), `role warning missing: ${JSON.stringify(w)}`);
  const l = runTool(['bin/layout.mjs', 'linear']);
  assert(l.code === 0 && Object.keys(l.json).includes('adapter'), `layout linear keys: ${Object.keys(l.json ?? {})}`);
});
test('REG-15/F1: intAdd warns specifically (i64 vs f32 graph)', () => {
  const s = oneLayerSpec({ args: [0], constructor: 'intAdd' }, { kind: 'binary', layer: 0, slots: [0, 1, 2] }, basePorts(), 2);
  const r = runTool(['bin/validate.mjs', writeSpec('intadd.json', s)]);
  assert(r.code === 0 && r.json.valid === true, `intAdd: ${r.out.slice(0, 200)}`);
  assert((r.json.static_warnings ?? []).some((x) => x.includes('WasmIntTensor')), `int warning: ${JSON.stringify(r.json.static_warnings)}`);
});
test('F4: expectedWeightCount table (unit)', () => {
  const t = [
    [{ constructor: 'linear', args: [0, 2, 2, true] }, 6],
    [{ constructor: 'linear', args: [0, 2, 2, false] }, 4],
    [{ constructor: 'batchNorm', args: [0, 3] }, 6],
    [{ constructor: 'groupNorm', args: [0, 2, 4] }, 8],
    [{ constructor: 'instanceNorm', args: [0, 4] }, 8],
    [{ constructor: 'layerNorm', args: [0, 5] }, 10],
    [{ constructor: 'rmsNorm', args: [0, 5] }, 5],
    [{ constructor: 'conv1d', args: [0, 2, 3, 4] }, 27],
    [{ constructor: 'conv2d', args: [0, 2, 3, 3, 3] }, 57],
    [{ constructor: 'convTranspose2d', args: [0, 2, 3, 3, 3] }, 57],
    [{ constructor: 'embedding', args: [0, 10, 4] }, 40],
    [{ constructor: 'relu', args: [0] }, null],
    [{ constructor: 'linear', args: [0, 2, 2, 'yes'] }, null],
  ];
  for (const [decl, want] of t) {
    const got = expectedWeightCount(decl);
    assert(got === want, `expectedWeightCount(${decl.constructor}, ${JSON.stringify(decl.args)}) = ${got}, want ${want}`);
  }
});
test('F4: conv bias is unconditional in the engine (probe 2026-10-30)', () => {
  // The plan_0 follow-up asked whether expectedWeightCount's unconditional
  // "+outCh" for conv is right for a bias-less conv. The engine's conv
  // constructors take NO bias flag at all, so the probe question became:
  // does the engine always add a per-output-channel bias, appended after
  // the kernel weights? Minimal convs (1 in, 1 out, kernel 1) answer it
  // with arithmetic, not docs: w=2, x=3, b=10 -> 16 iff the bias is real.
  const convSpec = (ctor, args, input, zeroShape, weights) => ({
    layers: [{ args, constructor: ctor }, { args: [1], constructor: 'add' }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    outputSlot: 3, numSlots: 5,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, 1, 1], layout: ctor === 'conv1d' ? 'channel_first_singleton_width' : 'channel_first' },
      { slot: 1, role: 'observation', shape: zeroShape, layout: ctor === 'conv1d' ? 'channel_first_singleton_width' : 'channel_first' },
    ],
    inputs: { 0: input, 1: new Array(zeroShape[1]).fill(0) },
    weights: { 0: weights },
  });
  const runOut = (name, s) => {
    const r = runTool(['bin/run.mjs', writeSpec(name, s)]);
    assert(r.json && r.json.ok !== false, `${name}: ${r.out.slice(0, 200)}`);
    return r.json.outputs.values;
  };
  assert(JSON.stringify(runOut('cvb-c1d.json', convSpec('conv1d', [0, 1, 1, 1], [3], [1, 1, 1, 1], [2, 10]))) === '[16]', 'conv1d bias');
  assert(JSON.stringify(runOut('cvb-c2d.json', convSpec('conv2d', [0, 1, 1, 1, 1], [3], [1, 1, 1, 1], [2, 10]))) === '[16]', 'conv2d bias');
  assert(JSON.stringify(runOut('cvb-ct2d.json', convSpec('convTranspose2d', [0, 1, 1, 1, 1], [3], [1, 1, 1, 1], [2, 10]))) === '[16]', 'convTranspose2d bias');
  // Kernel-first-then-bias, per output channel: [w0,w1,b0,b1] on x=5
  // gives [2*5+10, 3*5+20], not the interleaved [2*5+3, 10*5+20].
  assert(JSON.stringify(runOut('cvb-c1d2.json', convSpec('conv1d', [0, 1, 2, 1], [5], [1, 2, 1, 1], [2, 3, 10, 20]))) === '[20,35]', 'conv1d 2ch layout');
  // Bias-less weight vector is rejected by the engine at bind, and
  // validate's warning names the same expected count.
  const short = convSpec('conv1d', [0, 1, 1, 1], [3], [1, 1, 1, 1], [2]);
  const rr = runTool(['bin/run.mjs', writeSpec('cvb-short.json', short)]);
  assert(rr.json && rr.json.ok === false, `short weights must fail: ${rr.out.slice(0, 200)}`);
  const rv = runTool(['bin/validate.mjs', writeSpec('cvb-short-v.json', short)]);
  assert(rv.code === 0 && rv.json.valid === true, `short validate: ${rv.out.slice(0, 200)}`);
  assert((rv.json.static_warnings ?? []).some((x) => x.includes('expected 2 floats, got 1')), `short warning: ${JSON.stringify(rv.json.static_warnings)}`);
});
test('buildGraph exception-safe: failure errors unchanged, later builds healthy', () => {
  // 2026-10-30: a mid-construction throw used to leak the registry,
  // specs, builder, and plan (+4.3MB over 200 failing builds, WASM
  // linear memory). The fix frees the partial construction in
  // buildGraph's catch. These cases pin the observable behavior: the
  // three failure modes that reach buildGraph (past the runner's own
  // host checks) must report byte-identical errors before/after, and
  // a valid build right after the failures must still run correctly
  // (no corrupted state). The memory proof itself lives in the
  // build-safe battery (WASM-memory delta 0 over 200 failing builds),
  // not here: this harness is synchronous and cannot hold initSync's
  // memory handle.
  const ports = [
    { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
  ];
  const mk = (layers, steps, outputSlot, weights) => {
    const s = { layers, steps, outputSlot, numSlots: 5, ports, inputs: { 0: [1, 2], 1: [0, 0] } };
    if (weights) s.weights = weights;
    return s;
  };
  const direct = (name, s) => runTool(['bin/run.mjs', '--direct', writeSpec(name, s)]);
  const lin = { constructor: 'linear', args: [0, 2, 2, true] };
  const add = { constructor: 'add', args: [2] };
  const r1 = direct('bs-unknown.json', mk([lin, { constructor: 'noSuch', args: [1] }, add],
    [{ kind: 'unary', layer: 0, slots: [0, 2] }, { kind: 'binary', layer: 2, slots: [2, 1, 3] }], 3));
  assert(r1.json && r1.json.ok === false && r1.json.error_envelope?.code === 'unknown_layer_constructor', `unknown ctor: ${r1.out.slice(0, 200)}`);
  const r2 = direct('bs-badstep.json', mk([lin, { constructor: 'relu', args: [1] }, add],
    [{ kind: 'unary', layer: 0, slots: [0, 2] }, { kind: 'unary', layer: 1, slots: [2, 3, 9] }, { kind: 'binary', layer: 2, slots: [3, 1, 4] }], 4));
  assert(r2.json && r2.json.ok === false && r2.json.error_envelope?.code === 'invalid_argument' && r2.json.error.includes('bad step'), `bad step: ${r2.out.slice(0, 200)}`);
  const r3 = direct('bs-weights.json', mk([lin, { constructor: 'relu', args: [1] }, add],
    [{ kind: 'unary', layer: 0, slots: [0, 2] }, { kind: 'unary', layer: 1, slots: [2, 3] }, { kind: 'binary', layer: 2, slots: [3, 1, 4] }], 4, { 1: [1] }));
  assert(r3.json && r3.json.ok === false && r3.json.error === 'weights for layer index 1: not a weighted layer', `weights: ${r3.out.slice(0, 200)}`);
  const ok1 = runTool(['bin/run.mjs', '--direct', 'examples/linear-relu.json']);
  const ok2 = runTool(['bin/run.mjs', '--direct', 'examples/linear-relu.json']);
  assert(ok1.json && ok1.json.ok === true && JSON.stringify(ok1.json.outputs) === JSON.stringify(ok2.json.outputs), `post-failure build: ${ok1.out.slice(0, 200)}`);
});
test('F4: validate warns on weight/input length mismatch', () => {
  const s1 = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu-bias.json'), 'utf8')));
  s1.weights['0'] = [1, 2, 3];
  const r1 = runTool(['bin/validate.mjs', writeSpec('wmismatch.json', s1)]);
  assert(r1.code === 0 && r1.json.valid === true, `weight mismatch must warn: ${r1.out.slice(0, 200)}`);
  assert((r1.json.static_warnings ?? []).some((x) => x.includes('expected 6 floats, got 3')), `linear warning: ${JSON.stringify(r1.json.static_warnings)}`);
  // Two layers so both ports are consumed (warnings only print on the
  // validate success path; a plan failure would hide them).
  const s2 = {
    layers: [{ args: [0, 2], constructor: 'batchNorm' }, { args: [1], constructor: 'add' }],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    outputSlot: 3, numSlots: 5, ports: basePorts(),
  };
  s2.weights = { 0: [1, 2, 3] };
  const r2 = runTool(['bin/validate.mjs', writeSpec('bnmismatch.json', s2)]);
  assert(r2.code === 0 && r2.json.valid === true, `batchNorm mismatch must warn: ${r2.out.slice(0, 200)}`);
  assert((r2.json.static_warnings ?? []).some((x) => x.includes('expected 4 floats, got 3')), `batchNorm warning: ${JSON.stringify(r2.json.static_warnings)}`);
  const s3 = clone(JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8')));
  s3.inputs['0'] = [1];
  const r3 = runTool(['bin/validate.mjs', writeSpec('imismatch.json', s3)]);
  assert(r3.code === 0 && r3.json.valid === true, `input mismatch must warn: ${r3.out.slice(0, 200)}`);
  assert((r3.json.static_warnings ?? []).some((x) => x.includes('expected 2 values')), `input warning: ${JSON.stringify(r3.json.static_warnings)}`);
});
test('F5: verify tolerances validated (abc, negative rejected; 0 accepted)', () => {
  const b = runTool(['bin/verify.mjs', 'examples/linear-relu-bias.json', 'examples/linear-relu-bias.json']);
  assert(b.code === 0, 'baseline verify must pass');
  for (const bad of ['abc', '-1', 'Infinity']) {
    const r = runTool(['bin/verify.mjs', 'examples/linear-relu-bias.json', 'examples/linear-relu-bias.json', '', bad]);
    assert(r.code === 2, `absTol=${bad}: exit ${r.code}`);
  }
  const z = runTool(['bin/verify.mjs', 'examples/linear-relu-bias.json', 'examples/linear-relu-bias.json', '', '0', '0']);
  assert(z.code === 0 && z.json.verdict === 'ACCEPT', `absTol=0: ${z.out.slice(0, 200)}`);
});
test('F6: verify --batch output >64KB parses completely (drain-safe)', () => {
  const pairs = Array.from({ length: 100 }, () => ({
    baseline: 'examples/linear-relu-bias.json',
    candidate: 'examples/linear-relu-bias.json',
  }));
  const pf = path.join(TMP, 'pairs.json');
  fs.writeFileSync(pf, JSON.stringify(pairs));
  const r = runTool(['bin/verify.mjs', '--batch', pf]);
  assert(r.code === 0, `batch exit ${r.code}: ${r.out.slice(0, 200)}`);
  assert(r.out.length > 65536, `batch output ${r.out.length} bytes is not over the 64KB pipe buffer — test is vacuous`);
  assert(Array.isArray(r.json) && r.json.length === 100, `batch results: ${r.out.slice(0, 200)}`);
  assert(r.json.every((x) => x.ok === true && x.equivalent === true), 'every batch pair must ACCEPT');
});
test('REG-01: all 14 formerly rule-less constructors now have shape rules', () => {
  for (const c of ['matmul', 'transpose', 'conv1d', 'maxPool1d', 'avgPool1d', 'embedding', 'swiGlu', 'ghost', 'intAdd', 'intSub', 'intMul', 'intMatmul', 'intFromFloat', 'intToFloat']) {
    assert(typeof ruleFor(c) === 'function', `${c} has no shape rule`);
  }
});
test('arith: elementwise ops through the engine, clean errors', () => {
  // Increment 1 of arithmetic ease (2026-10-30): the tool must print
  // ENGINE output, not host arithmetic — so the test asserts the
  // values the engine computed, plus the four clean-error shapes.
  const run = (args) => runTool(['bin/arith.mjs', ...args]);
  const vals = (args) => {
    const r = run(args);
    assert(r.code === 0 && r.json && r.json.ok === true, `${args.join(' ')}: ${r.out.slice(0, 200)}`);
    return r.json.values;
  };
  assert(JSON.stringify(vals(['add', '3,4', '5,6'])) === '[8,10]', 'add');
  assert(JSON.stringify(vals(['sub', '10,3', '4,9'])) === '[6,-6]', 'sub');
  assert(JSON.stringify(vals(['mul', '3,4', '5,6'])) === '[15,24]', 'mul');
  assert(JSON.stringify(vals(['div', '12,20', '4,5'])) === '[3,4]', 'div via reciprocal');
  assert(JSON.stringify(vals(['add', '1', '2'])) === '[3]', 'scalar');
  const errs = [
    [['div', '1,2', '1,0'], 'division by zero at index 1'],
    [['add', '1,2', '3'], 'length mismatch'],
    [['pow', '2', '3'], 'usage:'],
    [['add', 'a,b', '1,2'], 'not a finite number'],
  ];
  for (const [args, want] of errs) {
    const r = run(args);
    assert(r.code === 2 && r.json && r.json.ok === false && r.json.error.includes(want), `${args.join(' ')}: ${r.out.slice(0, 200)}`);
  }
});
test('catalog surface: find-op --list 48, surface --list 169', () => {
  const f = runTool(['bin/find-op.mjs', '--list']);
  assert(f.code === 0 && f.out.trim().split('\n').length === 48, `find-op list: ${f.out.trim().split('\n').length} lines`);
  const s = runTool(['bin/surface.mjs', '--list']);
  assert(s.code === 0 && s.out.trim().split('\n').length === 169, `surface list: ${s.out.trim().split('\n').length} lines`);
});

const results = [];
for (const c of cases) {
  try { c.fn(); results.push({ name: c.name, ok: true }); }
  catch (e) { results.push({ name: c.name, ok: false, detail: e instanceof Error ? e.message : String(e) }); }
}
try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* ignore */ }
const passed = results.filter((r) => r.ok).length;
printJsonAndExit({ ok: passed === results.length, passed, failed: results.length - passed, total: results.length, cases: results }, passed === results.length ? 0 : 1);
