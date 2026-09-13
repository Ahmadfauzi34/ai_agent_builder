import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function tensor(values, shape) {
  return new m.WasmTensor(new Float32Array(values), new Uint32Array(shape));
}

function u32(values) {
  return new Uint32Array(values);
}

function values(t) {
  return Array.from(t.to_array());
}

function close(actual, expected, tolerance = 1e-6) {
  assert(actual.length === expected.length, `length mismatch: ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i += 1) {
    assert(Math.abs(actual[i] - expected[i]) <= tolerance,
      `value mismatch at ${i}: ${actual[i]} != ${expected[i]}`);
  }
}

function expectThrow(fn, label) {
  try {
    fn();
  } catch (error) {
    return String(error);
  }
  throw new Error(`${label}: expected rejection`);
}

const caps = JSON.parse(m.mathProgramCapabilities());
const v4Caps = JSON.parse(m.mathProgramV4Capabilities());
const findings = [];

function finding(id, score, status, evidence, recommendation = 'KEEP') {
  findings.push({ id, score, status, evidence, recommendation });
}

// 1) Branching DAG / fan-out / fan-in should be natural.
{
  const b = new m.WasmMathProgramBuilder(1, 4);
  b.addUnary(caps.opcodes.abs, 0, 1);
  b.addUnary(caps.opcodes.sqrt, 0, 2); // read same input again
  b.addBinary(caps.opcodes.add, 1, 2, 3);
  b.setOutput(3);
  const p = b.compile();
  const x = tensor([1, 4, 9], [1, 3, 1, 1]);
  const y = p.run1(x);
  close(values(y), [2, 6, 12]);
  finding('branching_fanout_fanin', 0, 'natural',
    'slot0 can be read by multiple steps; derived branches can fan-in through a binary op');
  y.free(); x.free(); p.free(); b.free();
}

// 2) Reusing the original input after derived work should remain natural.
{
  const b = new m.WasmMathProgramBuilder(1, 4);
  b.addUnary(caps.opcodes.abs, 0, 1);
  b.addUnary(caps.opcodes.sqrt, 0, 2);
  b.addBinary(caps.opcodes.add, 0, 2, 3);
  b.setOutput(3);
  const p = b.compile();
  const x = tensor([1, 4, 9], [1, 3, 1, 1]);
  const y = p.run1(x);
  close(values(y), [2, 6, 12]);
  finding('read_many_input_reuse', 0, 'natural',
    'write-once outputs do not imply single-use inputs; original slot remains reusable');
  y.free(); x.free(); p.free(); b.free();
}

// 3) Two-input branching remains natural.
{
  const b = new m.WasmMathProgramBuilder(2, 5);
  b.addBinary(caps.opcodes.add, 0, 1, 2);
  b.addUnary(caps.opcodes.abs, 0, 3);
  b.addBinary(caps.opcodes.sub, 2, 3, 4);
  b.setOutput(4);
  const p = b.compile();
  const a = tensor([-1, 2], [1, 2, 1, 1]);
  const c = tensor([4, 5], [1, 2, 1, 1]);
  const y = p.run2(a, c);
  close(values(y), [2, 5]);
  finding('two_input_branching', 0, 'natural',
    'two external inputs can branch independently and recombine without packing');
  y.free(); a.free(); c.free(); p.free(); b.free();
}

// 4) More than two external inputs is a hard representability boundary today.
{
  const err = expectThrow(() => new m.WasmMathProgramBuilder(3, 4), 'three external inputs');
  const v4Err = expectThrow(() => new m.WasmMathProgramV4Builder(3, 4), 'three external inputs v4');
  finding('external_inputs_gt2', 4, 'hard_block',
    `both v1-v3 and v4 builders reject 3 inputs: ${err}; ${v4Err}`,
    'AUDIT_FOR_EXTEND');
}

// 5) Write-once should reject overwrite atomically while preserving recovery.
{
  const b = new m.WasmMathProgramBuilder(1, 3);
  b.addUnary(caps.opcodes.abs, 0, 1);
  const before = b.numSteps();
  expectThrow(() => b.addUnary(caps.opcodes.sqrt, 0, 1), 'slot overwrite');
  assert(b.numSteps() === before, 'failed overwrite mutated builder step count');
  b.addUnary(caps.opcodes.sqrt, 0, 2);
  b.setOutput(2);
  const p = b.compile();
  const x = tensor([4], [1, 1, 1, 1]);
  const y = p.run1(x);
  close(values(y), [2]);
  finding('write_once_atomic_recovery', 0, 'natural',
    'overwrite rejects before mutation; same builder remains usable for a valid correction');
  y.free(); x.free(); p.free(); b.free();
}

// 6) Mixed variable-length transform -> statistics is expressible, but requires the v4 builder family.
{
  const b = new m.WasmMathProgramV4Builder(1, 3);
  b.addSelectAxis(0, 1, 1, u32([2, 0]));
  b.addUnary(caps.opcodes.mean, 1, 2);
  b.setOutput(2);
  const p = b.compile();
  const x = tensor([1, 2, 3], [1, 3, 1, 1]);
  const y = p.run1(x);
  close(values(y), [2]);
  assert(p.programPlan()[4] === 4, 'selectAxis composition did not produce v4 plan');
  finding('mixed_family_selectaxis_statistics', 1, 'minor_boilerplate',
    'composition is direct, but callers must choose WasmMathProgramV4Builder rather than the v1-v3 builder');
  y.free(); x.free(); p.free(); b.free();
}

// 7) Near-valid selectAxis mutations should be distinguished cleanly.
{
  const a = new m.WasmMathProgramV4Builder(1, 2);
  a.addSelectAxis(0, 1, 1, u32([2, 0]));
  a.setOutput(1);
  const pa = a.compile();

  const b = new m.WasmMathProgramV4Builder(1, 2);
  b.addSelectAxis(0, 1, 1, u32([0, 2]));
  b.setOutput(1);
  const pb = b.compile();
  assert(pa.programIdentity() !== pb.programIdentity(), 'index order change did not change identity');

  const dup = new m.WasmMathProgramV4Builder(1, 2);
  dup.addSelectAxis(0, 1, 1, u32([1, 1]));
  dup.setOutput(1);
  const pd = dup.compile();
  const x = tensor([1, 2, 3], [1, 3, 1, 1]);
  const yd = pd.run1(x);
  close(values(yd), [2, 2]);

  const invalid = new m.WasmMathProgramV4Builder(1, 2);
  expectThrow(() => invalid.addSelectAxis(0, 1, 4, u32([0])), 'axis=4');
  expectThrow(() => invalid.addSelectAxis(0, 1, 1, u32([])), 'empty indices');
  assert(invalid.numSteps() === 0, 'invalid select metadata mutated builder');

  finding('selectaxis_semantic_neighborhood', 0, 'natural',
    'order and duplicates are valid semantic variants; axis=4/empty indices reject atomically');

  yd.free(); x.free(); pd.free(); dup.free(); pb.free(); b.free(); pa.free(); a.free(); invalid.free();
}

// 8) Runtime-dependent validity should fail without changing immutable program identity, then recover.
{
  const b = new m.WasmMathProgramV4Builder(1, 2);
  b.addSelectAxis(0, 1, 1, u32([3]));
  b.setOutput(1);
  const p = b.compile();
  const identity = p.programIdentity();
  const small = tensor([1, 2], [1, 2, 1, 1]);
  expectThrow(() => p.run1(small), 'runtime select bounds');
  assert(p.programIdentity() === identity, 'runtime failure changed identity');
  const large = tensor([1, 2, 3, 4], [1, 4, 1, 1]);
  const y = p.run1(large);
  close(values(y), [4]);
  assert(p.programIdentity() === identity, 'valid retry changed identity');
  finding('runtime_dependent_validity_split', 0, 'natural',
    'structural metadata can compile while tensor-size bounds remain runtime-validated; failure is reusable');
  y.free(); large.free(); small.free(); p.free(); b.free();
}

// 9) Canonical identity is structural, not semantic equivalence.
{
  const one = new m.WasmMathProgramBuilder(1, 2);
  one.addUnary(caps.opcodes.abs, 0, 1);
  one.setOutput(1);
  const p1 = one.compile();

  const two = new m.WasmMathProgramBuilder(1, 3);
  two.addUnary(caps.opcodes.abs, 0, 1);
  two.addUnary(caps.opcodes.abs, 1, 2);
  two.setOutput(2);
  const p2 = two.compile();

  const x = tensor([-2, 3], [1, 2, 1, 1]);
  const y1 = p1.run1(x);
  const y2 = p2.run1(x);
  close(values(y1), values(y2));
  assert(p1.programIdentity() !== p2.programIdentity(), 'structurally different equivalent programs shared identity');
  finding('structural_identity_vs_equivalence', 0, 'natural',
    'semantically equivalent tested outputs may retain distinct structural identities; no unsafe canonical folding');
  y2.free(); y1.free(); x.free(); p2.free(); two.free(); p1.free(); one.free();
}

// 10) Rank-4 bridge pressure: lower ranks are normalized, rank >4 is blocked.
{
  const x = tensor([1, 2, 3], [3]);
  const shape = Array.from(x.shape());
  assert(JSON.stringify(shape) === JSON.stringify([3, 1, 1, 1]), `rank-1 normalization mismatch: ${shape}`);
  const rank5Err = expectThrow(
    () => tensor([1], [1, 1, 1, 1, 1]),
    'rank-5 bridge',
  );
  finding('rank4_transport_bridge', 4, 'hard_block_for_rank_gt4',
    `rank<4 is normalized, but rank>4 is rejected: ${rank5Err}`,
    'AUDIT_FOR_SPLIT');
  x.free();
}

// 11) Slot hard limit is explicit; verify the edge rather than assuming it.
{
  const max = new m.WasmMathProgramBuilder(1, 64);
  assert(max.numSlots() === 64, '64-slot builder not accepted');
  const err = expectThrow(() => new m.WasmMathProgramBuilder(1, 65), '65 slots');
  finding('slot_limit_64', 2, 'explicit_capacity_limit',
    `64 slots accepted and 65 rejected: ${err}`,
    'MEASURE_BEFORE_RELAX');
  max.free();
}

// 12) Ambiguity is intentionally not represented in the execution API today.
// This is an audit observation, not a failure: explicit opcodes are required before MathProgram construction.
{
  const ambiguitySurface = [
    'mathProgramResolve',
    'mathProgramResolutionCapabilities',
    'WasmMathProgramResolution',
  ].filter((name) => name in m);
  finding('ambiguity_resolution_state', 3, 'missing_above_execution_core',
    ambiguitySurface.length === 0
      ? 'no NeedsResolution/ambiguity surface exists; MathProgram begins after semantic intent has already been made explicit'
      : `unexpected ambiguity surfaces present: ${ambiguitySurface.join(', ')}`,
    'EXTEND_ABOVE_CORE');
}

const hardBlocks = findings.filter((item) => item.score >= 4).map((item) => item.id);
const awkward = findings.filter((item) => item.score >= 2 && item.score < 4).map((item) => item.id);
const natural = findings.filter((item) => item.score <= 1).map((item) => item.id);

const report = {
  verdict: 'PASS_WITH_FINDINGS',
  task: 'Math Program flexibility and ambiguity audit',
  baseline: {
    mathProgramSchema: caps.schema,
    mathProgramV4Schema: v4Caps.schema,
  },
  summary: {
    naturalCount: natural.length,
    awkwardCount: awkward.length,
    hardBlockCount: hardBlocks.length,
    natural,
    awkward,
    hardBlocks,
  },
  findings,
};

console.log(JSON.stringify(report, null, 2));
