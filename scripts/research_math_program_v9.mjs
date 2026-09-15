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

function values(t) {
  return Array.from(t.to_array());
}

function expectThrow(fn, label) {
  let threw = false;
  try {
    const value = fn();
    if (value?.free) value.free();
  } catch {
    threw = true;
  }
  assert(threw, `${label} should fail with a controlled error`);
}

const baseCaps = JSON.parse(m.mathProgramCapabilities());
const caps = JSON.parse(m.mathProgramV9Capabilities());
assert(caps.schema === 'burn-research.math-program.v9', 'v9 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v9', 'v9 plan schema mismatch');
assert(caps.min_inputs === 1 && caps.max_inputs === 8, 'v9 input bounds mismatch');
assert(caps.max_slots === 64, 'v9 max_slots mismatch');
assert(caps.legacy_step_validation === 'canonical_one_step_v8_replay', 'v9 legacy delegation mismatch');
assert(JSON.stringify(caps.delegated_ops) === JSON.stringify(['indicesLike', 'lessEqual01']), 'v9 delegated ops mismatch');
assert(caps.index_axis_binding === 'identity_bound_plan_metadata', 'v9 index axis binding mismatch');
assert(caps.index_source_semantics === 'delegate_index_source_v1', 'v9 index source delegation mismatch');
assert(caps.comparison_semantics === 'delegate_comparison_v1', 'v9 comparison delegation mismatch');
assert(caps.causal_mask_special_case === false, 'v9 must not add causalMask special case');
assert(caps.select_where_required === false, 'v9 causal path must not require select/where');
assert(caps.boolean_tensor_family === false, 'v9 must not add boolean tensor family');
assert(caps.implicit_broadcasting === false, 'v9 must not add implicit broadcasting');
assert(caps.legacy_v1_v8_decoders_frozen === true, 'legacy decoders must remain frozen');
assert(caps.registry_dependency === false, 'v9 must remain registry-independent');
assert(caps.mutable_state === false, 'v9 must remain stateless');
assert(caps.grants_authority === false, 'v9 must not grant authority');

// One-input causal additive mask entirely inside v9.
// scores -> query/key indices -> numeric predicate -> arithmetic mask -> masked scores.
const builder = new m.WasmMathProgramV9Builder(1, 9);
builder.addIndicesLike(0, 1, 1); // query row index
builder.addIndicesLike(0, 2, 2); // key column index
builder.addLessEqual01(2, 1, 3); // key <= query
builder.addFillLike(0, 4, 1.0);
builder.addBinary(baseCaps.opcodes.sub, 4, 3, 5);
builder.addFillLike(0, 6, -10000.0);
builder.addBinary(baseCaps.opcodes.mul, 5, 6, 7);
builder.addBinary(baseCaps.opcodes.add, 0, 7, 8);
builder.setOutput(8);
const program = builder.compile();
assert(builder.numSteps() === 8, 'v9 causal program step count mismatch');

const scoresValues = [
  10, 11, 12, 13,
  20, 21, 22, 23,
  30, 31, 32, 33,
  40, 41, 42, 43,
];
const scores = tensor(scoresValues, [1, 4, 4, 1]);
const masked = program.run1(scores);
const expectedMasked = [
  10, -9989, -9988, -9987,
  20, 21, -9978, -9977,
  30, 31, 32, -9967,
  40, 41, 42, 43,
];
assert(JSON.stringify(values(masked)) === JSON.stringify(expectedMasked), `masked scores mismatch: ${JSON.stringify(values(masked))}`);

const plan = program.programPlan();
assert(plan[4] === 9, `expected v9 plan byte, got ${plan[4]}`);
const identity = program.programIdentity();
const replay = m.WasmMathProgramV9.fromPlan(plan);
assert(replay.programIdentity() === identity, 'v9 replay identity mismatch');
assert(JSON.stringify(Array.from(replay.programPlan())) === JSON.stringify(Array.from(plan)), 'v9 replay changed plan bytes');
assert(JSON.stringify(values(replay.run1(scores))) === JSON.stringify(expectedMasked), 'v9 replay result mismatch');

// Axis is canonical plan metadata and therefore changes identity.
const axis1Builder = new m.WasmMathProgramV9Builder(1, 2);
axis1Builder.addIndicesLike(0, 1, 1);
axis1Builder.setOutput(1);
const axis1 = axis1Builder.compile();
const axis2Builder = new m.WasmMathProgramV9Builder(1, 2);
axis2Builder.addIndicesLike(0, 1, 2);
axis2Builder.setOutput(1);
const axis2 = axis2Builder.compile();
assert(axis1.programIdentity() !== axis2.programIdentity(), 'changing index axis did not change identity');
expectThrow(() => {
  const b = new m.WasmMathProgramV9Builder(1, 2);
  try {
    b.addIndicesLike(0, 1, 4);
  } finally {
    b.free();
  }
}, 'indicesLike axis 4');

// Comparison semantics stay exact-shape and finite through v9 delegation.
const compareBuilder = new m.WasmMathProgramV9Builder(2, 3);
compareBuilder.addLessEqual01(0, 1, 2);
compareBuilder.setOutput(2);
const compareProgram = compareBuilder.compile();
const lhs = tensor([-2, 0, 3, 5], [1, 4, 1, 1]);
const rhs = tensor([-1, 0, 2, 6], [1, 4, 1, 1]);
const predicate = compareProgram.run2(lhs, rhs);
assert(JSON.stringify(values(predicate)) === JSON.stringify([1, 1, 0, 1]), 'v9 lessEqual01 predicate mismatch');
const scalar = tensor([1], [1, 1, 1, 1]);
expectThrow(() => compareProgram.run2(lhs, scalar), 'implicit broadcasting through v9 comparison');
const nonfinite = tensor([1, Number.POSITIVE_INFINITY, 3, 4], [1, 4, 1, 1]);
expectThrow(() => compareProgram.run2(nonfinite, rhs), 'non-finite v9 comparison input');

// Historical decoder isolation is bidirectional.
expectThrow(() => m.WasmMathProgramV8.fromPlan(plan), 'v8 decoder accepting v9 plan');
const v8Builder = new m.WasmMathProgramV8Builder(1, 2);
v8Builder.addFillLike(0, 1, 2.0);
v8Builder.setOutput(1);
const v8Program = v8Builder.compile();
expectThrow(() => m.WasmMathProgramV9.fromPlan(v8Program.programPlan()), 'v9 decoder accepting v8 plan');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v9 delegated index/predicate packaged-WASM proof',
  schema: caps.schema,
  planVersion: plan[4],
  causalMaskInSinglePlan: true,
  causalMaskSpecialCase: caps.causal_mask_special_case,
  selectWhereRequired: caps.select_where_required,
  numericPredicate: true,
  indexAxisIdentityBound: true,
  replayIdentityStable: true,
  v8DelegationFrozen: true,
  versionIsolation: true,
  implicitBroadcasting: caps.implicit_broadcasting,
  grantsAuthority: caps.grants_authority,
}, null, 2));

for (const value of [scores, masked, lhs, rhs, predicate, scalar, nonfinite]) value.free();
for (const value of [program, replay, axis1, axis2, compareProgram, v8Program]) value.free();
for (const value of [builder, axis1Builder, axis2Builder, compareBuilder, v8Builder]) value.free();
