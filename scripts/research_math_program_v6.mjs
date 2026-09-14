import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function tensor(values, shape = [1, values.length, 1, 1]) {
  return new m.WasmTensor(new Float32Array(values), new Uint32Array(shape));
}

function scalar(value) {
  return tensor([value], [1, 1, 1, 1]);
}

function verify(actual, expected, absTol = 1e-6, relTol = 1e-6) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(Array.from(actual.to_array())),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
}

function sameBytes(a, b) {
  return JSON.stringify(Array.from(a)) === JSON.stringify(Array.from(b));
}

function expectControlledFailure(fn, message) {
  let rejected = false;
  try {
    const unexpected = fn();
    if (unexpected?.free) unexpected.free();
  } catch {
    rejected = true;
  }
  assert(rejected, message);
}

const baseCaps = JSON.parse(m.mathProgramCapabilities());
const caps = JSON.parse(m.mathProgramV6Capabilities());
assert(caps.schema === 'burn-research.math-program.v6', 'v6 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v6', 'v6 plan schema mismatch');
assert(caps.min_inputs === 1 && caps.max_inputs === 8, 'v6 input bounds mismatch');
assert(caps.max_slots === 64, 'v6 max_slots mismatch');
assert(caps.legacy_step_validation === 'canonical_one_step_v5_replay', 'v6 legacy reuse mismatch');
assert(caps.value_sources.includes('fillLike'), 'fillLike capability missing');
assert(caps.fill_like_scalar === 'finite_canonical_f32_identity_bound', 'fillLike scalar contract mismatch');
assert(caps.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');
assert(caps.legacy_v1_v5_decoders_frozen === true, 'legacy decoder freeze mismatch');
assert(caps.registry_dependency === false, 'v6 must remain registry-independent');
assert(caps.mutable_state === false, 'v6 must remain stateless');

const resources = [];

// One-input affine formula: constants are program metadata, not host tensor inputs.
const affineBuilder = new m.WasmMathProgramV6Builder(1, 5);
affineBuilder.addFillLike(0, 1, 0.5);
affineBuilder.addBinary(baseCaps.opcodes.mul, 0, 1, 2);
affineBuilder.addFillLike(0, 3, 3.0);
affineBuilder.addBinary(baseCaps.opcodes.add, 2, 3, 4);
affineBuilder.setOutput(4);
const affine = affineBuilder.compile();
const x = tensor([2, 4, 6]);
const affineOut = affine.run1(x);
verify(affineOut, [4, 5, 6]);
assert(JSON.stringify(Array.from(affineOut.shape())) === JSON.stringify([1, 3, 1, 1]), 'affine output shape changed');
const affinePlan = affine.programPlan();
assert(affinePlan[4] === 6, `fillLike program did not use plan v6: ${affinePlan[4]}`);
const affineIdentity = affine.programIdentity();
const replay = m.WasmMathProgramV6.fromPlan(affinePlan);
assert(replay.programIdentity() === affineIdentity, 'v6 replay identity mismatch');
assert(sameBytes(replay.programPlan(), affinePlan), 'v6 replay plan bytes changed');
const replayOut = replay.run1(x);
verify(replayOut, [4, 5, 6]);
resources.push(affineBuilder, affine, x, affineOut, replay, replayOut);

// Scalar metadata must affect identity.
const twoBuilder = new m.WasmMathProgramV6Builder(1, 2);
twoBuilder.addFillLike(0, 1, 2.0);
twoBuilder.setOutput(1);
const two = twoBuilder.compile();
const threeBuilder = new m.WasmMathProgramV6Builder(1, 2);
threeBuilder.addFillLike(0, 1, 3.0);
threeBuilder.setOutput(1);
const three = threeBuilder.compile();
assert(two.programIdentity() !== three.programIdentity(), 'fillLike scalar did not change program identity');
resources.push(twoBuilder, two, threeBuilder, three);

// -0.0 is canonicalized to +0.0 and non-finite construction is atomic/reusable.
const zeroBuilder = new m.WasmMathProgramV6Builder(1, 2);
zeroBuilder.addFillLike(0, 1, 0.0);
zeroBuilder.setOutput(1);
const zero = zeroBuilder.compile();
const negZeroBuilder = new m.WasmMathProgramV6Builder(1, 2);
negZeroBuilder.addFillLike(0, 1, -0.0);
negZeroBuilder.setOutput(1);
const negZero = negZeroBuilder.compile();
assert(sameBytes(zero.programPlan(), negZero.programPlan()), '-0.0 was not canonicalized');

const atomicBuilder = new m.WasmMathProgramV6Builder(1, 3);
assert(atomicBuilder.numSteps() === 0, 'fresh v6 builder is not empty');
expectControlledFailure(
  () => atomicBuilder.addFillLike(0, 1, Number.NaN),
  'v6 accepted non-finite fillLike scalar',
);
assert(atomicBuilder.numSteps() === 0, 'failed fillLike construction mutated builder');
atomicBuilder.addFillLike(0, 1, 1.0);
assert(atomicBuilder.numSteps() === 1, 'builder not reusable after failed fillLike');
resources.push(zeroBuilder, zero, negZeroBuilder, negZero, atomicBuilder);

// Nontrivial reference shape is preserved exactly.
const shapeBuilder = new m.WasmMathProgramV6Builder(1, 2);
shapeBuilder.addFillLike(0, 1, -2.0);
shapeBuilder.setOutput(1);
const shapeProgram = shapeBuilder.compile();
const shaped = tensor([1, 2, 3, 4], [1, 2, 2, 1]);
const shapedOut = shapeProgram.run1(shaped);
verify(shapedOut, [-2, -2, -2, -2]);
assert(JSON.stringify(Array.from(shapedOut.shape())) === JSON.stringify([1, 2, 2, 1]), 'fillLike did not preserve reference shape');
resources.push(shapeBuilder, shapeProgram, shaped, shapedOut);

// Existing binary semantics still fail closed on implicit broadcasting.
const broadcastBuilder = new m.WasmMathProgramV6Builder(2, 3);
broadcastBuilder.addBinary(baseCaps.opcodes.add, 0, 1, 2);
broadcastBuilder.setOutput(2);
const broadcastProgram = broadcastBuilder.compile();
const vector = tensor([1, 2, 3]);
const scalarTen = scalar(10);
expectControlledFailure(
  () => broadcastProgram.run2(vector, scalarTen),
  'v6 binary add unexpectedly performed implicit broadcasting',
);
resources.push(broadcastBuilder, broadcastProgram, vector, scalarTen);

// Runtime arity failure does not mutate identity and valid retry remains usable.
expectControlledFailure(
  () => affine.run2(x, x),
  '1-input v6 program accepted run2',
);
assert(affine.programIdentity() === affineIdentity, 'failed arity run changed v6 identity');
const retryOut = affine.run1(x);
verify(retryOut, [4, 5, 6]);
resources.push(retryOut);

// Old decoder does not learn v6, and v6 decoder does not reinterpret v5 bytes.
expectControlledFailure(
  () => m.WasmMathProgramV5.fromPlan(affinePlan),
  'v5 decoder accepted a v6 plan',
);
const v5Builder = new m.WasmMathProgramV5Builder(3, 4);
v5Builder.addBinary(baseCaps.opcodes.add, 0, 1, 3);
v5Builder.setOutput(3);
const v5 = v5Builder.compile();
expectControlledFailure(
  () => m.WasmMathProgramV6.fromPlan(v5.programPlan()),
  'v6 decoder accepted a v5 plan',
);
resources.push(v5Builder, v5);

// Bounded run8 remains available through v6 for legacy math composition.
const maxBuilder = new m.WasmMathProgramV6Builder(8, 15);
maxBuilder.addBinary(baseCaps.opcodes.add, 0, 1, 8);
for (let inputSlot = 2, outputSlot = 9; inputSlot < 8; inputSlot += 1, outputSlot += 1) {
  maxBuilder.addBinary(baseCaps.opcodes.add, outputSlot - 1, inputSlot, outputSlot);
}
maxBuilder.setOutput(14);
const maxProgram = maxBuilder.compile();
const eightInputs = Array.from({ length: 8 }, (_, index) => scalar(index + 1));
const maxOut = maxProgram.run8(...eightInputs);
verify(maxOut, [36]);
resources.push(maxBuilder, maxProgram, ...eightInputs, maxOut);

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v6 identity-bound fillLike packaged-WASM proof',
  schema: caps.schema,
  planVersion: affinePlan[4],
  affineWithoutHostConstants: true,
  fillLikeShapeExact: true,
  scalarIdentityBound: true,
  replayIdentityStable: true,
  negativeZeroCanonical: true,
  nonFiniteAtomicFailure: true,
  implicitBroadcastRejected: true,
  legacyDecoderFrozen: caps.legacy_v1_v5_decoders_frozen,
  run1: true,
  run8: true,
}, null, 2));

for (const value of resources.reverse()) {
  if (value?.free) value.free();
}
