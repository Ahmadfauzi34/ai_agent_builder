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

function verify(actual, expected, absTol = 1e-6, relTol = 1e-6) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(Array.from(actual.to_array())),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
}

const caps = JSON.parse(m.mathProgramCapabilities());
assert(caps.schema === 'burn-research.math-program.v3', 'schema mismatch');
assert(caps.v1_identity_compatibility === true, 'v1 identity compatibility must be explicit');
assert(caps.v2_identity_compatibility === true, 'v2 identity compatibility must be explicit');
assert(caps.registry_dependency === false, 'must remain registry-independent');
assert(caps.mutable_state === false, 'must remain stateless');
assert(caps.parameterized_ops.scalar_v2.includes('clamp'), 'clamp capability missing');
assert(caps.parameterized_ops.scalar_v2.includes('cosineSimilarity'), 'cosine capability missing');
assert(caps.parameterized_ops.fixed_rank4_v3.includes('reshape'), 'reshape capability missing');
assert(caps.parameterized_ops.fixed_rank4_v3.includes('permute'), 'permute capability missing');
assert(caps.parameterized_ops.fixed_rank4_v3.includes('slice'), 'slice capability missing');

// Existing non-parameterized programs must remain canonical v1 plans.
const builder = new m.WasmMathProgramBuilder(1, 3);
builder.addUnary(caps.opcodes.sqrt, 0, 1);
builder.addUnary(caps.opcodes.mean, 1, 2);
builder.setOutput(2);
const program = builder.compile();
const input = tensor([1, 4, 9], [1, 3, 1, 1]);
const output = program.run1(input);
verify(output, [2]);

const plan = program.programPlan();
assert(plan[4] === 1, `non-parameterized program changed plan version: ${plan[4]}`);
const replay = m.WasmMathProgram.fromPlan(plan);
assert(replay.programIdentity() === program.programIdentity(), 'v1 replay identity mismatch');
const replayOutput = replay.run1(input);
verify(replayOutput, [2]);

// Scalar-only programs remain canonical v2 plans and survive replay.
const clampBuilder = new m.WasmMathProgramBuilder(1, 3);
clampBuilder.addClamp(0, 1, -1, 1);
clampBuilder.addUnary(caps.opcodes.sum, 1, 2);
clampBuilder.setOutput(2);
const clampProgram = clampBuilder.compile();
const clampPlan = clampProgram.programPlan();
assert(clampPlan[4] === 2, `scalar program changed plan version: ${clampPlan[4]}`);
const clampReplay = m.WasmMathProgram.fromPlan(clampPlan);
assert(clampReplay.programIdentity() === clampProgram.programIdentity(), 'v2 replay identity mismatch');
const clampInput = tensor([-2, 0.5, 3], [1, 3, 1, 1]);
const clampOutput = clampProgram.run1(clampInput);
const clampReplayOutput = clampReplay.run1(clampInput);
verify(clampOutput, [0.5]);
verify(clampReplayOutput, [0.5]);

const altClampBuilder = new m.WasmMathProgramBuilder(1, 3);
altClampBuilder.addClamp(0, 1, -2, 2);
altClampBuilder.addUnary(caps.opcodes.sum, 1, 2);
altClampBuilder.setOutput(2);
const altClampProgram = altClampBuilder.compile();
assert(
  altClampProgram.programIdentity() !== clampProgram.programIdentity(),
  'scalar parameter change must change program identity',
);

// Binary scalar parameter path remains v2.
const cosineBuilder = new m.WasmMathProgramBuilder(2, 3);
cosineBuilder.addCosineSimilarity(0, 1, 2, 1e-6);
cosineBuilder.setOutput(2);
const cosineProgram = cosineBuilder.compile();
assert(cosineProgram.programPlan()[4] === 2, 'cosine program must remain plan v2');
const vectorA = tensor([1, 0], [1, 2, 1, 1]);
const vectorB = tensor([1, 0], [1, 2, 1, 1]);
const cosineOutput = cosineProgram.run2(vectorA, vectorB);
verify(cosineOutput, [1]);
const cosineReplay = m.WasmMathProgram.fromPlan(cosineProgram.programPlan());
assert(cosineReplay.programIdentity() === cosineProgram.programIdentity(), 'cosine replay identity mismatch');
const cosineReplayOutput = cosineReplay.run2(vectorA, vectorB);
verify(cosineReplayOutput, [1]);

// Fixed rank-4 metadata switches the canonical plan to v3.
const shapeBuilder = new m.WasmMathProgramBuilder(1, 4);
shapeBuilder.addReshape(0, 1, u32([1, 1, 3, 2]));
shapeBuilder.addPermute(1, 2, u32([0, 1, 3, 2]));
shapeBuilder.addSlice(2, 3, u32([0, 0, 0, 1]), u32([1, 1, 2, 3]));
shapeBuilder.setOutput(3);
const shapeProgram = shapeBuilder.compile();
const shapePlan = shapeProgram.programPlan();
assert(shapePlan[4] === 3, `fixed-shape program did not use plan v3: ${shapePlan[4]}`);
const shapeInput = tensor([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
const shapeOutput = shapeProgram.run1(shapeInput);
assert(JSON.stringify(Array.from(shapeOutput.shape())) === JSON.stringify([1, 1, 2, 2]), 'shape output mismatch');
verify(shapeOutput, [3, 5, 4, 6]);
const shapeReplay = m.WasmMathProgram.fromPlan(shapePlan);
assert(shapeReplay.programIdentity() === shapeProgram.programIdentity(), 'v3 replay identity mismatch');
const shapeReplayOutput = shapeReplay.run1(shapeInput);
verify(shapeReplayOutput, [3, 5, 4, 6]);

const alternateShapeBuilder = new m.WasmMathProgramBuilder(1, 2);
alternateShapeBuilder.addReshape(0, 1, u32([1, 1, 2, 3]));
alternateShapeBuilder.setOutput(1);
const alternateShapeProgram = alternateShapeBuilder.compile();
const alternateShapeBuilder2 = new m.WasmMathProgramBuilder(1, 2);
alternateShapeBuilder2.addReshape(0, 1, u32([1, 1, 3, 2]));
alternateShapeBuilder2.setOutput(1);
const alternateShapeProgram2 = alternateShapeBuilder2.compile();
assert(
  alternateShapeProgram.programIdentity() !== alternateShapeProgram2.programIdentity(),
  'shape metadata change must change program identity',
);

// Failed metadata validation must not consume a step.
const invalidBuilder = new m.WasmMathProgramBuilder(1, 2);
let rejectedInvalidShape = false;
try {
  invalidBuilder.addPermute(0, 1, u32([0, 1, 1, 3]));
} catch {
  rejectedInvalidShape = true;
}
assert(rejectedInvalidShape, 'invalid permute metadata was accepted');
assert(invalidBuilder.numSteps() === 0, 'failed shape validation mutated builder state');

// Input-dependent shape failure is controlled and the immutable program stays reusable.
const reusableBuilder = new m.WasmMathProgramBuilder(1, 2);
reusableBuilder.addReshape(0, 1, u32([1, 1, 2, 2]));
reusableBuilder.setOutput(1);
const reusableProgram = reusableBuilder.compile();
const reusableIdentity = reusableProgram.programIdentity();
const badShapeInput = tensor([1, 2, 3], [1, 3, 1, 1]);
let rejectedRuntimeShape = false;
try {
  const unexpected = reusableProgram.run1(badShapeInput);
  unexpected.free();
} catch {
  rejectedRuntimeShape = true;
}
assert(rejectedRuntimeShape, 'runtime reshape mismatch was accepted');
assert(reusableProgram.programIdentity() === reusableIdentity, 'failed run changed program identity');
const goodShapeInput = tensor([1, 2, 3, 4], [1, 4, 1, 1]);
const goodShapeOutput = reusableProgram.run1(goodShapeInput);
verify(goodShapeOutput, [1, 2, 3, 4]);
assert(reusableProgram.programIdentity() === reusableIdentity, 'reused program identity changed');

// Existing probability composition remains valid under the v3 capability envelope.
const probabilityBuilder = new m.WasmMathProgramBuilder(1, 3);
probabilityBuilder.addUnary(caps.opcodes.normalize, 0, 1);
probabilityBuilder.addUnary(caps.opcodes.entropy, 1, 2);
probabilityBuilder.setOutput(2);
const probabilityProgram = probabilityBuilder.compile();
const weights = tensor([1, 1], [1, 2, 1, 1]);
const entropy = probabilityProgram.run1(weights);
verify(entropy, [Math.log(2)], 2e-6, 2e-6);

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v3 fixed-rank4 packaged-WASM proof',
  schema: caps.schema,
  v1PlanCompatibility: plan[4] === 1,
  v2ScalarCompatibility: clampPlan[4] === 2,
  v3FixedShapeReplay: shapePlan[4] === 3,
  shapeIdentitySensitive: true,
  invalidMetadataAtomicity: true,
  runtimeFailureReusable: true,
  registryDependency: caps.registry_dependency,
  mutableState: caps.mutable_state,
  referenceVerification: 'mathVerifyVectors',
}, null, 2));

for (const value of [
  input,
  output,
  replayOutput,
  clampInput,
  clampOutput,
  clampReplayOutput,
  vectorA,
  vectorB,
  cosineOutput,
  cosineReplayOutput,
  shapeInput,
  shapeOutput,
  shapeReplayOutput,
  badShapeInput,
  goodShapeInput,
  goodShapeOutput,
  weights,
  entropy,
]) value.free();
for (const value of [
  program,
  replay,
  clampProgram,
  clampReplay,
  altClampProgram,
  cosineProgram,
  cosineReplay,
  shapeProgram,
  shapeReplay,
  alternateShapeProgram,
  alternateShapeProgram2,
  reusableProgram,
  probabilityProgram,
]) value.free();
for (const value of [
  builder,
  clampBuilder,
  altClampBuilder,
  cosineBuilder,
  shapeBuilder,
  alternateShapeBuilder,
  alternateShapeBuilder2,
  invalidBuilder,
  reusableBuilder,
  probabilityBuilder,
]) value.free();
