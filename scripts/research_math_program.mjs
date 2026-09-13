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
assert(caps.schema === 'burn-research.math-program.v2', 'schema mismatch');
assert(caps.v1_identity_compatibility === true, 'v1 identity compatibility must be explicit');
assert(caps.registry_dependency === false, 'must remain registry-independent');
assert(caps.mutable_state === false, 'must remain stateless');
assert(caps.parameterized_ops.scalar_v2.includes('clamp'), 'clamp capability missing');
assert(caps.parameterized_ops.scalar_v2.includes('cosineSimilarity'), 'cosine capability missing');

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

// Scalar parameters switch the canonical plan to v2 and survive replay.
const clampBuilder = new m.WasmMathProgramBuilder(1, 3);
clampBuilder.addClamp(0, 1, -1, 1);
clampBuilder.addUnary(caps.opcodes.sum, 1, 2);
clampBuilder.setOutput(2);
const clampProgram = clampBuilder.compile();
const clampPlan = clampProgram.programPlan();
assert(clampPlan[4] === 2, `parameterized program did not use plan v2: ${clampPlan[4]}`);
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

// Binary scalar parameter path: cosineSimilarity(epsilon).
const cosineBuilder = new m.WasmMathProgramBuilder(2, 3);
cosineBuilder.addCosineSimilarity(0, 1, 2, 1e-6);
cosineBuilder.setOutput(2);
const cosineProgram = cosineBuilder.compile();
assert(cosineProgram.programPlan()[4] === 2, 'cosine program must use plan v2');
const vectorA = tensor([1, 0], [1, 2, 1, 1]);
const vectorB = tensor([1, 0], [1, 2, 1, 1]);
const cosineOutput = cosineProgram.run2(vectorA, vectorB);
verify(cosineOutput, [1]);
const cosineReplay = m.WasmMathProgram.fromPlan(cosineProgram.programPlan());
assert(cosineReplay.programIdentity() === cosineProgram.programIdentity(), 'cosine replay identity mismatch');
const cosineReplayOutput = cosineReplay.run2(vectorA, vectorB);
verify(cosineReplayOutput, [1]);

// Failed parameter validation must not consume a step.
const invalidBuilder = new m.WasmMathProgramBuilder(1, 2);
let rejectedInvalidClamp = false;
try {
  invalidBuilder.addClamp(0, 1, Number.NaN, 1);
} catch {
  rejectedInvalidClamp = true;
}
assert(rejectedInvalidClamp, 'invalid clamp parameter was accepted');
assert(invalidBuilder.numSteps() === 0, 'failed scalar validation mutated builder state');

// Existing probability composition remains valid under the v2 capability envelope.
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
  task: 'Math Program v2 scalar-parameter packaged-WASM proof',
  schema: caps.schema,
  v1PlanCompatibility: plan[4] === 1,
  v2ScalarReplay: clampPlan[4] === 2,
  parameterIdentitySensitive: true,
  invalidParameterAtomicity: true,
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
  probabilityProgram,
]) value.free();
for (const value of [
  builder,
  clampBuilder,
  altClampBuilder,
  cosineBuilder,
  invalidBuilder,
  probabilityBuilder,
]) value.free();
