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

const baseCaps = JSON.parse(m.mathProgramCapabilities());
const caps = JSON.parse(m.mathProgramV4Capabilities());
assert(caps.schema === 'burn-research.math-program.v4', 'v4 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v4', 'v4 plan schema mismatch');
assert(caps.requires_select_axis === true, 'v4 must require selectAxis');
assert(caps.select_axis_opcode === 36, 'selectAxis opcode mismatch');
assert(caps.registry_dependency === false, 'v4 must remain registry-independent');
assert(caps.mutable_state === false, 'v4 must remain stateless');
assert(caps.core_execution_reuse === 'MathProgram.v1-v3.one-step', 'v4 core reuse contract mismatch');

// Packaged composition: selectAxis -> existing Statistics mean via proven core reuse.
const builder = new m.WasmMathProgramV4Builder(1, 3);
builder.addSelectAxis(0, 1, 1, u32([2, 0]));
builder.addUnary(baseCaps.opcodes.mean, 1, 2);
builder.setOutput(2);
const program = builder.compile();
const plan = program.programPlan();
assert(plan[4] === 4, `selectAxis program did not use plan v4: ${plan[4]}`);
const input = tensor([1, 2, 3], [1, 3, 1, 1]);
const output = program.run1(input);
verify(output, [2]);

// Canonical replay must preserve bytes, identity, and execution.
const replay = m.WasmMathProgramV4.fromPlan(plan);
assert(replay.programIdentity() === program.programIdentity(), 'v4 replay identity mismatch');
assert(
  JSON.stringify(Array.from(replay.programPlan())) === JSON.stringify(Array.from(plan)),
  'v4 replay plan bytes changed',
);
const replayOutput = replay.run1(input);
verify(replayOutput, [2]);

// Semantic metadata changes must change structural identity even if a commutative reducer
// happens to produce the same numerical output.
const altBuilder = new m.WasmMathProgramV4Builder(1, 3);
altBuilder.addSelectAxis(0, 1, 1, u32([0, 2]));
altBuilder.addUnary(baseCaps.opcodes.mean, 1, 2);
altBuilder.setOutput(2);
const altProgram = altBuilder.compile();
assert(
  altProgram.programIdentity() !== program.programIdentity(),
  'selectAxis index-order change must change program identity',
);

// Invalid metadata must reject atomically before consuming a builder step.
const invalidBuilder = new m.WasmMathProgramV4Builder(1, 2);
let rejectedAxis = false;
try {
  invalidBuilder.addSelectAxis(0, 1, 4, u32([0]));
} catch {
  rejectedAxis = true;
}
assert(rejectedAxis, 'invalid selectAxis axis was accepted');
assert(invalidBuilder.numSteps() === 0, 'invalid axis mutated builder state');
let rejectedEmpty = false;
try {
  invalidBuilder.addSelectAxis(0, 1, 1, u32([]));
} catch {
  rejectedEmpty = true;
}
assert(rejectedEmpty, 'empty selectAxis indices were accepted');
assert(invalidBuilder.numSteps() === 0, 'empty indices mutated builder state');

// Input-dependent index failure is controlled; immutable identity survives and a later valid
// execution succeeds on the same program instance.
const reusableBuilder = new m.WasmMathProgramV4Builder(1, 2);
reusableBuilder.addSelectAxis(0, 1, 1, u32([3]));
reusableBuilder.setOutput(1);
const reusableProgram = reusableBuilder.compile();
const reusableIdentity = reusableProgram.programIdentity();
const tooSmall = tensor([1, 2], [1, 2, 1, 1]);
let rejectedRuntimeIndex = false;
try {
  const unexpected = reusableProgram.run1(tooSmall);
  unexpected.free();
} catch {
  rejectedRuntimeIndex = true;
}
assert(rejectedRuntimeIndex, 'out-of-bounds runtime selectAxis index was accepted');
assert(reusableProgram.programIdentity() === reusableIdentity, 'failed run changed v4 identity');
const valid = tensor([1, 2, 3, 4], [1, 4, 1, 1]);
const selected = reusableProgram.run1(valid);
verify(selected, [4]);
assert(reusableProgram.programIdentity() === reusableIdentity, 'valid retry changed v4 identity');

// v4 decoder must reject lower-version plans rather than silently reinterpret them.
const legacyBuilder = new m.WasmMathProgramBuilder(1, 2);
legacyBuilder.addUnary(baseCaps.opcodes.mean, 0, 1);
legacyBuilder.setOutput(1);
const legacyProgram = legacyBuilder.compile();
let rejectedLegacyPlan = false;
try {
  const unexpected = m.WasmMathProgramV4.fromPlan(legacyProgram.programPlan());
  unexpected.free();
} catch {
  rejectedLegacyPlan = true;
}
assert(rejectedLegacyPlan, 'v4 decoder accepted a lower-version plan');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v4 selectAxis packaged-WASM proof',
  schema: caps.schema,
  planVersion: plan[4],
  replayIdentityStable: true,
  metadataIdentitySensitive: true,
  invalidMetadataAtomicity: true,
  runtimeFailureReusable: true,
  lowerVersionFailClosed: true,
  coreExecutionReuse: caps.core_execution_reuse,
  registryDependency: caps.registry_dependency,
  mutableState: caps.mutable_state,
}, null, 2));

for (const value of [
  input,
  output,
  replayOutput,
  tooSmall,
  valid,
  selected,
]) value.free();
for (const value of [
  program,
  replay,
  altProgram,
  reusableProgram,
  legacyProgram,
]) value.free();
for (const value of [
  builder,
  altBuilder,
  invalidBuilder,
  reusableBuilder,
  legacyBuilder,
]) value.free();
