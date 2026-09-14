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

function values(t) {
  return Array.from(t.to_array());
}

function verify(actual, expected, absTol = 3e-6, relTol = 3e-6) {
  const actualValues = actual?.to_array ? values(actual) : Array.from(actual);
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(actualValues),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
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
const caps = JSON.parse(m.mathProgramV8Capabilities());
assert(caps.schema === 'burn-research.math-program.v8', 'v8 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v8', 'v8 plan schema mismatch');
assert(caps.min_inputs === 1 && caps.max_inputs === 8, 'v8 input bounds mismatch');
assert(caps.max_slots === 64, 'v8 max_slots mismatch');
assert(JSON.stringify(caps.generic_reduction_ops) === JSON.stringify(['sumAxis', 'meanAxis', 'minAxis', 'maxAxis']), 'v8 reduction ops mismatch');
assert(caps.reduction_axis_binding === 'identity_bound_plan_metadata', 'v8 axis binding mismatch');
assert(caps.reduction_keepdim_rank4 === true, 'v8 keepdim rank-4 contract mismatch');
assert(caps.zero_sized_dimensions === false, 'v8 zero-sized dimensions must fail closed');
assert(caps.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');
assert(caps.legacy_v1_v7_decoders_frozen === true, 'v1-v7 decoders must remain frozen');
assert(caps.registry_dependency === false, 'v8 must remain registry-independent');
assert(caps.mutable_state === false, 'v8 must remain stateless');

// Stable softmax fully inside one v8 plan:
// maxAxis -> expandLike -> sub -> exp -> sumAxis -> expandLike -> div.
const softmaxBuilder = new m.WasmMathProgramV8Builder(1, 8);
softmaxBuilder.addMaxAxis(0, 1, 2);
softmaxBuilder.addExpandLike(1, 0, 2);
softmaxBuilder.addBinary(baseCaps.opcodes.sub, 0, 2, 3);
softmaxBuilder.addUnary(baseCaps.opcodes.exp, 3, 4);
softmaxBuilder.addSumAxis(4, 5, 2);
softmaxBuilder.addExpandLike(5, 4, 6);
softmaxBuilder.addBinary(baseCaps.opcodes.div, 4, 6, 7);
softmaxBuilder.setOutput(7);
const softmaxProgram = softmaxBuilder.compile();
assert(softmaxBuilder.numSteps() === 7, 'v8 softmax step count mismatch');

const input = tensor([1000, 1001, 1002, 1, 2, 3], [1, 2, 3, 1]);
const output = softmaxProgram.run1(input);
const e0 = Math.exp(-2);
const e1 = Math.exp(-1);
const e2 = 1;
const den = e0 + e1 + e2;
const group = [e0 / den, e1 / den, e2 / den];
const expected = [...group, ...group];
verify(output, expected);
const outValues = values(output);
assert(outValues.every(Number.isFinite), 'stable softmax produced non-finite output');
assert(Math.abs(outValues.slice(0, 3).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'softmax group 0 mass mismatch');
assert(Math.abs(outValues.slice(3, 6).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'softmax group 1 mass mismatch');

const plan = softmaxProgram.programPlan();
assert(plan[4] === 8, `expected v8 plan byte, got ${plan[4]}`);
const identity = softmaxProgram.programIdentity();
const replay = m.WasmMathProgramV8.fromPlan(plan);
assert(replay.programIdentity() === identity, 'v8 replay identity mismatch');
assert(JSON.stringify(Array.from(replay.programPlan())) === JSON.stringify(Array.from(plan)), 'v8 replay plan bytes changed');
const replayOutput = replay.run1(input);
verify(replayOutput, expected);

// Axis is identity-bound metadata, not runtime state or shape inference.
const axis1Builder = new m.WasmMathProgramV8Builder(1, 2);
axis1Builder.addMaxAxis(0, 1, 1);
axis1Builder.setOutput(1);
const axis1Program = axis1Builder.compile();
const axis2Builder = new m.WasmMathProgramV8Builder(1, 2);
axis2Builder.addMaxAxis(0, 1, 2);
axis2Builder.setOutput(1);
const axis2Program = axis2Builder.compile();
assert(axis1Program.programIdentity() !== axis2Program.programIdentity(), 'changing reduction axis did not change v8 identity');
expectThrow(() => {
  const b = new m.WasmMathProgramV8Builder(1, 2);
  try {
    b.addMaxAxis(0, 1, 4);
  } finally {
    b.free();
  }
}, 'axis 4');

// Runtime failure must be atomic and reusable.
const failIdentity = axis1Program.programIdentity();
const nonfinite = tensor([1, Number.POSITIVE_INFINITY], [1, 2, 1, 1]);
expectThrow(() => axis1Program.run1(nonfinite), 'non-finite reduction input');
assert(axis1Program.programIdentity() === failIdentity, 'failed v8 run changed identity');
const finite = tensor([1, 2], [1, 2, 1, 1]);
const finiteOut = axis1Program.run1(finite);
verify(finiteOut, [2]);
assert(axis1Program.programIdentity() === failIdentity, 'valid retry changed v8 identity');

// Historical decoder isolation is bidirectional.
expectThrow(() => m.WasmMathProgramV7.fromPlan(plan), 'v7 decoder accepting v8 plan');
const v7Builder = new m.WasmMathProgramV7Builder(1, 2);
v7Builder.addFillLike(0, 1, 2.0);
v7Builder.setOutput(1);
const v7Program = v7Builder.compile();
expectThrow(() => m.WasmMathProgramV8.fromPlan(v7Program.programPlan()), 'v8 decoder accepting v7 plan');

// Bounded 8-input legacy composition remains available through v8.
const eightBuilder = new m.WasmMathProgramV8Builder(8, 15);
eightBuilder.addBinary(baseCaps.opcodes.add, 0, 1, 8);
eightBuilder.addBinary(baseCaps.opcodes.add, 8, 2, 9);
eightBuilder.addBinary(baseCaps.opcodes.add, 9, 3, 10);
eightBuilder.addBinary(baseCaps.opcodes.add, 10, 4, 11);
eightBuilder.addBinary(baseCaps.opcodes.add, 11, 5, 12);
eightBuilder.addBinary(baseCaps.opcodes.add, 12, 6, 13);
eightBuilder.addBinary(baseCaps.opcodes.add, 13, 7, 14);
eightBuilder.setOutput(14);
const eightProgram = eightBuilder.compile();
const eightInputs = Array.from({ length: 8 }, (_, i) => scalar(i + 1));
const eightOut = eightProgram.run8(...eightInputs);
verify(eightOut, [36]);

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v8 identity-bound generic reduction packaged-WASM proof',
  schema: caps.schema,
  planVersion: plan[4],
  stableSoftmaxInSinglePlan: true,
  reductionAxisIdentityBound: true,
  replayIdentityStable: true,
  runtimeFailureAtomicAndReusable: true,
  versionIsolation: true,
  run8BoundedMaximum: true,
  implicitBroadcasting: caps.implicit_broadcasting,
}, null, 2));

for (const value of [input, output, replayOutput, nonfinite, finite, finiteOut, ...eightInputs, eightOut]) value.free();
for (const value of [softmaxProgram, replay, axis1Program, axis2Program, v7Program, eightProgram]) value.free();
for (const value of [softmaxBuilder, axis1Builder, axis2Builder, v7Builder, eightBuilder]) value.free();
