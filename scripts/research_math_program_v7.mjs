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

function verify(actual, expected, absTol = 2e-6, relTol = 2e-6) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(Array.from(actual.to_array())),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
}

const baseCaps = JSON.parse(m.mathProgramCapabilities());
const caps = JSON.parse(m.mathProgramV7Capabilities());
assert(caps.schema === 'burn-research.math-program.v7', 'v7 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v7', 'v7 plan schema mismatch');
assert(caps.min_inputs === 1 && caps.max_inputs === 8, 'v7 input bounds mismatch');
assert(caps.max_slots === 64, 'v7 max_slots mismatch');
assert(caps.expand_like_rule === 'per_axis_equal_or_singleton', 'expandLike rule mismatch');
assert(caps.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');
assert(caps.legacy_v1_v6_decoders_frozen === true, 'v1-v6 decoders must remain frozen');
assert(caps.registry_dependency === false, 'v7 must remain registry-independent');
assert(caps.mutable_state === false, 'v7 must remain stateless');

// One-host-input decomposed softmax: exp -> sum -> explicit expandLike -> div.
const softmaxBuilder = new m.WasmMathProgramV7Builder(1, 5);
softmaxBuilder.addUnary(baseCaps.opcodes.exp, 0, 1);
softmaxBuilder.addUnary(baseCaps.opcodes.sum, 1, 2);
softmaxBuilder.addExpandLike(2, 1, 3);
softmaxBuilder.addBinary(baseCaps.opcodes.div, 1, 3, 4);
softmaxBuilder.setOutput(4);
const softmaxProgram = softmaxBuilder.compile();

const input = tensor([1, 2, 3, 1, 1, 1], [2, 3, 1, 1]);
const output = softmaxProgram.run1(input);
const e1 = Math.exp(1);
const e2 = Math.exp(2);
const e3 = Math.exp(3);
const sum = e1 + e2 + e3;
const expected = [e1 / sum, e2 / sum, e3 / sum, 1 / 3, 1 / 3, 1 / 3];
verify(output, expected);
const outValues = Array.from(output.to_array());
assert(Math.abs(outValues.slice(0, 3).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'batch 0 softmax mass mismatch');
assert(Math.abs(outValues.slice(3, 6).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'batch 1 softmax mass mismatch');

const plan = softmaxProgram.programPlan();
assert(plan[4] === 7, `expected v7 plan byte, got ${plan[4]}`);
const identity = softmaxProgram.programIdentity();
const replay = m.WasmMathProgramV7.fromPlan(plan);
assert(replay.programIdentity() === identity, 'v7 replay identity mismatch');
assert(JSON.stringify(Array.from(replay.programPlan())) === JSON.stringify(Array.from(plan)), 'v7 replay plan bytes changed');
const replayOutput = replay.run1(input);
verify(replayOutput, expected);

// Runtime mismatch must fail closed without mutating program identity; retry with valid singleton source.
const expandBuilder = new m.WasmMathProgramV7Builder(2, 3);
expandBuilder.addExpandLike(0, 1, 2);
expandBuilder.setOutput(2);
const expandProgram = expandBuilder.compile();
const expandIdentity = expandProgram.programIdentity();
const badSource = tensor([1, 2], [1, 2, 1, 1]);
const badReference = tensor([0, 0, 0], [1, 3, 1, 1]);
let mismatchRejected = false;
try {
  const unexpected = expandProgram.run2(badSource, badReference);
  unexpected.free();
} catch {
  mismatchRejected = true;
}
assert(mismatchRejected, 'expandLike accepted mismatched non-singleton source');
assert(expandProgram.programIdentity() === expandIdentity, 'failed expandLike run changed identity');

const goodSource = scalar(7);
const goodReference = tensor([0, 0, 0], [1, 3, 1, 1]);
const expanded = expandProgram.run2(goodSource, goodReference);
verify(expanded, [7, 7, 7]);
assert(expandProgram.programIdentity() === expandIdentity, 'valid retry changed identity');

// v6 fillLike remains composable inside v7.
const fillBuilder = new m.WasmMathProgramV7Builder(1, 4);
fillBuilder.addFillLike(0, 1, 0.5);
fillBuilder.addBinary(baseCaps.opcodes.mul, 0, 1, 2);
fillBuilder.addFillLike(0, 3, 3.0);
fillBuilder.setOutput(2);
const fillProgram = fillBuilder.compile();
const fillInput = tensor([2, 4, 6]);
const fillOut = fillProgram.run1(fillInput);
verify(fillOut, [1, 2, 3]);

// Historical decoder isolation is bidirectional.
let v6RejectedV7 = false;
try {
  const unexpected = m.WasmMathProgramV6.fromPlan(plan);
  unexpected.free();
} catch {
  v6RejectedV7 = true;
}
assert(v6RejectedV7, 'v6 decoder accepted a v7 plan');

const v6Builder = new m.WasmMathProgramV6Builder(1, 2);
v6Builder.addFillLike(0, 1, 2.0);
v6Builder.setOutput(1);
const v6Program = v6Builder.compile();
let v7RejectedV6 = false;
try {
  const unexpected = m.WasmMathProgramV7.fromPlan(v6Program.programPlan());
  unexpected.free();
} catch {
  v7RejectedV6 = true;
}
assert(v7RejectedV6, 'v7 decoder accepted a v6 plan');

// Bounded 8-input legacy composition remains available through v7.
const eightBuilder = new m.WasmMathProgramV7Builder(8, 15);
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
  task: 'Math Program v7 explicit runtime expandLike packaged-WASM proof',
  schema: caps.schema,
  planVersion: plan[4],
  decomposedSoftmax: true,
  batchMassPreserved: true,
  replayIdentityStable: true,
  runtimeMismatchFailClosedAndReusable: true,
  fillLikeDelegationPreserved: true,
  versionIsolation: true,
  run8BoundedMaximum: true,
  implicitBroadcasting: caps.implicit_broadcasting,
}, null, 2));

for (const value of [input, output, replayOutput, badSource, badReference, goodSource, goodReference, expanded, fillInput, fillOut, ...eightInputs, eightOut]) value.free();
for (const value of [softmaxProgram, replay, expandProgram, fillProgram, v6Program, eightProgram]) value.free();
for (const value of [softmaxBuilder, expandBuilder, fillBuilder, v6Builder, eightBuilder]) value.free();
