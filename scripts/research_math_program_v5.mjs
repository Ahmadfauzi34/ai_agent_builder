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
const caps = JSON.parse(m.mathProgramV5Capabilities());
assert(caps.schema === 'burn-research.math-program.v5', 'v5 schema mismatch');
assert(caps.plan_schema === 'burn-research.math-program-plan.v5', 'v5 plan schema mismatch');
assert(caps.min_inputs === 3, 'v5 min_inputs mismatch');
assert(caps.max_inputs === 8, 'v5 max_inputs mismatch');
assert(caps.max_slots === 64, 'v5 max_slots mismatch');
assert(caps.legacy_v1_v4_input_contract_frozen === true, 'legacy contract must remain frozen');
assert(caps.core_execution_reuse === 'MathProgram.v1-v3.one-step', 'v5 core reuse mismatch');
assert(caps.select_axis_execution_reuse === 'MathProgram.v4.one-step', 'v5 selectAxis reuse mismatch');
assert(caps.registry_dependency === false, 'v5 must remain registry-independent');
assert(caps.mutable_state === false, 'v5 must remain stateless');

const fanInBuilder = new m.WasmMathProgramV5Builder(3, 5);
fanInBuilder.addBinary(baseCaps.opcodes.add, 0, 1, 3);
fanInBuilder.addBinary(baseCaps.opcodes.add, 3, 2, 4);
fanInBuilder.setOutput(4);
const fanIn = fanInBuilder.compile();
const a = tensor([1, 2]);
const b = tensor([3, 4]);
const c = tensor([5, 6]);
const fanInOut = fanIn.run3(a, b, c);
verify(fanInOut, [9, 12]);
const fanInPlan = fanIn.programPlan();
assert(fanInPlan[4] === 5, `multi-input program did not use plan v5: ${fanInPlan[4]}`);

const fanInReplay = m.WasmMathProgramV5.fromPlan(fanInPlan);
assert(fanInReplay.programIdentity() === fanIn.programIdentity(), 'v5 replay identity mismatch');
assert(JSON.stringify(Array.from(fanInReplay.programPlan())) === JSON.stringify(Array.from(fanInPlan)), 'v5 replay plan bytes changed');
const replayOut = fanInReplay.run3(a, b, c);
verify(replayOut, [9, 12]);

const fanInIdentity = fanIn.programIdentity();
let rejectedRun4 = false;
try {
  const unexpected = fanIn.run4(a, b, c, a);
  unexpected.free();
} catch {
  rejectedRun4 = true;
}
assert(rejectedRun4, '3-input program accepted run4');
assert(fanIn.programIdentity() === fanInIdentity, 'failed arity run changed v5 identity');
const retryOut = fanIn.run3(a, b, c);
verify(retryOut, [9, 12]);

const maxBuilder = new m.WasmMathProgramV5Builder(8, 15);
maxBuilder.addBinary(baseCaps.opcodes.add, 0, 1, 8);
maxBuilder.addBinary(baseCaps.opcodes.add, 8, 2, 9);
maxBuilder.addBinary(baseCaps.opcodes.add, 9, 3, 10);
maxBuilder.addBinary(baseCaps.opcodes.add, 10, 4, 11);
maxBuilder.addBinary(baseCaps.opcodes.add, 11, 5, 12);
maxBuilder.addBinary(baseCaps.opcodes.add, 12, 6, 13);
maxBuilder.addBinary(baseCaps.opcodes.add, 13, 7, 14);
maxBuilder.setOutput(14);
const maxProgram = maxBuilder.compile();
const scalars = Array.from({ length: 8 }, (_, index) => scalar(index + 1));
const maxOut = maxProgram.run8(...scalars);
verify(maxOut, [36]);

let rejectedNineInputs = false;
try {
  const unexpected = new m.WasmMathProgramV5Builder(9, 10);
  unexpected.free();
} catch {
  rejectedNineInputs = true;
}
assert(rejectedNineInputs, 'v5 accepted 9 external inputs');

const selectBuilder = new m.WasmMathProgramV5Builder(3, 6);
selectBuilder.addSelectAxis(0, 3, 1, u32([2, 0]));
selectBuilder.addBinary(baseCaps.opcodes.add, 1, 2, 4);
selectBuilder.addBinary(baseCaps.opcodes.add, 3, 4, 5);
selectBuilder.setOutput(5);
const selectProgram = selectBuilder.compile();
const selectA = tensor([10, 20, 30]);
const selectB = tensor([1, 1]);
const selectC = tensor([2, 2]);
const selectOut = selectProgram.run3(selectA, selectB, selectC);
verify(selectOut, [33, 13]);

const reusableBuilder = new m.WasmMathProgramV5Builder(3, 4);
reusableBuilder.addSelectAxis(0, 3, 1, u32([3]));
reusableBuilder.setOutput(3);
const reusableProgram = reusableBuilder.compile();
const reusableIdentity = reusableProgram.programIdentity();
const tooSmall = tensor([1, 2]);
const ignoredB = scalar(0);
const ignoredC = scalar(0);
let rejectedRuntimeIndex = false;
try {
  const unexpected = reusableProgram.run3(tooSmall, ignoredB, ignoredC);
  unexpected.free();
} catch {
  rejectedRuntimeIndex = true;
}
assert(rejectedRuntimeIndex, 'out-of-bounds runtime selectAxis index was accepted');
assert(reusableProgram.programIdentity() === reusableIdentity, 'failed run changed v5 identity');
const largeEnough = tensor([1, 2, 3, 4]);
const selected = reusableProgram.run3(largeEnough, ignoredB, ignoredC);
verify(selected, [4]);
assert(reusableProgram.programIdentity() === reusableIdentity, 'valid retry changed v5 identity');

const legacyBuilder = new m.WasmMathProgramBuilder(1, 2);
legacyBuilder.addUnary(baseCaps.opcodes.mean, 0, 1);
legacyBuilder.setOutput(1);
const legacyProgram = legacyBuilder.compile();
let rejectedLegacyPlan = false;
try {
  const unexpected = m.WasmMathProgramV5.fromPlan(legacyProgram.programPlan());
  unexpected.free();
} catch {
  rejectedLegacyPlan = true;
}
assert(rejectedLegacyPlan, 'v5 decoder accepted a lower-version plan');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v5 bounded multi-input packaged-WASM proof',
  schema: caps.schema,
  planVersion: fanInPlan[4],
  minInputs: caps.min_inputs,
  maxInputs: caps.max_inputs,
  run3FanIn: true,
  run8BoundedMaximum: true,
  replayIdentityStable: true,
  wrongArityFailClosedAndReusable: true,
  runtimeFailureReusable: true,
  lowerVersionFailClosed: true,
  legacyContractFrozen: caps.legacy_v1_v4_input_contract_frozen,
  coreExecutionReuse: caps.core_execution_reuse,
  selectAxisExecutionReuse: caps.select_axis_execution_reuse,
}, null, 2));

for (const value of [a, b, c, fanInOut, replayOut, retryOut, ...scalars, maxOut, selectA, selectB, selectC, selectOut, tooSmall, ignoredB, ignoredC, largeEnough, selected]) value.free();
for (const value of [fanIn, fanInReplay, maxProgram, selectProgram, reusableProgram, legacyProgram]) value.free();
for (const value of [fanInBuilder, maxBuilder, selectBuilder, reusableBuilder, legacyBuilder]) value.free();
