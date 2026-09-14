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

const caps = JSON.parse(m.mathProgramCapabilities());
const v5Caps = JSON.parse(m.mathProgramV5Capabilities());

assert(caps.schema === 'burn-research.math-program.v3', 'legacy math program schema mismatch');
assert(v5Caps.schema === 'burn-research.math-program.v5', 'v5 schema mismatch');
assert(v5Caps.min_inputs === 3 && v5Caps.max_inputs === 8, 'v5 external-input bound changed');
assert(v5Caps.max_slots === 64, 'v5 slot bound changed');
assert(v5Caps.legacy_v1_v4_input_contract_frozen === true, 'legacy input contract must remain frozen');

const resources = [];

// Workload A: semantic-memory style cosine program reused across candidates.
const cosineBuilder = new m.WasmMathProgramBuilder(2, 3);
cosineBuilder.addCosineSimilarity(0, 1, 2, 1e-6);
cosineBuilder.setOutput(2);
const cosineProgram = cosineBuilder.compile();
const query = tensor([0.8, 0.4, 0.2, 0.1]);
const near = tensor([0.79, 0.41, 0.19, 0.1]);
const far = tensor([0.1, 0.2, 0.3, 0.9]);
const nearScore = cosineProgram.run2(query, near);
const farScore = cosineProgram.run2(query, far);
const nearValue = nearScore.to_array()[0];
const farValue = farScore.to_array()[0];
assert(nearValue > farValue, 'cosine workload did not rank the near vector first');
const cosineIdentity = cosineProgram.programIdentity();
const cosineReplay = m.WasmMathProgram.fromPlan(cosineProgram.programPlan());
assert(cosineReplay.programIdentity() === cosineIdentity, 'cosine replay identity mismatch');
resources.push(query, near, far, nearScore, farScore, cosineProgram, cosineReplay, cosineBuilder);

// Workload B: drift measurement fully composed in Math Program.
const driftBuilder = new m.WasmMathProgramBuilder(2, 5);
driftBuilder.addUnary(caps.opcodes.normalize, 0, 2);
driftBuilder.addUnary(caps.opcodes.normalize, 1, 3);
driftBuilder.addBinary(caps.opcodes.klDivergence, 2, 3, 4);
driftBuilder.setOutput(4);
const driftProgram = driftBuilder.compile();
const baselineWeights = tensor([4, 3, 2, 1]);
const currentWeights = tensor([3, 3, 2, 2]);
const drift = driftProgram.run2(baselineWeights, currentWeights);
const expectedDrift = 0.4 * Math.log(4 / 3) + 0.1 * Math.log(0.5);
verify(drift, [expectedDrift], 2e-6, 2e-6);
resources.push(baselineWeights, currentWeights, drift, driftProgram, driftBuilder);

// Workload C: read-many branching and fan-in are natural in the write-once DAG.
const branchBuilder = new m.WasmMathProgramV5Builder(3, 6);
branchBuilder.addBinary(caps.opcodes.add, 0, 1, 3);
branchBuilder.addBinary(caps.opcodes.sub, 0, 2, 4);
branchBuilder.addBinary(caps.opcodes.add, 3, 4, 5);
branchBuilder.setOutput(5);
const branchProgram = branchBuilder.compile();
const branchX = tensor([10, 20]);
const branchY = tensor([1, 2]);
const branchZ = tensor([3, 4]);
const branchOut = branchProgram.run3(branchX, branchY, branchZ);
verify(branchOut, [18, 38]);
resources.push(branchX, branchY, branchZ, branchOut, branchProgram, branchBuilder);

// Workload D: affine transform is representable only by supplying full-shape constants as inputs.
// This is valid but intentionally exposes the missing general value-source/constant primitive.
const affineBuilder = new m.WasmMathProgramV5Builder(3, 5);
affineBuilder.addBinary(caps.opcodes.mul, 0, 1, 3);
affineBuilder.addBinary(caps.opcodes.add, 3, 2, 4);
affineBuilder.setOutput(4);
const affineProgram = affineBuilder.compile();
const affineX = tensor([2, 4, 6]);
const scaleTensor = tensor([0.5, 0.5, 0.5]);
const biasTensor = tensor([3, 3, 3]);
const affineOut = affineProgram.run3(affineX, scaleTensor, biasTensor);
verify(affineOut, [4, 5, 6]);
resources.push(affineX, scaleTensor, biasTensor, affineOut, affineProgram, affineBuilder);

// Workload E: simple softmax decomposition is structurally expressible but runtime-invalid
// because the reduced denominator cannot be explicitly broadcast back to feature shape.
const softmaxBuilder = new m.WasmMathProgramBuilder(1, 4);
softmaxBuilder.addUnary(caps.opcodes.exp, 0, 1);
softmaxBuilder.addUnary(caps.opcodes.sum, 1, 2);
softmaxBuilder.addBinary(caps.opcodes.div, 1, 2, 3);
softmaxBuilder.setOutput(3);
const softmaxProgram = softmaxBuilder.compile();
const logits = tensor([1, 2, 3]);
const softmaxIdentity = softmaxProgram.programIdentity();
expectControlledFailure(
  () => softmaxProgram.run1(logits),
  'softmax decomposition unexpectedly succeeded without explicit broadcasting',
);
assert(softmaxProgram.programIdentity() === softmaxIdentity, 'failed softmax run mutated program identity');
resources.push(logits, softmaxProgram, softmaxBuilder);

// Workload F: statistics remain deliberately feature-axis specific; arbitrary-axis reduction
// is not currently represented by the Math Program surface.
const reductionBuilder = new m.WasmMathProgramBuilder(1, 2);
reductionBuilder.addUnary(caps.opcodes.mean, 0, 1);
reductionBuilder.setOutput(1);
const reductionProgram = reductionBuilder.compile();
const matrixLike = tensor([1, 2, 3, 4], [1, 2, 2, 1]);
expectControlledFailure(
  () => reductionProgram.run1(matrixLike),
  'feature statistics unexpectedly accepted a non-[B,F,1,1] reduction workload',
);
resources.push(matrixLike, reductionProgram, reductionBuilder);

// Workload G: explicit-broadcast gap is distinct from implicit broadcasting policy.
const broadcastBuilder = new m.WasmMathProgramBuilder(2, 3);
broadcastBuilder.addBinary(caps.opcodes.add, 0, 1, 2);
broadcastBuilder.setOutput(2);
const broadcastProgram = broadcastBuilder.compile();
const broadcastVector = tensor([1, 2, 3]);
const broadcastScalar = scalar(10);
expectControlledFailure(
  () => broadcastProgram.run2(broadcastVector, broadcastScalar),
  'binary add unexpectedly performed implicit broadcasting',
);
resources.push(broadcastVector, broadcastScalar, broadcastProgram, broadcastBuilder);

// Workload H: the current 64-slot execution profile supports a long deterministic chain.
const pressureBuilder = new m.WasmMathProgramV5Builder(3, 64);
let previousSlot = 0;
for (let out = 3; out < 64; out += 1) {
  pressureBuilder.addUnary(caps.opcodes.abs, previousSlot, out);
  previousSlot = out;
}
pressureBuilder.setOutput(63);
const pressureProgram = pressureBuilder.compile();
const pressureInput = tensor([-1, -2, 3]);
const pressureDummyA = scalar(0);
const pressureDummyB = scalar(0);
const pressureOut = pressureProgram.run3(pressureInput, pressureDummyA, pressureDummyB);
verify(pressureOut, [1, 2, 3]);
assert(pressureProgram.numSteps() === 61, '64-slot pressure program step count mismatch');
expectControlledFailure(
  () => new m.WasmMathProgramV5Builder(3, 65),
  'v5 accepted more than the declared 64-slot execution profile',
);
resources.push(pressureInput, pressureDummyA, pressureDummyB, pressureOut, pressureProgram, pressureBuilder);

// The 8-input bound remains a valid bounded profile and 9 inputs remain fail-closed.
const maxInputBuilder = new m.WasmMathProgramV5Builder(8, 15);
maxInputBuilder.addBinary(caps.opcodes.add, 0, 1, 8);
for (let inputSlot = 2, outputSlot = 9; inputSlot < 8; inputSlot += 1, outputSlot += 1) {
  maxInputBuilder.addBinary(caps.opcodes.add, outputSlot - 1, inputSlot, outputSlot);
}
maxInputBuilder.setOutput(14);
const maxInputProgram = maxInputBuilder.compile();
const eightInputs = Array.from({ length: 8 }, (_, index) => scalar(index + 1));
const maxInputOut = maxInputProgram.run8(...eightInputs);
verify(maxInputOut, [36]);
expectControlledFailure(
  () => new m.WasmMathProgramV5Builder(9, 10),
  'v5 accepted a ninth external input',
);
resources.push(...eightInputs, maxInputOut, maxInputProgram, maxInputBuilder);

const findings = [
  {
    boundary: 'multi-input DAG / fan-out / fan-in (1..8 external inputs)',
    representability: 0,
    classification: 'KEEP',
    evidence: 'read-many inputs, write-once derived slots, 3-input branching and 8-input fan-in execute naturally',
  },
  {
    boundary: '64-slot bounded execution profile',
    representability: 1,
    classification: 'KEEP',
    evidence: '61 derived write-once steps execute within 64 slots; >64 fails closed; no real workload here requires relaxation',
  },
  {
    boundary: 'general constants / value sources',
    representability: 2,
    classification: 'EXTEND',
    evidence: 'affine y=0.5x+3 works only by materializing full-shape scale/bias tensors as external inputs',
  },
  {
    boundary: 'implicit broadcasting',
    representability: 4,
    classification: 'KEEP',
    evidence: 'vector + scalar tensor fails closed; implicit broadcasting should remain forbidden',
  },
  {
    boundary: 'explicit broadcast/expand primitive',
    representability: 4,
    classification: 'EXTEND',
    evidence: 'softmax decomposition cannot divide a feature tensor by its reduced scalar without an explicit expansion operation',
  },
  {
    boundary: 'generic axis-driven reduction',
    representability: 4,
    classification: 'SPLIT',
    evidence: 'Statistics v1 correctly rejects non-[B,F,1,1] input; preserve that semantic profile and add generic reduction separately if required',
  },
  {
    boundary: 'semantic-memory cosine + KL drift workloads',
    representability: 0,
    classification: 'KEEP',
    evidence: 'cosine retrieval and normalize→KL composition execute and replay deterministically using existing primitives',
  },
];

const report = {
  verdict: 'PASS_WITH_FINDINGS',
  task: 'Issue #113 post-v5 Math Program flexibility audit',
  productionSemanticsChanged: false,
  packagedWasmSurface: true,
  schemas: {
    legacy: caps.schema,
    multiInput: v5Caps.schema,
  },
  workloads: {
    semanticMemoryCosine: { passed: true, near: nearValue, far: farValue },
    probabilityDrift: { passed: true, kl: drift.to_array()[0] },
    readManyBranching: true,
    affineWithExternalConstants: true,
    softmaxBlockedByMissingExplicitBroadcast: true,
    arbitraryAxisReductionUnavailable: true,
    implicitBroadcastRejected: true,
    slotPressure61DerivedSteps: true,
    boundedEightInputFanIn: true,
  },
  findings,
  nextRecommendedProductionSlice: 'general constant/value-source primitive; keep implicit broadcasting forbidden',
};

console.log(JSON.stringify(report, null, 2));

for (const value of resources.reverse()) {
  if (value?.free) value.free();
}
