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

function verify(actual, expected, absTol = 4e-5, relTol = 4e-5) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(Array.from(actual.to_array())),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
}

function softmaxRows(values, rows, cols) {
  const out = [];
  for (let r = 0; r < rows; r += 1) {
    const row = values.slice(r * cols, (r + 1) * cols);
    const max = Math.max(...row);
    const exps = row.map((value) => Math.exp(value - max));
    const den = exps.reduce((a, b) => a + b, 0);
    out.push(...exps.map((value) => value / den));
  }
  return out;
}

function matmul2d(a, aRows, inner, b, bCols) {
  const out = new Array(aRows * bCols).fill(0);
  for (let i = 0; i < aRows; i += 1) {
    for (let j = 0; j < bCols; j += 1) {
      let sum = 0;
      for (let k = 0; k < inner; k += 1) {
        sum += a[i * inner + k] * b[k * bCols + j];
      }
      out[i * bCols + j] = sum;
    }
  }
  return out;
}

function transpose2d(values, rows, cols) {
  const out = new Array(values.length);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) out[c * rows + r] = values[r * cols + c];
  }
  return out;
}

function expectControlledFailure(fn, message) {
  let rejected = false;
  try {
    const value = fn();
    if (value?.free) value.free();
  } catch {
    rejected = true;
  }
  assert(rejected, message);
}

const caps = JSON.parse(m.mathProgramCapabilities());
const v8Caps = JSON.parse(m.mathProgramV8Capabilities());
assert(v8Caps.schema === 'burn-research.math-program.v8', 'v8 schema mismatch');
assert(v8Caps.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');
assert(v8Caps.generic_reduction_ops.includes('maxAxis') && v8Caps.generic_reduction_ops.includes('meanAxis'), 'v8 reduction surface changed');

const resources = [];
const permuteLastTwo = new Uint32Array([0, 1, 3, 2]);

// Workload A: scaled dot-product attention Q,K,V entirely in one v8 program.
const attentionBuilder = new m.WasmMathProgramV8Builder(3, 15);
attentionBuilder.addPermute(1, 3, permuteLastTwo);
attentionBuilder.addBinary(caps.opcodes.matmul, 0, 3, 4);
attentionBuilder.addFillLike(4, 5, 1 / Math.sqrt(2));
attentionBuilder.addBinary(caps.opcodes.mul, 4, 5, 6);
attentionBuilder.addMaxAxis(6, 7, 3);
attentionBuilder.addExpandLike(7, 6, 8);
attentionBuilder.addBinary(caps.opcodes.sub, 6, 8, 9);
attentionBuilder.addUnary(caps.opcodes.exp, 9, 10);
attentionBuilder.addSumAxis(10, 11, 3);
attentionBuilder.addExpandLike(11, 10, 12);
attentionBuilder.addBinary(caps.opcodes.div, 10, 12, 13);
attentionBuilder.addBinary(caps.opcodes.matmul, 13, 2, 14);
attentionBuilder.setOutput(14);
const attentionProgram = attentionBuilder.compile();

const qValues = [1, 0, 0, 1];
const kValues = [1, 0, 0, 1];
const vValues = [10, 20, 30, 40];
const q = tensor(qValues, [1, 1, 2, 2]);
const k = tensor(kValues, [1, 1, 2, 2]);
const v = tensor(vValues, [1, 1, 2, 2]);
const attentionOut = attentionProgram.run3(q, k, v);
const kT = transpose2d(kValues, 2, 2);
const rawScores = matmul2d(qValues, 2, 2, kT, 2);
const scaledScores = rawScores.map((value) => value / Math.sqrt(2));
const weights = softmaxRows(scaledScores, 2, 2);
const expectedAttention = matmul2d(weights, 2, 2, vValues, 2);
verify(attentionOut, expectedAttention);
const attentionIdentity = attentionProgram.programIdentity();
const attentionReplay = m.WasmMathProgramV8.fromPlan(attentionProgram.programPlan());
assert(attentionReplay.programIdentity() === attentionIdentity, 'attention replay identity mismatch');
verify(attentionReplay.run3(q, k, v), expectedAttention);
resources.push(q, k, v, attentionOut, attentionReplay, attentionProgram, attentionBuilder);

// Workload B: layer-normalization math along the last axis, with explicit epsilon and expansion.
const layerNormBuilder = new m.WasmMathProgramV8Builder(1, 11);
layerNormBuilder.addMeanAxis(0, 1, 3);
layerNormBuilder.addExpandLike(1, 0, 2);
layerNormBuilder.addBinary(caps.opcodes.sub, 0, 2, 3);
layerNormBuilder.addBinary(caps.opcodes.mul, 3, 3, 4);
layerNormBuilder.addMeanAxis(4, 5, 3);
layerNormBuilder.addFillLike(5, 6, 1e-5);
layerNormBuilder.addBinary(caps.opcodes.add, 5, 6, 7);
layerNormBuilder.addUnary(caps.opcodes.sqrt, 7, 8);
layerNormBuilder.addExpandLike(8, 3, 9);
layerNormBuilder.addBinary(caps.opcodes.div, 3, 9, 10);
layerNormBuilder.setOutput(10);
const layerNormProgram = layerNormBuilder.compile();
const layerNormInputValues = [1, 2, 3, 4, 5, 6];
const layerNormInput = tensor(layerNormInputValues, [1, 1, 2, 3]);
const layerNormOut = layerNormProgram.run1(layerNormInput);
const expectedLayerNorm = [];
for (let row = 0; row < 2; row += 1) {
  const group = layerNormInputValues.slice(row * 3, row * 3 + 3);
  const mean = group.reduce((a, b) => a + b, 0) / group.length;
  const variance = group.reduce((acc, value) => acc + (value - mean) ** 2, 0) / group.length;
  const denom = Math.sqrt(variance + 1e-5);
  expectedLayerNorm.push(...group.map((value) => (value - mean) / denom));
}
verify(layerNormOut, expectedLayerNorm);
resources.push(layerNormInput, layerNormOut, layerNormProgram, layerNormBuilder);

// Workload C: externally supplied exact-shape additive mask composes naturally.
const maskedBuilder = new m.WasmMathProgramV8Builder(4, 17);
maskedBuilder.addPermute(1, 4, permuteLastTwo);
maskedBuilder.addBinary(caps.opcodes.matmul, 0, 4, 5);
maskedBuilder.addFillLike(5, 6, 1 / Math.sqrt(2));
maskedBuilder.addBinary(caps.opcodes.mul, 5, 6, 7);
maskedBuilder.addBinary(caps.opcodes.add, 7, 3, 8);
maskedBuilder.addMaxAxis(8, 9, 3);
maskedBuilder.addExpandLike(9, 8, 10);
maskedBuilder.addBinary(caps.opcodes.sub, 8, 10, 11);
maskedBuilder.addUnary(caps.opcodes.exp, 11, 12);
maskedBuilder.addSumAxis(12, 13, 3);
maskedBuilder.addExpandLike(13, 12, 14);
maskedBuilder.addBinary(caps.opcodes.div, 12, 14, 15);
maskedBuilder.addBinary(caps.opcodes.matmul, 15, 2, 16);
maskedBuilder.setOutput(16);
const maskedProgram = maskedBuilder.compile();
const maskValues = [0, -10000, 0, 0];
const mask = tensor(maskValues, [1, 1, 2, 2]);
const maskedOut = maskedProgram.run4(q, k, v, mask);
const maskedScores = scaledScores.map((value, index) => value + maskValues[index]);
const maskedWeights = softmaxRows(maskedScores, 2, 2);
const expectedMasked = matmul2d(maskedWeights, 2, 2, vValues, 2);
verify(maskedOut, expectedMasked);
resources.push(mask, maskedOut, maskedProgram, maskedBuilder);

// Exact-shape arithmetic remains strict: a singleton mask is not implicitly expanded.
const badMask = tensor([0], [1, 1, 1, 1]);
const maskedIdentity = maskedProgram.programIdentity();
expectControlledFailure(
  () => maskedProgram.run4(q, k, v, badMask),
  'masked attention unexpectedly accepted an implicitly broadcast singleton mask',
);
assert(maskedProgram.programIdentity() === maskedIdentity, 'failed mask run changed identity');
resources.push(badMask);

// Workload D: stable log-sum-exp no longer needs a dedicated reduction primitive.
const lseBuilder = new m.WasmMathProgramV8Builder(1, 8);
lseBuilder.addMaxAxis(0, 1, 3);
lseBuilder.addExpandLike(1, 0, 2);
lseBuilder.addBinary(caps.opcodes.sub, 0, 2, 3);
lseBuilder.addUnary(caps.opcodes.exp, 3, 4);
lseBuilder.addSumAxis(4, 5, 3);
lseBuilder.addUnary(caps.opcodes.log, 5, 6);
lseBuilder.addBinary(caps.opcodes.add, 1, 6, 7);
lseBuilder.setOutput(7);
const lseProgram = lseBuilder.compile();
const lseInputValues = [1000, 1001, 1002, 1, 2, 3];
const lseInput = tensor(lseInputValues, [1, 1, 2, 3]);
const lseOut = lseProgram.run1(lseInput);
const expectedLse = [];
for (let row = 0; row < 2; row += 1) {
  const group = lseInputValues.slice(row * 3, row * 3 + 3);
  const max = Math.max(...group);
  expectedLse.push(max + Math.log(group.reduce((acc, value) => acc + Math.exp(value - max), 0)));
}
verify(lseOut, expectedLse);
resources.push(lseInput, lseOut, lseProgram, lseBuilder);

// Workload E: generating a causal/index-derived mask inside the plan is not currently represented.
// This is deliberately measured as a distinct structural gap rather than emulated with hidden host logic.
const probeBuilder = new m.WasmMathProgramV8Builder(1, 2);
const hasComparisonSurface = typeof probeBuilder.addCompare === 'function'
  || typeof probeBuilder.addWhere === 'function'
  || typeof probeBuilder.addLess === 'function'
  || typeof probeBuilder.addLessEqual === 'function';
const hasIndexSource = typeof probeBuilder.addIota === 'function'
  || typeof probeBuilder.addArange === 'function'
  || typeof probeBuilder.addIndicesLike === 'function';
assert(!hasComparisonSurface, 'unexpected comparison/select surface appeared; update audit classification');
assert(!hasIndexSource, 'unexpected index-source surface appeared; update audit classification');
resources.push(probeBuilder);

const findings = [
  {
    boundary: 'scaled dot-product attention with external Q/K/V',
    representability: 0,
    classification: 'KEEP',
    evidence: 'permute + matmul + fillLike scaling + v8 stable softmax + matmul executes in one canonical plan',
  },
  {
    boundary: 'layer-normalization arithmetic',
    representability: 1,
    classification: 'KEEP',
    evidence: 'meanAxis + expandLike + exact-shape arithmetic + fillLike epsilon + sqrt composes naturally, though explicitly',
  },
  {
    boundary: 'stable softmax / log-sum-exp',
    representability: 1,
    classification: 'KEEP',
    evidence: 'generic maxAxis/sumAxis plus explicit expandLike removes the former reduction/broadcast block; dedicated softmax opcode is not required for expressiveness',
  },
  {
    boundary: 'externally supplied exact-shape additive attention mask',
    representability: 0,
    classification: 'KEEP',
    evidence: 'finite additive mask is a normal fourth input; singleton mask still fails closed because implicit broadcasting remains disabled',
  },
  {
    boundary: 'program-generated causal/index-derived mask',
    representability: 4,
    classification: 'SPLIT',
    evidence: 'v8 exposes neither an index/iota value source nor elementwise comparison/select; comparison alone would not create positional indices',
  },
];

console.log(JSON.stringify({
  verdict: 'PASS_WITH_FINDINGS',
  task: 'Issue #148 post-v8 model composition audit',
  productionSemanticsChanged: false,
  packagedWasmSurface: true,
  schema: v8Caps.schema,
  workloads: {
    scaledDotProductAttention: true,
    layerNormArithmetic: true,
    externalAdditiveMask: true,
    stableLogSumExp: true,
    programGeneratedCausalMask: false,
    implicitBroadcastRejected: true,
    replayIdentityStable: true,
  },
  findings,
  nextRecommendedProductionSlice: 'none yet: audit explicit index-source + predicate/select semantics together before adding comparison or masking opcodes',
}, null, 2));

for (const value of resources.reverse()) {
  if (value?.free) value.free();
}
