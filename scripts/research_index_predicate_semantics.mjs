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

function elementCount(shape) {
  return shape.reduce((acc, value) => acc * value, 1);
}

function indicesLikeReference(shape, axis) {
  assert(shape.length === 4, 'reference index source expects rank 4');
  assert(Number.isInteger(axis) && axis >= 0 && axis < 4, 'reference axis must be 0..3');
  assert(shape.every((value) => Number.isInteger(value) && value > 0), 'reference dimensions must be positive');

  const total = elementCount(shape);
  const out = new Array(total);
  const [d0, d1, d2, d3] = shape;
  let offset = 0;
  for (let i0 = 0; i0 < d0; i0 += 1) {
    for (let i1 = 0; i1 < d1; i1 += 1) {
      for (let i2 = 0; i2 < d2; i2 += 1) {
        for (let i3 = 0; i3 < d3; i3 += 1) {
          const coords = [i0, i1, i2, i3];
          out[offset] = coords[axis];
          offset += 1;
        }
      }
    }
  }
  return out;
}

function lessEqual01Reference(lhs, rhs) {
  assert(lhs.length === rhs.length, 'reference comparison length mismatch');
  return lhs.map((value, index) => (value <= rhs[index] ? 1 : 0));
}

function softmaxRows(values, rows, cols) {
  const out = [];
  for (let row = 0; row < rows; row += 1) {
    const group = values.slice(row * cols, (row + 1) * cols);
    const max = Math.max(...group);
    const exp = group.map((value) => Math.exp(value - max));
    const den = exp.reduce((a, b) => a + b, 0);
    out.push(...exp.map((value) => value / den));
  }
  return out;
}

const caps = JSON.parse(m.mathProgramCapabilities());
const v8Caps = JSON.parse(m.mathProgramV8Capabilities());
assert(v8Caps.schema === 'burn-research.math-program.v8', 'v8 schema mismatch');
assert(v8Caps.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');

// The current packaged surface intentionally has no positional source, comparison, or select op.
const probe = new m.WasmMathProgramV8Builder(1, 2);
const hasIndexSource = typeof probe.addIndicesLike === 'function'
  || typeof probe.addIota === 'function'
  || typeof probe.addArange === 'function';
const hasComparison = typeof probe.addLessEqual === 'function'
  || typeof probe.addCompare === 'function'
  || typeof probe.addLess === 'function';
const hasSelect = typeof probe.addSelect === 'function'
  || typeof probe.addWhere === 'function'
  || typeof probe.addSelectWhere === 'function';
assert(!hasIndexSource, 'index source unexpectedly already exists');
assert(!hasComparison, 'comparison unexpectedly already exists');
assert(!hasSelect, 'select/where unexpectedly already exists');

const resources = [probe];

// Reference positional semantics for an attention score tensor [B,G,Q,K].
const scoreShape = [1, 1, 3, 3];
const scoresValues = [
  3, 1, 2,
  2, 4, 1,
  1, 0, 5,
];
const qIndices = indicesLikeReference(scoreShape, 2);
const kIndices = indicesLikeReference(scoreShape, 3);
const allowed01 = lessEqual01Reference(kIndices, qIndices);
assert(JSON.stringify(qIndices) === JSON.stringify([0,0,0,1,1,1,2,2,2]), 'query indices reference mismatch');
assert(JSON.stringify(kIndices) === JSON.stringify([0,1,2,0,1,2,0,1,2]), 'key indices reference mismatch');
assert(JSON.stringify(allowed01) === JSON.stringify([1,0,0,1,1,0,1,1,1]), 'causal predicate reference mismatch');

// Important finding: once an exact 0/1 numeric predicate exists, additive masking does not need
// a ternary select/where primitive. Existing exact-shape arithmetic is sufficient:
// blocked = (1 - allowed01) * (-C)
// masked = scores + blocked
// followed by the already-proven v8 stable softmax.
const causalBuilder = new m.WasmMathProgramV8Builder(2, 14);
causalBuilder.addFillLike(1, 2, 1.0);
causalBuilder.addBinary(caps.opcodes.sub, 2, 1, 3);
causalBuilder.addFillLike(0, 4, -10000.0);
causalBuilder.addBinary(caps.opcodes.mul, 3, 4, 5);
causalBuilder.addBinary(caps.opcodes.add, 0, 5, 6);
causalBuilder.addMaxAxis(6, 7, 3);
causalBuilder.addExpandLike(7, 6, 8);
causalBuilder.addBinary(caps.opcodes.sub, 6, 8, 9);
causalBuilder.addUnary(caps.opcodes.exp, 9, 10);
causalBuilder.addSumAxis(10, 11, 3);
causalBuilder.addExpandLike(11, 10, 12);
causalBuilder.addBinary(caps.opcodes.div, 10, 12, 13);
causalBuilder.setOutput(13);
const causalProgram = causalBuilder.compile();

const scores = tensor(scoresValues, scoreShape);
const predicate = tensor(allowed01, scoreShape);
const causalOut = causalProgram.run2(scores, predicate);
const expectedMaskedScores = scoresValues.map((value, index) => value + (1 - allowed01[index]) * -10000);
const expectedCausalSoftmax = softmaxRows(expectedMaskedScores, 3, 3);
verify(causalOut, expectedCausalSoftmax);

const causalIdentity = causalProgram.programIdentity();
const causalReplay = m.WasmMathProgramV8.fromPlan(causalProgram.programPlan());
assert(causalReplay.programIdentity() === causalIdentity, 'causal arithmetic replay identity mismatch');
const replayOut = causalReplay.run2(scores, predicate);
verify(replayOut, expectedCausalSoftmax);

// Exact-shape policy remains strong: a singleton predicate cannot silently broadcast.
const badPredicate = tensor([1], [1, 1, 1, 1]);
expectControlledFailure(
  () => causalProgram.run2(scores, badPredicate),
  'causal arithmetic unexpectedly accepted an implicitly broadcast singleton predicate',
);
assert(causalProgram.programIdentity() === causalIdentity, 'failed predicate-shape run changed identity');

resources.push(scores, predicate, causalOut, causalReplay, replayOut, badPredicate, causalProgram, causalBuilder);

// Generality check beyond attention: the same numeric 0/1 predicate can gate arbitrary exact-shape
// values through ordinary multiplication. This is not a causal-mask-specific semantic.
const gateBuilder = new m.WasmMathProgramV8Builder(2, 3);
gateBuilder.addBinary(caps.opcodes.mul, 0, 1, 2);
gateBuilder.setOutput(2);
const gateProgram = gateBuilder.compile();
const gateValues = tensor([1,2,3,4,5,6,7,8,9], scoreShape);
const gatePredicate = tensor(allowed01, scoreShape);
const gated = gateProgram.run2(gateValues, gatePredicate);
verify(gated, [1,0,0,4,5,0,7,8,9]);
resources.push(gateValues, gatePredicate, gated, gateProgram, gateBuilder);

// Candidate comparison. The audit deliberately avoids adding production semantics; it only proves
// which missing semantics are necessary for the target workload and which are not.
const findings = [
  {
    boundary: 'shape-bound positional source: indicesLike(reference, axis)',
    representability: 4,
    classification: 'EXTEND',
    evidence: 'causal coordinates require query/key positions; current v8 has no positional source and host preprocessing is otherwise required',
  },
  {
    boundary: 'elementwise lessEqual01 exact-shape comparison',
    representability: 4,
    classification: 'EXTEND',
    evidence: 'reference k<=q produces an exact reusable 0/1 tensor; current v8 has no comparison surface',
  },
  {
    boundary: 'additive mask composition from an exact 0/1 numeric predicate',
    representability: 0,
    classification: 'KEEP',
    evidence: 'existing fillLike/sub/mul/add plus v8 stable softmax execute the causal mask once the predicate is supplied',
  },
  {
    boundary: 'generic select/where for the proven causal-mask workload',
    representability: 0,
    classification: 'KEEP',
    evidence: 'not required: (1-predicate)*blockedValue composes through existing exact-shape arithmetic, avoiding a new ternary execution semantic',
  },
  {
    boundary: 'distinct boolean/predicate tensor family',
    representability: 3,
    classification: 'SPLIT',
    evidence: 'would widen the current f32 WasmTensor/value model; the proven target workload does not require that bridge expansion if comparison emits canonical 0/1 f32',
  },
  {
    boundary: 'causalMask-specific opcode',
    representability: 2,
    classification: 'KEEP',
    evidence: 'domain-specific shortcut is unnecessary if generic positional source + comparison are provided; the same 0/1 predicate also supports non-attention gating',
  },
];

console.log(JSON.stringify({
  verdict: 'PASS_WITH_FINDINGS',
  task: 'Issue #150 index-source + predicate/select semantics audit',
  productionSemanticsChanged: false,
  packagedWasmSurface: true,
  schema: v8Caps.schema,
  workloads: {
    referenceIndicesLike: true,
    referenceLessEqual01: true,
    causalMaskArithmeticWithoutSelect: true,
    stableCausalSoftmaxWithInjectedPredicate: true,
    nonAttentionNumericPredicateGating: true,
    implicitBroadcastRejected: true,
    replayIdentityStable: true,
    internalIndexSourceAvailable: false,
    internalComparisonAvailable: false,
    selectRequiredForTargetWorkload: false,
  },
  findings,
  nextRecommendedProductionSequence: [
    'prove generic indicesLike rank-4 value-source core independently',
    'prove exact-shape lessEqual01 comparison core independently',
    'only after both primitives are green, integrate them through a new Math Program plan version that delegates to the proven cores',
    'defer generic select/where and a distinct boolean tensor family until a real workload requires them',
  ],
}, null, 2));

for (const value of resources.reverse()) {
  if (value?.free) value.free();
}
