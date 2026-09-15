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

function values(t) {
  return Array.from(t.to_array());
}

function shape(t) {
  return Array.from(t.shape());
}

function assertExact(actual, expected, label) {
  assert(actual.length === expected.length, `${label}: length mismatch`);
  for (let i = 0; i < actual.length; i += 1) {
    assert(Object.is(actual[i], expected[i]), `${label}: value mismatch at ${i}: ${actual[i]} !== ${expected[i]}`);
  }
}

function assertCanonicalPredicate(actual, label) {
  for (let i = 0; i < actual.length; i += 1) {
    assert(
      Object.is(actual[i], 0) || Object.is(actual[i], 1),
      `${label}: non-canonical predicate at ${i}: ${actual[i]}`,
    );
  }
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

const caps = JSON.parse(m.comparisonCapabilities());
assert(caps.schema === 'burn-research.comparison.v1', 'comparison schema mismatch');
assert(caps.rank === 4, 'comparison rank mismatch');
assert(caps.comparison_ops.length === 1 && caps.comparison_ops[0] === 'lessEqual01', 'comparison op contract mismatch');
assert(caps.predicate_representation === 'canonical_f32_0_or_1', 'predicate representation mismatch');
assert(caps.broadcasting === 'forbidden_v1', 'broadcasting contract mismatch');
assert(caps.contracts.finite_inputs === true, 'finite input contract missing');
assert(caps.contracts.shape_preserved === true, 'shape preservation contract missing');
assert(caps.contracts.equality_is_true === true, 'equality contract mismatch');
assert(caps.contracts.boolean_tensor_family === false, 'boolean tensor family must remain absent');
assert(caps.contracts.stateless === true, 'comparison must remain stateless');
assert(caps.contracts.registry_independent === true, 'comparison must remain registry-independent');
assert(caps.contracts.grants_authority === false, 'comparison must not grant authority');

const comparison = new m.WasmComparison();
const lhs = tensor([-2, -1, 0, 1, 2, 3], [1, 6, 1, 1]);
const rhs = tensor([-1, -1, 0, 0, 3, 2], [1, 6, 1, 1]);
const mixed = comparison.lessEqual01(lhs, rhs);
assertExact(shape(mixed), [1, 6, 1, 1], 'mixed shape');
assertExact(values(mixed), [1, 1, 1, 0, 1, 0], 'mixed less/equal/greater values');
assertCanonicalPredicate(values(mixed), 'mixed predicate');

const signedZeroLhs = tensor([-3.5, 0, -0, 7], [1, 4, 1, 1]);
const signedZeroRhs = tensor([-3.5, -0, 0, 7], [1, 4, 1, 1]);
const equality = comparison.lessEqual01(signedZeroLhs, signedZeroRhs);
assertExact(values(equality), [1, 1, 1, 1], 'equality including signed zero');

const repeatA = comparison.lessEqual01(lhs, rhs);
const repeatB = comparison.lessEqual01(lhs, rhs);
assertExact(values(repeatA), values(repeatB), 'deterministic replay');
assertCanonicalPredicate(values(repeatA), 'deterministic predicate');
assert(values(repeatA).every(Number.isFinite), 'comparison produced non-finite predicate values');

const mismatch = tensor([1, 2], [1, 1, 2, 1]);
const finite = tensor([1, 2], [1, 2, 1, 1]);
expectThrow(() => comparison.lessEqual01(finite, mismatch), 'shape mismatch');
const nan = tensor([0, Number.NaN], [1, 2, 1, 1]);
const posInf = tensor([0, Number.POSITIVE_INFINITY], [1, 2, 1, 1]);
const negInf = tensor([0, Number.NEGATIVE_INFINITY], [1, 2, 1, 1]);
expectThrow(() => comparison.lessEqual01(nan, finite), 'NaN lhs');
expectThrow(() => comparison.lessEqual01(posInf, finite), '+Inf lhs');
expectThrow(() => comparison.lessEqual01(finite, nan), 'NaN rhs');
expectThrow(() => comparison.lessEqual01(finite, negInf), '-Inf rhs');

// Direct packaged composition proof: positional source + comparison generates a causal numeric
// predicate without host preprocessing, boolean tensors, or a ternary select/where primitive.
const indexSource = new m.WasmIndexSource();
const causalReference = tensor(Array(16).fill(0), [1, 4, 4, 1]);
const queryIndex = indexSource.indicesLike(causalReference, 1);
const keyIndex = indexSource.indicesLike(causalReference, 2);
const allowed01 = comparison.lessEqual01(keyIndex, queryIndex);
const expectedCausal = [
  1, 0, 0, 0,
  1, 1, 0, 0,
  1, 1, 1, 0,
  1, 1, 1, 1,
];
assertExact(shape(allowed01), [1, 4, 4, 1], 'causal predicate shape');
assertExact(values(allowed01), expectedCausal, 'causal predicate values');
assertCanonicalPredicate(values(allowed01), 'causal predicate');

assert(typeof m.WasmBoolTensor === 'undefined', 'boolean tensor family leaked into packaged surface');
assert(typeof m.WasmMathProgramV9 === 'undefined', 'Math Program v9 must not exist in this slice');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Comparison v1 packaged-WASM proof',
  schema: caps.schema,
  exactShape: true,
  canonicalNumericPredicate: true,
  equalityIsTrue: true,
  signedZeroEquality: true,
  deterministicReplay: true,
  controlledErrors: true,
  causalCompositionWithIndexSource: true,
  booleanTensorFamily: caps.contracts.boolean_tensor_family,
  implicitBroadcasting: false,
  grantsAuthority: caps.contracts.grants_authority,
  mathProgramV9Present: false,
}, null, 2));

for (const t of [
  lhs, rhs, mixed, signedZeroLhs, signedZeroRhs, equality, repeatA, repeatB,
  mismatch, finite, nan, posInf, negInf,
  causalReference, queryIndex, keyIndex, allowed01,
]) t.free();
comparison.free();
indexSource.free();
