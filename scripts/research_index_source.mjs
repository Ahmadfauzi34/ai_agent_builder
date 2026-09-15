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

function repeat(pattern, count) {
  return Array.from({ length: count }, () => pattern).flat();
}

function assertExact(actual, expected, label) {
  assert(actual.length === expected.length, `${label}: length mismatch`);
  for (let i = 0; i < actual.length; i += 1) {
    assert(Object.is(actual[i], expected[i]), `${label}: value mismatch at ${i}: ${actual[i]} !== ${expected[i]}`);
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

const caps = JSON.parse(m.indexSourceCapabilities());
assert(caps.schema === 'burn-research.index-source.v1', 'index source schema mismatch');
assert(caps.rank === 4, 'index source rank mismatch');
assert(caps.axis === 'runtime_0_to_3', 'runtime axis contract mismatch');
assert(caps.max_exact_f32_coordinate === 16777216, 'exact f32 coordinate bound mismatch');
assert(caps.max_supported_axis_length === 16777217, 'axis-length bound mismatch');
assert(caps.contracts.reference_values_ignored === true, 'reference values must be ignored');
assert(caps.contracts.shape_preserved === true, 'shape-preservation contract missing');
assert(caps.contracts.zero_sized_dimensions === false, 'zero-sized dimensions must fail closed');
assert(caps.contracts.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');
assert(caps.contracts.stateless === true, 'index source must remain stateless');
assert(caps.contracts.registry_independent === true, 'index source must remain registry-independent');
assert(caps.contracts.grants_authority === false, 'index source must not grant authority');

const source = new m.WasmIndexSource();
const reference = tensor(Array(24).fill(9), [2, 3, 2, 2]);
const axis0 = source.indicesLike(reference, 0);
const axis1 = source.indicesLike(reference, 1);
const axis2 = source.indicesLike(reference, 2);
const axis3 = source.indicesLike(reference, 3);

assertExact(shape(axis0), [2, 3, 2, 2], 'axis0 shape');
assertExact(shape(axis1), [2, 3, 2, 2], 'axis1 shape');
assertExact(shape(axis2), [2, 3, 2, 2], 'axis2 shape');
assertExact(shape(axis3), [2, 3, 2, 2], 'axis3 shape');
assertExact(values(axis0), [...Array(12).fill(0), ...Array(12).fill(1)], 'axis0 values');
assertExact(values(axis1), [
  ...Array(4).fill(0), ...Array(4).fill(1), ...Array(4).fill(2),
  ...Array(4).fill(0), ...Array(4).fill(1), ...Array(4).fill(2),
], 'axis1 values');
assertExact(values(axis2), repeat([0, 0, 1, 1], 6), 'axis2 values');
assertExact(values(axis3), repeat([0, 1], 12), 'axis3 values');

const refA = tensor([1, 2, 3, 4], [1, 2, 2, 1]);
const refB = tensor([-7, 99, 0.25, -0], [1, 2, 2, 1]);
const fromA = source.indicesLike(refA, 2);
const fromB = source.indicesLike(refB, 2);
assertExact(values(fromA), values(fromB), 'reference-value independence');

const repeatA = source.indicesLike(reference, 1);
const repeatB = source.indicesLike(reference, 1);
assertExact(values(repeatA), values(repeatB), 'deterministic replay');
assert(values(repeatA).every(Number.isFinite), 'indicesLike produced non-finite values');

expectThrow(() => source.indicesLike(reference, 4), 'axis 4');
const empty = tensor([], [1, 0, 1, 1]);
expectThrow(() => source.indicesLike(empty, 1), 'zero-sized shape');

// Keep this proof scoped to the Index Source adapter itself. Other independently-versioned
// packaged surfaces may coexist without changing Index Source v1 semantics.
assert(typeof source.lessEqual01 === 'undefined', 'comparison leaked into index-source adapter');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Index Source v1 packaged-WASM proof',
  schema: caps.schema,
  allAxes: true,
  exactCoordinateGrids: true,
  referenceValuesIgnored: true,
  deterministicReplay: true,
  controlledErrors: true,
  maxExactF32Coordinate: caps.max_exact_f32_coordinate,
  maxSupportedAxisLength: caps.max_supported_axis_length,
  implicitBroadcasting: caps.contracts.implicit_broadcasting,
  grantsAuthority: caps.contracts.grants_authority,
}, null, 2));

for (const t of [
  reference, axis0, axis1, axis2, axis3,
  refA, refB, fromA, fromB, repeatA, repeatB, empty,
]) t.free();
source.free();
