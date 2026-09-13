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

function verify(actual, expected, absTol = 1e-6, relTol = 1e-6) {
  const report = JSON.parse(
    m.mathVerifyVectors(
      new Float32Array(expected),
      new Float32Array(actual),
      absTol,
      relTol,
    ),
  );
  assert(report.passed, `mathVerifyVectors failed: ${JSON.stringify(report)}`);
}

function expectThrow(fn, label) {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  assert(threw, `${label} should fail with a controlled error`);
}

const capabilities = JSON.parse(m.statisticsCapabilities());
assert(capabilities.schema === 'burn-research.statistics.v1', 'statistics schema mismatch');
assert(capabilities.input_layout === '[B,F,1,1]', 'statistics input layout mismatch');
assert(capabilities.output_layout === '[B,1,1,1]', 'statistics output layout mismatch');
assert(capabilities.reduction_axis === 1, 'statistics reduction axis mismatch');
assert(capabilities.contracts.variance_semantics === 'population_no_bessel_correction', 'variance semantics mismatch');

const stats = new m.WasmStatistics();
const input = tensor([2, 4, 6, 8, 1, 1, 3, 3], [2, 4, 1, 1]);

const sum = stats.sum(input);
const mean = stats.mean(input);
const variance = stats.variancePopulation(input);
const std = stats.stdPopulation(input);
const min = stats.min(input);
const max = stats.max(input);

for (const out of [sum, mean, variance, std, min, max]) {
  assert(JSON.stringify(shape(out)) === JSON.stringify([2, 1, 1, 1]), 'statistics output shape mismatch');
}
verify(values(sum), [20, 8]);
verify(values(mean), [5, 2]);
verify(values(variance), [5, 1]);
verify(values(std), [Math.sqrt(5), 1], 2e-6, 2e-6);
verify(values(min), [2, 1]);
verify(values(max), [8, 3]);

const singleton = tensor([3, -2], [2, 1, 1, 1]);
const singletonVariance = stats.variancePopulation(singleton);
const singletonStd = stats.stdPopulation(singleton);
verify(values(singletonVariance), [0, 0]);
verify(values(singletonStd), [0, 0]);

const badLayout = tensor([1, 2, 3, 4], [1, 2, 2, 1]);
expectThrow(() => stats.mean(badLayout), 'invalid feature layout');
const nonfinite = tensor([1, Number.POSITIVE_INFINITY], [1, 2, 1, 1]);
expectThrow(() => stats.sum(nonfinite), 'non-finite sum input');
expectThrow(() => stats.variancePopulation(nonfinite), 'non-finite variance input');
expectThrow(() => stats.max(nonfinite), 'non-finite max input');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Statistics v1 packaged-WASM proof',
  schema: capabilities.schema,
  reducers: capabilities.reducers,
  inputLayout: capabilities.input_layout,
  outputLayout: capabilities.output_layout,
  reductionAxis: capabilities.reduction_axis,
  varianceSemantics: capabilities.contracts.variance_semantics,
  referenceVerification: 'mathVerifyVectors',
  controlledErrors: true,
}, null, 2));

for (const t of [input, sum, mean, variance, std, min, max, singleton, singletonVariance, singletonStd, badLayout, nonfinite]) {
  t.free();
}
stats.free();
