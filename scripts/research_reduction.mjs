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

function verify(actual, expected, absTol = 2e-6, relTol = 2e-6) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(actual),
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

const caps = JSON.parse(m.reductionCapabilities());
assert(caps.schema === 'burn-research.reduction.v1', 'reduction schema mismatch');
assert(caps.rank === 4, 'reduction rank mismatch');
assert(caps.axis === 'runtime_0_to_3', 'runtime axis contract mismatch');
assert(caps.keepdim === true, 'keepdim must remain true');
assert(caps.contracts.statistics_v1_independent === true, 'Statistics v1 independence missing');
assert(caps.contracts.implicit_broadcasting === false, 'implicit broadcasting must remain disabled');

const reduction = new m.WasmReduction();
const input = tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2, 1]);

const sum0 = reduction.sumAxis(input, 0);
const sum1 = reduction.sumAxis(input, 1);
const sum2 = reduction.sumAxis(input, 2);
const sum3 = reduction.sumAxis(input, 3);
assert(JSON.stringify(shape(sum0)) === JSON.stringify([1, 2, 2, 1]), 'axis 0 keepdim mismatch');
assert(JSON.stringify(shape(sum1)) === JSON.stringify([2, 1, 2, 1]), 'axis 1 keepdim mismatch');
assert(JSON.stringify(shape(sum2)) === JSON.stringify([2, 2, 1, 1]), 'axis 2 keepdim mismatch');
assert(JSON.stringify(shape(sum3)) === JSON.stringify([2, 2, 2, 1]), 'axis 3 keepdim mismatch');
verify(values(sum0), [6, 8, 10, 12]);
verify(values(sum1), [4, 6, 12, 14]);
verify(values(sum2), [3, 7, 11, 15]);
verify(values(sum3), values(input));

const mean0 = reduction.meanAxis(input, 0);
const min1 = reduction.minAxis(input, 1);
const max2 = reduction.maxAxis(input, 2);
verify(values(mean0), [3, 4, 5, 6]);
verify(values(min1), [1, 2, 5, 6]);
verify(values(max2), [2, 4, 6, 8]);

expectThrow(() => reduction.sumAxis(input, 4), 'axis 4');
const empty = tensor([], [1, 0, 1, 1]);
expectThrow(() => reduction.sumAxis(empty, 1), 'zero-sized reduction');
const nonfinite = tensor([1, Number.POSITIVE_INFINITY], [1, 2, 1, 1]);
expectThrow(() => reduction.maxAxis(nonfinite, 1), 'non-finite reduction input');

// Stable softmax on axis 2 using the new generic reducer plus the already-proven explicit expandLike.
const stableInput = tensor([1000, 1001, 1002, 1, 2, 3], [1, 2, 3, 1]);
const max = reduction.maxAxis(stableInput, 2);
assert(JSON.stringify(shape(max)) === JSON.stringify([1, 2, 1, 1]), 'stable-softmax max shape mismatch');
verify(values(max), [1002, 3]);

const expandBuilder = new m.WasmMathProgramV7Builder(2, 3);
expandBuilder.addExpandLike(0, 1, 2);
expandBuilder.setOutput(2);
const expandProgram = expandBuilder.compile();

const kernel = new m.WasmNumericKernel();
const maxFull = expandProgram.run2(max, stableInput);
const shifted = kernel.sub(stableInput, maxFull);
const exp = kernel.exp(shifted);
const den = reduction.sumAxis(exp, 2);
const denFull = expandProgram.run2(den, exp);
const softmax = kernel.div(exp, denFull);
const softmaxValues = values(softmax);
assert(softmaxValues.every(Number.isFinite), 'stable softmax produced non-finite output');
assert(Math.abs(softmaxValues.slice(0, 3).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'softmax group 0 mass mismatch');
assert(Math.abs(softmaxValues.slice(3, 6).reduce((a, b) => a + b, 0) - 1) <= 3e-6, 'softmax group 1 mass mismatch');
verify(softmaxValues.slice(0, 3), softmaxValues.slice(3, 6));

const statsCaps = JSON.parse(m.statisticsCapabilities());
assert(statsCaps.schema === 'burn-research.statistics.v1', 'Statistics v1 schema changed');
assert(statsCaps.reduction_axis === 1, 'Statistics v1 reduction axis changed');
assert(statsCaps.input_layout === '[B,F,1,1]', 'Statistics v1 input layout changed');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Generic Reduction v1 packaged-WASM proof',
  schema: caps.schema,
  allAxes: true,
  keepdimRank4: true,
  controlledErrors: true,
  stableSoftmaxAxis2: true,
  statisticsV1Unchanged: true,
  implicitBroadcasting: caps.contracts.implicit_broadcasting,
}, null, 2));

for (const t of [
  input, sum0, sum1, sum2, sum3, mean0, min1, max2, empty, nonfinite,
  stableInput, max, maxFull, shifted, exp, den, denFull, softmax,
]) t.free();
expandProgram.free();
expandBuilder.free();
kernel.free();
reduction.free();
