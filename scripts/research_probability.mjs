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

const capabilities = JSON.parse(m.probabilityCapabilities());
assert(capabilities.schema === 'burn-research.probability.v1', 'probability schema mismatch');
assert(capabilities.input_layout === '[B,F,1,1]', 'probability input layout mismatch');
assert(capabilities.scalar_output_layout === '[B,1,1,1]', 'probability output layout mismatch');
assert(capabilities.log_base === 'e', 'probability log base mismatch');
assert(capabilities.contracts.rng_sampling === 'deferred', 'RNG must remain deferred in v1');

const probability = new m.WasmProbability();

const weights = tensor([1, 1, 2, 0, 3, 1], [2, 3, 1, 1]);
const normalized = probability.normalize(weights);
assert(JSON.stringify(shape(normalized)) === JSON.stringify([2, 3, 1, 1]), 'normalize shape mismatch');
verify(values(normalized), [0.25, 0.25, 0.5, 0, 0.75, 0.25]);

const p = tensor([0.5, 0.5], [1, 2, 1, 1]);
const q = tensor([0.75, 0.25], [1, 2, 1, 1]);
const entropy = probability.entropy(p);
const crossEntropy = probability.crossEntropy(p, q);
const kl = probability.klDivergence(p, q);

for (const out of [entropy, crossEntropy, kl]) {
  assert(JSON.stringify(shape(out)) === JSON.stringify([1, 1, 1, 1]), 'probability scalar output shape mismatch');
}

const expectedEntropy = Math.log(2);
const expectedCrossEntropy = -0.5 * Math.log(0.75) - 0.5 * Math.log(0.25);
const expectedKl = expectedCrossEntropy - expectedEntropy;
verify(values(entropy), [expectedEntropy], 2e-6, 2e-6);
verify(values(crossEntropy), [expectedCrossEntropy], 2e-6, 2e-6);
verify(values(kl), [expectedKl], 2e-6, 2e-6);

const zeroP = tensor([0, 1], [1, 2, 1, 1]);
const zeroQ = tensor([0, 1], [1, 2, 1, 1]);
const zeroEntropy = probability.entropy(zeroP);
const zeroCrossEntropy = probability.crossEntropy(zeroP, zeroQ);
const zeroKl = probability.klDivergence(zeroP, zeroQ);
verify(values(zeroEntropy), [0]);
verify(values(zeroCrossEntropy), [0]);
verify(values(zeroKl), [0]);

const negative = tensor([-1, 2], [1, 2, 1, 1]);
expectThrow(() => probability.normalize(negative), 'negative weight');
const zeroMass = tensor([0, 0], [1, 2, 1, 1]);
expectThrow(() => probability.normalize(zeroMass), 'zero mass');
const unnormalized = tensor([0.2, 0.2], [1, 2, 1, 1]);
expectThrow(() => probability.entropy(unnormalized), 'unnormalized entropy input');
const supportP = tensor([1, 0], [1, 2, 1, 1]);
const missingSupportQ = tensor([0, 1], [1, 2, 1, 1]);
expectThrow(() => probability.crossEntropy(supportP, missingSupportQ), 'cross-entropy support mismatch');
expectThrow(() => probability.klDivergence(supportP, missingSupportQ), 'KL support mismatch');
const nonfinite = tensor([Number.NaN, 1], [1, 2, 1, 1]);
expectThrow(() => probability.normalize(nonfinite), 'non-finite weight');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Probability v1 packaged-WASM proof',
  schema: capabilities.schema,
  ops: capabilities.ops,
  logBase: capabilities.log_base,
  normalizationTolerance: capabilities.contracts.normalization_tolerance,
  zeroPContribution: capabilities.contracts.zero_p_contribution,
  qZeroWherePPositive: capabilities.contracts.q_zero_where_p_positive,
  rngSampling: capabilities.contracts.rng_sampling,
  referenceVerification: 'mathVerifyVectors',
  controlledErrors: true,
}, null, 2));

for (const t of [
  weights, normalized, p, q, entropy, crossEntropy, kl,
  zeroP, zeroQ, zeroEntropy, zeroCrossEntropy, zeroKl,
  negative, zeroMass, unnormalized, supportP, missingSupportQ, nonfinite,
]) {
  t.free();
}
probability.free();
