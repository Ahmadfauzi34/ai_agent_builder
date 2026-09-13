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
  return report;
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

const capabilities = JSON.parse(m.linearAlgebraCapabilities());
assert(capabilities.schema === 'burn-research.linear-algebra.v1', 'linear algebra schema mismatch');
assert(capabilities.vector_layout === '[B,F,1,1]', 'vector layout mismatch');
assert(capabilities.contracts.solve === 'deferred_v1', 'solve must remain deferred in v1');

const linalg = new m.WasmLinearAlgebra();

const a = tensor([1, 2, 3], [1, 3, 1, 1]);
const b = tensor([3, 1, 2], [1, 3, 1, 1]);

const dot = linalg.dot(a, b);
assert(JSON.stringify(shape(dot)) === JSON.stringify([1, 1, 1, 1]), 'dot output shape mismatch');
verify(values(dot), [11]);

const norm = linalg.l2Norm(a);
assert(JSON.stringify(shape(norm)) === JSON.stringify([1, 1, 1, 1]), 'l2Norm output shape mismatch');
verify(values(norm), [Math.sqrt(14)]);

const cosine = linalg.cosineSimilarity(a, b);
verify(values(cosine), [11 / 14], 2e-6, 2e-6);

const distance = linalg.l2Distance(a, b);
verify(values(distance), [Math.sqrt(6)]);

const zero = tensor([0, 0, 0], [1, 3, 1, 1]);
const zeroCosine = linalg.cosineSimilarity(zero, b);
verify(values(zeroCosine), [0]);

const ma = tensor([1, 2, 3, 4, 5, 6], [1, 1, 2, 3]);
const mb = tensor([7, 8, 9, 10, 11, 12], [1, 1, 3, 2]);
const product = linalg.matmul(ma, mb);
assert(JSON.stringify(shape(product)) === JSON.stringify([1, 1, 2, 2]), 'matmul output shape mismatch');
verify(values(product), [58, 64, 139, 154]);

const badFeatureLayout = tensor([1, 2], [1, 1, 2, 1]);
expectThrow(() => linalg.dot(a, badFeatureLayout), 'feature layout mismatch');
expectThrow(() => linalg.cosineSimilarity(a, b, 0), 'zero epsilon');
expectThrow(() => linalg.cosineSimilarity(a, b, Number.NaN), 'NaN epsilon');

const incompatible = tensor([1, 2, 3, 4], [1, 1, 2, 2]);
const incompatibleRhs = tensor([1, 2, 3], [1, 1, 3, 1]);
expectThrow(() => linalg.matmul(incompatible, incompatibleRhs), 'incompatible matmul');

const nonfinite = tensor([1, Number.POSITIVE_INFINITY, 2], [1, 3, 1, 1]);
expectThrow(() => linalg.l2Norm(nonfinite), 'non-finite norm input');
expectThrow(() => linalg.dot(a, nonfinite), 'non-finite dot input');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Linear Algebra v1 packaged-WASM proof',
  schema: capabilities.schema,
  matrixOps: capabilities.matrix_ops,
  vectorOps: capabilities.vector_ops,
  vectorLayout: capabilities.vector_layout,
  zeroVectorCosine: capabilities.contracts.cosine_zero_vector,
  solve: capabilities.contracts.solve,
  referenceVerification: 'mathVerifyVectors',
  controlledErrors: true,
}, null, 2));

for (const t of [
  a, b, dot, norm, cosine, distance, zero, zeroCosine,
  ma, mb, product, badFeatureLayout, incompatible, incompatibleRhs, nonfinite,
]) {
  t.free();
}
linalg.free();
