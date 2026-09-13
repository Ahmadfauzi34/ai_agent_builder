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

function verifyExact(actual, expected) {
  const report = JSON.parse(
    m.mathVerifyVectors(
      new Float32Array(expected),
      new Float32Array(actual),
      0,
      0,
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

const capabilities = JSON.parse(m.tensorTransformCapabilities());
assert(capabilities.schema === 'burn-research.tensor-transform.v1', 'tensor transform schema mismatch');
assert(capabilities.rank === 4, 'tensor transform rank contract mismatch');
assert(capabilities.ops.includes('selectAxis'), 'selectAxis capability missing');

const ops = new m.WasmTensorTransform();

const reshapeInput = tensor([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
const reshaped = ops.reshape(reshapeInput, new Uint32Array([1, 1, 3, 2]));
assert(JSON.stringify(shape(reshaped)) === JSON.stringify([1, 1, 3, 2]), 'reshape shape mismatch');
verifyExact(values(reshaped), [1, 2, 3, 4, 5, 6]);

const transposeInput = tensor([1, 2, 3, 4, 5, 6], [1, 1, 2, 3]);
const transposed = ops.transpose(transposeInput);
assert(JSON.stringify(shape(transposed)) === JSON.stringify([1, 1, 3, 2]), 'transpose shape mismatch');
verifyExact(values(transposed), [1, 4, 2, 5, 3, 6]);

const permuteInput = tensor([1, 2, 3, 4, 5, 6], [1, 2, 1, 3]);
const permuted = ops.permute(permuteInput, new Uint32Array([0, 2, 3, 1]));
assert(JSON.stringify(shape(permuted)) === JSON.stringify([1, 1, 3, 2]), 'permute shape mismatch');
verifyExact(values(permuted), [1, 4, 2, 5, 3, 6]);

const sliceInput = tensor(Array.from({ length: 12 }, (_, index) => index + 1), [1, 2, 2, 3]);
const sliced = ops.slice(
  sliceInput,
  new Uint32Array([0, 0, 0, 1]),
  new Uint32Array([1, 2, 2, 3]),
);
assert(JSON.stringify(shape(sliced)) === JSON.stringify([1, 2, 2, 2]), 'slice shape mismatch');
verifyExact(values(sliced), [2, 3, 5, 6, 8, 9, 11, 12]);

const selected = ops.selectAxis(
  permuteInput,
  1,
  new Uint32Array([1, 0]),
);
assert(JSON.stringify(shape(selected)) === JSON.stringify([1, 2, 1, 3]), 'selectAxis shape mismatch');
verifyExact(values(selected), [4, 5, 6, 1, 2, 3]);

expectThrow(
  () => ops.reshape(reshapeInput, new Uint32Array([1, 1, 2, 2])),
  'reshape element-count mismatch',
);
expectThrow(
  () => ops.reshape(reshapeInput, new Uint32Array([1, 6, 1])),
  'reshape rank mismatch',
);
expectThrow(
  () => ops.permute(permuteInput, new Uint32Array([0, 1, 1, 3])),
  'duplicate permutation axis',
);
expectThrow(
  () => ops.permute(permuteInput, new Uint32Array([0, 1, 2, 4])),
  'out-of-range permutation axis',
);
expectThrow(
  () => ops.slice(sliceInput, new Uint32Array([0, 0, 0, 2]), new Uint32Array([1, 2, 2, 2])),
  'empty slice range',
);
expectThrow(
  () => ops.slice(sliceInput, new Uint32Array([0, 0, 0, 0]), new Uint32Array([1, 2, 2, 4])),
  'out-of-bounds slice range',
);
expectThrow(
  () => ops.selectAxis(permuteInput, 4, new Uint32Array([0])),
  'out-of-range select axis',
);
expectThrow(
  () => ops.selectAxis(permuteInput, 1, new Uint32Array([])),
  'empty select indices',
);
expectThrow(
  () => ops.selectAxis(permuteInput, 1, new Uint32Array([2])),
  'out-of-bounds select index',
);

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Tensor/Layout Transform v1 packaged-WASM proof',
  schema: capabilities.schema,
  rank: capabilities.rank,
  operations: capabilities.ops,
  reshapeExact: true,
  permutationValidated: true,
  sliceBoundsValidated: true,
  selectIndicesValidatedBeforeBackend: true,
  referenceVerification: 'mathVerifyVectors exact',
}, null, 2));

for (const t of [
  reshapeInput, reshaped, transposeInput, transposed, permuteInput, permuted,
  sliceInput, sliced, selected,
]) {
  t.free();
}
ops.free();
