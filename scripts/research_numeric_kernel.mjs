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

function values(t) {
  return Array.from(t.to_array());
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

const capabilities = JSON.parse(m.numericKernelCapabilities());
assert(capabilities.schema === 'burn-research.numeric-kernel.v1', 'numeric kernel schema mismatch');
assert(capabilities.broadcasting === 'forbidden_v1', 'v1 must forbid implicit broadcasting');
assert(capabilities.contracts.finite_outputs === true, 'finite output contract missing');

const kernel = new m.WasmNumericKernel();
const a = tensor([1, -2, 6]);
const b = tensor([2, 4, 3]);

const add = kernel.add(a, b);
const sub = kernel.sub(a, b);
const mul = kernel.mul(a, b);
const div = kernel.div(a, b);
verify(values(add), [3, 2, 9]);
verify(values(sub), [-1, -6, 3]);
verify(values(mul), [2, -8, 18]);
verify(values(div), [0.5, -0.5, 2]);
assert(JSON.stringify(Array.from(add.shape())) === JSON.stringify([1, 3, 1, 1]), 'binary op changed shape');

const absInput = tensor([-2, 0, 3]);
const absOut = kernel.abs(absInput);
verify(values(absOut), [2, 0, 3]);

const sqrtInput = tensor([0, 4, 9]);
const sqrtOut = kernel.sqrt(sqrtInput);
verify(values(sqrtOut), [0, 2, 3]);

const expInput = tensor([0, 1]);
const expOut = kernel.exp(expInput);
verify(values(expOut), [1, Math.E], 2e-6, 2e-6);

const logInput = tensor([1, Math.E]);
const logOut = kernel.log(logInput);
verify(values(logOut), [0, 1], 2e-6, 2e-6);

const clampInput = tensor([-2, 0.5, 3]);
const clampOut = kernel.clamp(clampInput, 0, 1);
verify(values(clampOut), [0, 0.5, 1]);

assert(kernel.allFinite(a) === true, 'finite tensor reported non-finite');
const nonfinite = tensor([1, Number.POSITIVE_INFINITY]);
assert(kernel.allFinite(nonfinite) === false, 'non-finite tensor was not detected');

const mismatch = tensor([1, 2], [1, 1, 2, 1]);
const short = tensor([1, 2]);
expectThrow(() => kernel.add(short, mismatch), 'shape mismatch');

const zeroDivisor = tensor([1, 0]);
expectThrow(() => kernel.div(short, zeroDivisor), 'zero divisor');

const negative = tensor([-1]);
expectThrow(() => kernel.sqrt(negative), 'negative sqrt');

const zero = tensor([0]);
expectThrow(() => kernel.log(zero), 'non-positive log');
expectThrow(() => kernel.clamp(short, 2, 1), 'reversed clamp bounds');
expectThrow(() => kernel.exp(tensor([100])), 'non-finite exp output');
expectThrow(() => kernel.abs(nonfinite), 'non-finite input');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Numeric Kernel v1 packaged-WASM proof',
  schema: capabilities.schema,
  binaryOps: capabilities.binary_ops,
  unaryOps: capabilities.unary_ops,
  broadcasting: capabilities.broadcasting,
  finiteInputs: capabilities.contracts.finite_inputs,
  finiteOutputs: capabilities.contracts.finite_outputs,
  domainErrorsControlled: true,
  referenceVerification: 'mathVerifyVectors',
}, null, 2));

for (const t of [
  add, sub, mul, div, absInput, absOut, sqrtInput, sqrtOut, expInput, expOut,
  logInput, logOut, clampInput, clampOut, a, b, nonfinite, mismatch, short,
  zeroDivisor, negative, zero,
]) {
  t.free();
}
kernel.free();
