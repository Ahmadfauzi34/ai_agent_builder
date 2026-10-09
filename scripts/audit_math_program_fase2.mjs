// Fase 2 structural audit at the wasm boundary.
// Exercises the entry-validation + unchecked-step invariant through the real
// compiled wasm package (pkg/), so the audit pipeline reaches the exact code
// the JS host executes:
//   1. every program version (v1, v4-v9) rejects non-finite inputs at entry
//      with a controlled error;
//   2. multi-step programs produce exact expected values through the
//      unchecked dispatch path (intermediates consumed without re-validation);
//   3. each version's distinctive step variants execute correctly;
//   4. overflow in an intermediate is still caught by output validation.
//
// Usage: node scripts/audit_math_program_fase2.mjs [packageDir]
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function check(condition, message) {
  if (!condition) throw new Error(`fase2 audit failed: ${message}`);
}
function tensor(values, shape) {
  return new m.WasmTensor(new Float32Array(values), new Uint32Array(shape));
}
function values(t) {
  return Array.from(t.to_array());
}
function close(actual, expected, tol, label) {
  check(actual.length === expected.length, `${label}: length ${actual.length} != ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    check(Math.abs(actual[i] - expected[i]) <= tol, `${label}[${i}]: ${actual[i]} != ${expected[i]}`);
  }
}
function expectThrow(fn, label) {
  let threw = false;
  let message = '';
  try {
    const value = fn();
    if (value?.free) value.free();
  } catch (e) {
    threw = true;
    message = String(e?.message ?? e);
  }
  check(threw, `${label} should fail with a controlled error`);
  return message;
}

const baseCaps = JSON.parse(m.mathProgramCapabilities());
const OP = baseCaps.opcodes;
const good = () => tensor([1, 2], [1, 2, 1, 1]);
const badNaN = () => tensor([1, NaN], [1, 2, 1, 1]);
const badInf = () => tensor([1, Infinity], [1, 2, 1, 1]);

// ---- 1. entry validation, all versions ----
{
  const b = new m.WasmMathProgramBuilder(1, 2);
  b.addUnary(OP.abs, 0, 1);
  b.setOutput(1);
  const p = b.compile();
  for (const bad of [badNaN(), badInf()]) {
    const msg = expectThrow(() => p.run1(bad), 'v1 run1 NaN/Inf');
    check(msg.includes('run1 input'), `v1 entry validation, got: ${msg.slice(0, 120)}`);
  }
}
{
  // v4 canonicality requires a selectAxis step.
  const b = new m.WasmMathProgramV4Builder(1, 3);
  b.addUnary(OP.abs, 0, 1);
  b.addSelectAxis(1, 2, 1, new Uint32Array([0, 1]));
  b.setOutput(2);
  const p = b.compile();
  for (const bad of [badNaN(), badInf()]) {
    const msg = expectThrow(() => p.run1(bad), 'v4 run1 NaN/Inf');
    check(msg.includes('run1 input'), `v4 entry validation, got: ${msg.slice(0, 120)}`);
  }
  close(values(p.run1(tensor([-1, 2], [1, 2, 1, 1]))), [1, 2], 1e-6, 'v4 abs+selectAxis');
}
{
  const versions = [
    ['v5', () => new m.WasmMathProgramV5Builder(3, 4), 3],
    ['v6', () => new m.WasmMathProgramV6Builder(1, 2), 1],
    ['v7', () => new m.WasmMathProgramV7Builder(1, 2), 1],
    ['v8', () => new m.WasmMathProgramV8Builder(1, 2), 1],
    ['v9', () => new m.WasmMathProgramV9Builder(1, 2), 1],
  ];
  for (const [name, make, nInputs] of versions) {
    const b = make();
    b.addUnary(OP.abs, 0, nInputs);
    b.setOutput(nInputs);
    const p = b.compile();
    const inputs = [];
    for (let i = 0; i < nInputs; i++) inputs.push(good());
    for (const bad of [badNaN(), badInf()]) {
      const trial = inputs.slice();
      trial[nInputs - 1] = bad;
      const msg = expectThrow(() => p[`run${nInputs}`](...trial), `${name} NaN/Inf`);
      check(msg.includes('runInputs input'), `${name} entry validation, got: ${msg.slice(0, 120)}`);
    }
    p.free?.();
  }
}

// ---- 2. multi-step chaining through the unchecked path (v1) ----
{
  const b = new m.WasmMathProgramBuilder(2, 4);
  b.addBinary(OP.add, 0, 1, 2);
  b.addBinary(OP.mul, 2, 0, 3);
  b.setOutput(3);
  const p = b.compile();
  const out = p.run2(tensor([1, 2], [1, 2, 1, 1]), tensor([3, 4], [1, 2, 1, 1]));
  close(values(out), [4, 12], 1e-6, 'v1 add->mul chain');
  // overflow in an intermediate is still caught by output validation.
  const b2 = new m.WasmMathProgramBuilder(1, 3);
  b2.addUnary(OP.exp, 0, 1);
  b2.addUnary(OP.abs, 1, 2);
  b2.setOutput(2);
  const p2 = b2.compile();
  expectThrow(() => p2.run1(tensor([100], [1, 1, 1, 1])), 'v1 exp(100) overflow');
}

// ---- 3. per-version distinctive steps ----
{
  // v6 fill_like
  const b = new m.WasmMathProgramV6Builder(1, 3);
  b.addUnary(OP.abs, 0, 1);
  b.addFillLike(1, 2, 7.0);
  b.setOutput(2);
  const p = b.compile();
  close(values(p.run1(tensor([-1, -2], [1, 2, 1, 1]))), [7, 7], 1e-6, 'v6 fillLike');
}
{
  // v7 expand_like
  const b = new m.WasmMathProgramV7Builder(2, 3);
  b.addExpandLike(0, 1, 2);
  b.setOutput(2);
  const p = b.compile();
  close(
    values(p.run2(tensor([5], [1, 1, 1, 1]), tensor([0, 0, 0], [1, 3, 1, 1]))),
    [5, 5, 5], 1e-6, 'v7 expandLike',
  );
}
{
  // v8 reductions
  const cases = [
    ['addSumAxis', [10]], ['addMeanAxis', [2.5]], ['addMinAxis', [1]], ['addMaxAxis', [4]],
  ];
  for (const [method, expected] of cases) {
    const b = new m.WasmMathProgramV8Builder(1, 2);
    b[method](0, 1, 1);
    b.setOutput(1);
    const p = b.compile();
    close(values(p.run1(tensor([1, 2, 3, 4], [1, 4, 1, 1]))), expected, 1e-6, `v8 ${method}`);
  }
}
{
  // v9 indices_like + less_equal_01
  const b = new m.WasmMathProgramV9Builder(1, 3);
  b.addIndicesLike(0, 1, 1);
  b.addLessEqual01(0, 1, 2);
  b.setOutput(2);
  const p = b.compile();
  close(values(p.run1(tensor([5, 5, 5], [1, 3, 1, 1]))), [0, 0, 0], 1e-6, 'v9 indices+lessEqual');
}

console.log('fase2 wasm audit: all checks passed');
