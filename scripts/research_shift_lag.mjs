import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function mse(a, b) {
  assert(a.length === b.length, 'MSE length mismatch');
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return sum / a.length;
}

function closeEnough(a, b, tol = 1e-6) {
  return a.length === b.length && a.every((value, i) => Math.abs(value - b[i]) <= tol);
}

// Real application-shaped proof: a telemetry stream arrived one sample late.
// ShiftUp(1) removes the lag while preserving shape and zero-filling the tail.
const observed = new Float32Array([0.0, 0.2, 0.8, 0.4, 0.1]);
const expectedAligned = new Float32Array([0.2, 0.8, 0.4, 0.1, 0.0]);

const registry = new m.LayerRegistry();
const spec = m.AgentLayerSpec.shiftUp(41, 1);
registry.initAgentLayer(spec);

const builder = new m.AgentGraphBuilder(2);
builder.addUnary(spec, 0, 1);
builder.setOutput(1);
const graph = builder.compile(registry);

const input = new m.WasmTensor(observed, new Uint32Array([1, 1, 5, 1]));
const output = graph.run(registry, input);
const corrected = Array.from(output.to_array());
const target = Array.from(expectedAligned);
const before = mse(Array.from(observed), target);
const after = mse(corrected, target);

assert(before > 0, `expected non-zero pre-correction MSE, got ${before}`);
assert(after <= 1e-12, `lag correction MSE must be zero within tolerance, got ${after}`);
assert(closeEnough(corrected, target), `unexpected corrected telemetry: ${corrected}`);
assert(registry.totalParams() === 0, 'Shift must remain parameter-free');

const identity = graph.programIdentity();
const bundle = m.exportProgramBundle(graph, registry, true);
const replayRegistry = new m.LayerRegistry();
const replayGraph = m.importProgramBundle(replayRegistry, bundle);
const replayOutput = replayGraph.run(replayRegistry, input);
const replayed = Array.from(replayOutput.to_array());

assert(replayGraph.programIdentity() === identity, 'Program Bundle changed structural identity');
assert(closeEnough(replayed, corrected), `Program Bundle replay mismatch: ${replayed}`);
assert(replayRegistry.totalParams() === 0, 'replayed Shift must remain parameter-free');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'one-sample telemetry lag correction',
  observed: Array.from(observed),
  expectedAligned: target,
  corrected,
  mseBefore: before,
  mseAfter: after,
  parameterCount: registry.totalParams(),
  programIdentity: identity,
  bundleBytes: bundle.length,
  replayExact: true,
}, null, 2));

replayOutput.free();
replayGraph.free();
replayRegistry.free();
output.free();
input.free();
graph.free();
builder.free();
spec.free();
registry.free();
