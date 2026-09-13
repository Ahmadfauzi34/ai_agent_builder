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

function verify(actual, expected, absTol = 1e-6, relTol = 1e-6) {
  const report = JSON.parse(m.mathVerifyVectors(
    new Float32Array(expected),
    new Float32Array(Array.from(actual.to_array())),
    absTol,
    relTol,
  ));
  assert(report.passed, `verification failed: ${JSON.stringify(report)}`);
}

const caps = JSON.parse(m.mathProgramCapabilities());
assert(caps.schema === 'burn-research.math-program.v1', 'schema mismatch');
assert(caps.registry_dependency === false, 'must remain registry-independent');
assert(caps.mutable_state === false, 'must remain stateless');

const builder = new m.WasmMathProgramBuilder(1, 3);
builder.addUnary(caps.opcodes.sqrt, 0, 1);
builder.addUnary(caps.opcodes.mean, 1, 2);
builder.setOutput(2);
const program = builder.compile();
const input = tensor([1, 4, 9], [1, 3, 1, 1]);
const output = program.run1(input);
verify(output, [2]);

const plan = program.programPlan();
const replay = m.WasmMathProgram.fromPlan(plan);
assert(replay.programIdentity() === program.programIdentity(), 'replay identity mismatch');
const replayOutput = replay.run1(input);
verify(replayOutput, [2]);

const probabilityBuilder = new m.WasmMathProgramBuilder(1, 3);
probabilityBuilder.addUnary(caps.opcodes.normalize, 0, 1);
probabilityBuilder.addUnary(caps.opcodes.entropy, 1, 2);
probabilityBuilder.setOutput(2);
const probabilityProgram = probabilityBuilder.compile();
const weights = tensor([1, 1], [1, 2, 1, 1]);
const entropy = probabilityProgram.run1(weights);
verify(entropy, [Math.log(2)], 2e-6, 2e-6);

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'Math Program v1 packaged-WASM composition proof',
  schema: caps.schema,
  replayIdentity: true,
  registryDependency: caps.registry_dependency,
  mutableState: caps.mutable_state,
  referenceVerification: 'mathVerifyVectors',
}, null, 2));

for (const value of [input, output, replayOutput, weights, entropy]) value.free();
for (const value of [program, replay, probabilityProgram]) value.free();
for (const value of [builder, probabilityBuilder]) value.free();
