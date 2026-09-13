import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function dot(a, b) {
  return a.reduce((sum, value, index) => sum + value * b[index], 0);
}

function closeArray(a, b, tolerance = 1e-6) {
  return a.length === b.length && a.every((value, index) => Math.abs(value - b[index]) <= tolerance);
}

function build(layerId, epsilon) {
  const registry = new m.LayerRegistry();
  const spec = m.AgentLayerSpec.featureNorm(layerId, epsilon);
  registry.initAgentLayer(spec);
  const builder = new m.AgentGraphBuilder(2);
  builder.addUnary(spec, 0, 1);
  builder.setOutput(1);
  const graph = builder.compile(registry);
  return { registry, spec, builder, graph };
}

function run(graph, registry, values) {
  const input = new m.WasmTensor(
    new Float32Array(values),
    new Uint32Array([1, values.length, 1, 1]),
  );
  const output = graph.run(registry, input);
  const result = Array.from(output.to_array());
  output.free();
  input.free();
  return result;
}

const base = build(17, undefined);
assert(base.registry.totalParams() === 0, 'FeatureNorm registry entry must be parameter-free');
assert(base.registry.layerExists(base.spec.layerType(), base.spec.layerId()), 'FeatureNorm missing from Registry');
assert(base.registry.getLayerState(base.spec.layerId(), base.spec.layerType()).length === 0, 'FeatureNorm state must be empty');

const query = [1.0, 0.0];
const relevant = [1.0, 0.1];
const distractor = [2.0, 2.0];
const rawRelevant = dot(query, relevant);
const rawDistractor = dot(query, distractor);
assert(rawDistractor > rawRelevant, 'fixture must expose raw magnitude bias');

const normalizedQuery = run(base.graph, base.registry, query);
const normalizedRelevant = run(base.graph, base.registry, relevant);
const normalizedDistractor = run(base.graph, base.registry, distractor);
const relevantScore = dot(normalizedQuery, normalizedRelevant);
const distractorScore = dot(normalizedQuery, normalizedDistractor);
assert(relevantScore > distractorScore, 'typed FeatureNorm graph did not repair ranking');
assert(closeArray(run(base.graph, base.registry, [0, 0]), [0, 0]), 'zero vector is not stable');

const identity = base.graph.programIdentity();
const bundle = m.exportProgramBundle(base.graph, base.registry, true);
const replayRegistry = new m.LayerRegistry();
const replayGraph = m.importProgramBundle(replayRegistry, bundle);
assert(replayGraph.programIdentity() === identity, 'Program Bundle changed structural identity');
const replayRelevant = run(replayGraph, replayRegistry, relevant);
assert(closeArray(replayRelevant, normalizedRelevant), 'Program Bundle replay output mismatch');
replayGraph.validateRegistryBinding(replayRegistry);

const changed = build(17, 1e-6);
assert(changed.graph.programIdentity() !== identity, 'epsilon did not participate in program identity');
const changedPlan = base.graph.programPlan();
const replacement = m.AgentLayerSpec.featureNorm(17, 1e-6);
base.registry.initAgentLayer(replacement);
replacement.free();
let driftRejected = false;
try {
  base.graph.validateRegistryBinding(base.registry);
} catch {
  driftRejected = true;
}
assert(driftRejected, 'old CompiledGraph accepted FeatureNorm structural drift');
const rebound = base.registry.compileGraph(changedPlan);
assert(rebound.programIdentity() !== identity, 'recompile did not capture changed epsilon identity');

let invalidEpsilonRejected = false;
try {
  m.AgentLayerSpec.featureNorm(99, 0);
} catch {
  invalidEpsilonRejected = true;
}
assert(invalidEpsilonRejected, 'typed FeatureNorm accepted invalid epsilon');

const capabilities = JSON.parse(m.agentCapabilities());
assert(capabilities.agent_facade.constructors.includes('featureNorm'), 'featureNorm missing from agent capabilities');
assert(capabilities.layers.feature_norm.code === base.spec.layerType(), 'featureNorm capability code mismatch');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'first-class FeatureNorm graph + identity + Program Bundle retrieval proof',
  layerType: base.spec.layerType(),
  parameterCount: base.registry.totalParams(),
  normalizedWinner: relevantScore > distractorScore ? 'relevant' : 'distractor',
  programIdentity: identity,
  programBundleReplay: true,
  identityBindsEpsilon: true,
  structuralDriftRejected: driftRejected,
  invalidEpsilonRejected,
}, null, 2));

rebound.free();
changed.graph.free(); changed.builder.free(); changed.spec.free(); changed.registry.free();
replayGraph.free(); replayRegistry.free();
base.graph.free(); base.builder.free(); base.spec.free(); base.registry.free();
