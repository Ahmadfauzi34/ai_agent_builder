import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

const LAYER_LINEAR = 0x01;
const LAYER_ACTIVATION = 0x04;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function exactArrayEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
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

function tensor(values, shape) {
  return new m.WasmTensor(new Float32Array(values), new Uint32Array(shape));
}

function runVector(graph, registry, values) {
  const input = tensor(values, [1, values.length, 1, 1]);
  const output = graph.run(registry, input);
  const result = Array.from(output.to_array());
  output.free();
  input.free();
  return result;
}

function parsePlanLayerRefs(planLike) {
  const plan = Array.from(planLike);
  assert(plan.length >= 9, 'compiled graph plan is unexpectedly short');
  const view = new DataView(Uint8Array.from(plan).buffer);
  const numSteps = view.getUint32(0, true);
  const expectedLength = 8 + numSteps * 9 + 1;
  assert(plan.length === expectedLength, `compiled graph plan length mismatch: ${plan.length} !== ${expectedLength}`);
  const refs = [];
  for (let i = 0; i < numSteps; i += 1) {
    const offset = 8 + i * 9;
    refs.push({
      arity: plan[offset],
      layerType: plan[offset + 1],
      layerId: view.getUint32(offset + 2, true),
    });
  }
  return refs;
}

function uniqueRefs(refs) {
  const seen = new Set();
  const out = [];
  for (const ref of refs) {
    const key = `${ref.layerType}:${ref.layerId}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(ref);
    }
  }
  return out;
}

function filled(length, value) {
  const out = new Float32Array(length);
  out.fill(value);
  return out;
}

function buildSingleLinear() {
  const registry = new m.LayerRegistry();
  const linear = m.AgentLayerSpec.linear(101, 2, 1, true);
  registry.initAgentLayer(linear);
  const builder = new m.AgentGraphBuilder(2);
  builder.addUnary(linear, 0, 1);
  builder.setOutput(1);
  const graph = builder.compile(registry);
  return { registry, linear, builder, graph };
}

function buildMultiLayer() {
  const registry = new m.LayerRegistry();
  const linear1 = m.AgentLayerSpec.linear(201, 2, 2, true);
  const relu = m.AgentLayerSpec.relu(202);
  const linear2 = m.AgentLayerSpec.linear(203, 2, 1, true);
  registry.initAgentLayer(linear1);
  registry.initAgentLayer(relu);
  registry.initAgentLayer(linear2);
  const builder = new m.AgentGraphBuilder(4);
  builder.addUnary(linear1, 0, 1);
  builder.addUnary(relu, 1, 2);
  builder.addUnary(linear2, 2, 3);
  builder.setOutput(3);
  const graph = builder.compile(registry);
  return { registry, linear1, relu, linear2, builder, graph };
}

const esCaps = JSON.parse(m.esCapabilities());
const programCaps = JSON.parse(m.programCapabilities());
const bundleCaps = JSON.parse(m.programBundleCapabilities());
assert(esCaps.lifecycle === 'ask->tell', 'ES lifecycle capability mismatch');
assert(esCaps.strict_factory === 'EsOptimizer.strict', 'strict ES factory capability mismatch');
assert(programCaps.execution_binding === 'required', 'CompiledGraph execution binding must remain required');
assert(programCaps.mutable_state_in_identity === false, 'mutable weights/state must remain outside structural program identity');
assert(bundleCaps.mutable_state === 'optional_separate_section', 'program bundle mutable-state separation mismatch');

// This audit intentionally remains a compatibility proof for the raw Registry/CompiledGraph
// slice that motivated graph-parameter-binding v1. Newer graph-level binding APIs are a
// separate layer and must not make this legacy proof claim that CompiledGraph itself owns
// optimizer/controller semantics.
const packagedGraphParameterBindingPresent =
  typeof m.graphParameterLayout === 'function' &&
  typeof m.graphParameterIdentity === 'function' &&
  typeof m.getGraphParametersFlat === 'function' &&
  typeof m.setGraphParametersFlat === 'function';

// -----------------------------------------------------------------------------
// 1. Single-layer candidate binding through the raw Registry is deterministic.
// -----------------------------------------------------------------------------
const single = buildSingleLinear();
const singleIdentity = single.graph.programIdentity();
const singleLayout = single.registry.weightLayout(101, LAYER_LINEAR);
const singleOriginal = Array.from(single.registry.getWeightsFlat(101, LAYER_LINEAR));
assert(singleOriginal.length > 0, 'single Linear must expose optimizer-visible flat weights');
assert(typeof singleLayout === 'string' && singleLayout.length > 0, 'single Linear weight layout must be discoverable');

const esA = m.EsOptimizer.strict(singleOriginal.length, 0, 424242, 8, 0.1, 0.05);
const esB = m.EsOptimizer.strict(singleOriginal.length, 0, 424242, 8, 0.1, 0.05);
const askA = Array.from(esA.ask());
const askB = Array.from(esB.ask());
assert(exactArrayEqual(askA, askB), 'same strict ES seed/config must produce the same first candidate batch');
assert(askA.every(Number.isFinite), 'strict ES first candidate batch must be finite');
assert(esA.batchSize() === 8, 'strict OpenES batch size mismatch');
assert(askA.length === singleOriginal.length * esA.batchSize(), 'flat ES candidate batch shape mismatch');

// Do not make this proof depend on randomized layer initialization. Establish two explicit
// parameter states and prove that compatible mutation changes execution while identity stays put.
const deterministicSingleBaseline = filled(singleOriginal.length, 0.0);
const deterministicSingleProbe = filled(singleOriginal.length, 0.25);
single.registry.setWeightsFlat(101, LAYER_LINEAR, deterministicSingleBaseline);
const beforeSingle = runVector(single.graph, single.registry, [0.75, -1.25]);
single.registry.setWeightsFlat(101, LAYER_LINEAR, deterministicSingleProbe);
const afterSingle = runVector(single.graph, single.registry, [0.75, -1.25]);
assert(!exactArrayEqual(beforeSingle, afterSingle), 'explicit compatible parameter states should affect graph execution');
assert(single.graph.programIdentity() === singleIdentity, 'compatible weight mutation changed structural program identity');
single.graph.validateRegistryBinding(single.registry);

const firstCandidate = askA.slice(0, singleOriginal.length);
single.registry.setWeightsFlat(101, LAYER_LINEAR, new Float32Array(firstCandidate));
const appliedSingle = Array.from(single.registry.getWeightsFlat(101, LAYER_LINEAR));
assert(exactArrayEqual(appliedSingle, firstCandidate), 'single ES candidate was not applied exactly');

const dataset = [
  { x: [1, 0], y: 0.5 },
  { x: [0, 1], y: -0.25 },
  { x: [1, 1], y: 0.25 },
  { x: [-1, 2], y: -1.0 },
];
function singleFitness(candidate) {
  single.registry.setWeightsFlat(101, LAYER_LINEAR, new Float32Array(candidate));
  let mse = 0;
  for (const sample of dataset) {
    const got = runVector(single.graph, single.registry, sample.x);
    assert(got.length === 1 && Number.isFinite(got[0]), 'single graph produced invalid scalar output');
    const error = got[0] - sample.y;
    mse += error * error;
  }
  return -(mse / dataset.length);
}

const fitnesses = [];
for (let i = 0; i < esA.batchSize(); i += 1) {
  const start = i * singleOriginal.length;
  const candidate = askA.slice(start, start + singleOriginal.length);
  fitnesses.push(singleFitness(candidate));
}
assert(fitnesses.every(Number.isFinite), 'graph-derived ES fitness must remain finite');
esA.tell(new Float32Array(fitnesses));
assert(esA.generation() === 1, 'ES generation should advance after one valid graph-derived fitness batch');

// Per-layer malformed application is fail-closed and retryable at that layer boundary.
single.registry.setWeightsFlat(101, LAYER_LINEAR, new Float32Array(firstCandidate));
const beforeMalformedSingle = Array.from(single.registry.getWeightsFlat(101, LAYER_LINEAR));
expectThrow(
  () => single.registry.setWeightsFlat(101, LAYER_LINEAR, new Float32Array(firstCandidate.slice(0, -1))),
  'single-layer malformed candidate length',
);
assert(
  exactArrayEqual(Array.from(single.registry.getWeightsFlat(101, LAYER_LINEAR)), beforeMalformedSingle),
  'failed single-layer candidate application mutated prior valid weights',
);
single.registry.setWeightsFlat(101, LAYER_LINEAR, new Float32Array(singleOriginal));
assert(single.graph.programIdentity() === singleIdentity, 'single-layer restore changed structural identity');

// -----------------------------------------------------------------------------
// 2. Raw Registry composition remains per-layer. Canonical graph-level binding,
//    when packaged, is a separate capability rather than a CompiledGraph method.
// -----------------------------------------------------------------------------
const multi = buildMultiLayer();
const multiIdentity = multi.graph.programIdentity();
const w1Original = Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR));
const w2Original = Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR));
const layout1 = multi.registry.weightLayout(201, LAYER_LINEAR);
const layout2 = multi.registry.weightLayout(203, LAYER_LINEAR);
assert(w1Original.length > 0 && w2Original.length > 0, 'both trainable Linear layers must expose flat weights');
assert(layout1.length > 0 && layout2.length > 0, 'both trainable Linear layers must expose layouts');
expectThrow(
  () => multi.registry.getWeightsFlat(202, LAYER_ACTIVATION),
  'stateless ReLU optimizer-visible weights',
);

const multiRefs = parsePlanLayerRefs(multi.graph.programPlan());
const multiUniqueRefs = uniqueRefs(multiRefs);
const hostChosenTrainableRefs = multiUniqueRefs.filter((ref) => ref.layerType === LAYER_LINEAR);
assert(hostChosenTrainableRefs.length === 2, 'audit host should discover two unique Linear refs from this plan');
const hostLayout = [];
let hostOffset = 0;
for (const ref of hostChosenTrainableRefs) {
  const weights = Array.from(multi.registry.getWeightsFlat(ref.layerId, ref.layerType));
  hostLayout.push({ layerType: ref.layerType, layerId: ref.layerId, offset: hostOffset, length: weights.length });
  hostOffset += weights.length;
}
assert(hostOffset === w1Original.length + w2Original.length, 'host-derived candidate length mismatch');

// CompiledGraph remains structural/execution authority. Even when graph parameter-binding is
// available, it is intentionally exposed as a separate free-function capability.
const compiledGraphOwnsParameterSurface =
  typeof multi.graph.parameterLayout === 'function' ||
  typeof multi.graph.parameterPlan === 'function' ||
  typeof multi.graph.getWeightsFlat === 'function' ||
  typeof multi.graph.setWeightsFlat === 'function';
assert(!compiledGraphOwnsParameterSurface, 'CompiledGraph should not absorb optimizer/controller parameter ownership');

const w1Candidate = filled(w1Original.length, 0.5);
const w2Candidate = filled(w2Original.length, -0.25);
multi.registry.setWeightsFlat(201, LAYER_LINEAR, w1Candidate);
multi.registry.setWeightsFlat(203, LAYER_LINEAR, w2Candidate);
assert(
  exactArrayEqual(Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR)), Array.from(w1Candidate)),
  'first deterministic raw Registry candidate did not apply exactly',
);
assert(
  exactArrayEqual(Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR)), Array.from(w2Candidate)),
  'second deterministic raw Registry candidate did not apply exactly',
);
assert(runVector(multi.graph, multi.registry, [0.5, -1.0]).every(Number.isFinite), 'multi-layer graph output must remain finite');
assert(multi.graph.programIdentity() === multiIdentity, 'multi-layer compatible weights changed structural program identity');
multi.graph.validateRegistryBinding(multi.registry);

// Repeated references prove that raw step count cannot be used as candidate cardinality.
const repeatBuilder = new m.AgentGraphBuilder(3);
repeatBuilder.addUnary(multi.linear1, 0, 1);
repeatBuilder.addUnary(multi.linear1, 1, 2);
repeatBuilder.setOutput(2);
const repeatGraph = repeatBuilder.compile(multi.registry);
const repeatRefs = parsePlanLayerRefs(repeatGraph.programPlan()).filter((ref) => ref.layerType === LAYER_LINEAR);
const repeatUnique = uniqueRefs(repeatRefs);
assert(repeatRefs.length === 2, 'repeat graph must contain two uses of the same Linear layer');
assert(repeatUnique.length === 1 && repeatUnique[0].layerId === 201, 'candidate binding must deduplicate repeated layer references');

// -----------------------------------------------------------------------------
// 3. Raw per-layer setters are deliberately not a composite transaction.
//    This remains a local Registry fact; graph-parameter-binding v1 owns the
//    canonical whole-candidate prevalidation/atomic-apply contract when present.
// -----------------------------------------------------------------------------
multi.registry.setWeightsFlat(201, LAYER_LINEAR, new Float32Array(w1Original));
multi.registry.setWeightsFlat(203, LAYER_LINEAR, new Float32Array(w2Original));
const atomicProbeFirst = filled(w1Original.length, 0.5);
const secondBeforeAtomicProbe = Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR));
multi.registry.setWeightsFlat(201, LAYER_LINEAR, atomicProbeFirst);
expectThrow(
  () => multi.registry.setWeightsFlat(203, LAYER_LINEAR, new Float32Array(w2Original.slice(0, -1))),
  'later malformed layer slice during raw per-layer composite application',
);
const firstAfterAtomicFailure = Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR));
const secondAfterAtomicFailure = Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR));
assert(
  exactArrayEqual(firstAfterAtomicFailure, Array.from(atomicProbeFirst)),
  'audit expected the earlier valid raw per-layer mutation to remain committed',
);
assert(
  exactArrayEqual(secondAfterAtomicFailure, secondBeforeAtomicProbe),
  'malformed later raw layer slice should not mutate that layer',
);
const rawPerLayerCompositeAtomic = false;

// Explicit host rollback restores the raw Registry slice. This is not the canonical graph-level
// candidate protocol once graph-parameter-binding v1 exists.
multi.registry.setWeightsFlat(201, LAYER_LINEAR, new Float32Array(w1Original));
multi.registry.setWeightsFlat(203, LAYER_LINEAR, new Float32Array(w2Original));
assert(multi.graph.programIdentity() === multiIdentity, 'rollback changed structural program identity');
multi.graph.validateRegistryBinding(multi.registry);

const findings = [
  {
    boundary: 'es_optimizer_core',
    score: 0,
    classification: 'KEEP',
    evidence: 'strict seeded ask/tell supplies deterministic finite flat candidates and accepts graph-derived scalar fitness',
  },
  {
    boundary: 'single_layer_raw_registry_binding',
    score: 0,
    classification: 'KEEP',
    evidence: 'one trainable layer maps through getWeightsFlat/setWeightsFlat and preserves CompiledGraph structural identity',
  },
  {
    boundary: 'compiled_graph_parameter_ownership',
    score: 0,
    classification: 'KEEP',
    evidence: 'CompiledGraph remains structural/execution authority and does not absorb optimizer/controller parameter methods',
  },
  {
    boundary: 'canonical_graph_parameter_binding',
    score: packagedGraphParameterBindingPresent ? 0 : 2,
    classification: packagedGraphParameterBindingPresent ? 'KEEP' : 'SPLIT',
    evidence: packagedGraphParameterBindingPresent
      ? 'a separate packaged graph-level parameter-binding capability is present; its dedicated research gate owns canonical layout/atomicity proofs'
      : 'raw Registry composition still requires host-chosen unique-owner layout and offsets',
  },
  {
    boundary: 'raw_registry_composite_atomicity',
    score: packagedGraphParameterBindingPresent ? 0 : 3,
    classification: packagedGraphParameterBindingPresent ? 'KEEP' : 'SPLIT',
    evidence: packagedGraphParameterBindingPresent
      ? 'raw per-layer setters remain non-transactional by design; whole-candidate atomicity belongs to the separate graph parameter-binding layer'
      : 'a valid early raw per-layer mutation remains committed when a later setter rejects its slice',
  },
  {
    boundary: 'compiled_graph_structural_identity',
    score: 0,
    classification: 'KEEP',
    evidence: 'compatible mutable weights do not alter structural program identity or Registry binding',
  },
  {
    boundary: 'program_bundle_relationship',
    score: 0,
    classification: 'KEEP',
    evidence: 'program-bundle.v1 keeps mutable state separate and is not reinterpreted as an ES candidate vector format',
  },
];

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'ES -> CompiledGraph raw Registry compatibility audit',
  issue: 161,
  singleLayer: {
    parameterCount: singleOriginal.length,
    weightLayout: singleLayout,
    sameSeedFirstBatch: true,
    finiteFirstBatch: true,
    graphDerivedFitness: true,
    deterministicExplicitStateProbe: true,
    malformedCandidateAtomicAtLayerBoundary: true,
    programIdentityStable: true,
  },
  multiLayer: {
    layerParameterCounts: [w1Original.length, w2Original.length],
    weightLayouts: [layout1, layout2],
    hostChosenParameterLayout: hostLayout,
    packagedGraphParameterBindingPresent,
    compiledGraphOwnsParameterSurface,
    repeatedLayerStepCount: repeatRefs.length,
    repeatedLayerUniqueParameterOwnerCount: repeatUnique.length,
    statelessLayerConsumesCandidateCoordinates: false,
    rawPerLayerCompositeAtomic,
    rawLaterFailureLeavesEarlierMutationCommitted: true,
    programIdentityStable: true,
  },
  findings,
  nextRecommendedProductionSlice: packagedGraphParameterBindingPresent
    ? 'keep ES, CompiledGraph, and raw Registry responsibilities separate; validate graph-level binding in its dedicated gate before any controller work'
    : 'separate canonical graph parameter binding v1: deterministic unique-layer layout + full prevalidation + atomic candidate application; keep ES and CompiledGraph algorithms unchanged',
  autodiffRequired: false,
  grantsAuthority: false,
}, null, 2));

repeatGraph.free();
repeatBuilder.free();
multi.graph.free();
multi.builder.free();
multi.linear1.free();
multi.relu.free();
multi.linear2.free();
multi.registry.free();
esA.free();
esB.free();
single.graph.free();
single.builder.free();
single.linear.free();
single.registry.free();