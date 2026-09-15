import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

const LAYER_LINEAR = 0x01;

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

function tensor(values) {
  return new m.WasmTensor(
    new Float32Array(values),
    new Uint32Array([1, values.length, 1, 1]),
  );
}

function runVector(graph, registry, values) {
  const input = tensor(values);
  const output = graph.run(registry, input);
  const result = Array.from(output.to_array());
  output.free();
  input.free();
  return result;
}

function buildMultiLayer() {
  const registry = new m.LayerRegistry();
  const first = m.AgentLayerSpec.linear(201, 2, 2, true);
  const relu = m.AgentLayerSpec.relu(202);
  const second = m.AgentLayerSpec.linear(203, 2, 1, true);
  registry.initAgentLayer(first);
  registry.initAgentLayer(relu);
  registry.initAgentLayer(second);

  const builder = new m.AgentGraphBuilder(4);
  builder.addUnary(first, 0, 1);
  builder.addUnary(relu, 1, 2);
  builder.addUnary(second, 2, 3);
  builder.setOutput(3);
  const graph = builder.compile(registry);
  return { registry, first, relu, second, builder, graph };
}

const caps = JSON.parse(m.graphParameterCapabilities());
assert(caps.schema_version === 1, 'graph parameter capability schema version mismatch');
assert(caps.schema === 'burn-research.graph-parameter-binding.v1', 'binding schema mismatch');
assert(caps.layout_schema === 'burn-research.graph-parameter-layout.v1', 'layout schema mismatch');
assert(caps.ordering === 'unique_first_use_graph_plan', 'owner ordering mismatch');
assert(caps.apply_atomicity === 'full_prevalidation_before_first_mutation', 'atomicity contract mismatch');
assert(caps.mutable_parameter_values_in_identity === false, 'mutable values must remain outside binding identity');
assert(caps.program_bundle_is_candidate_format === false, 'program bundle must not become candidate format');
assert(caps.optimizer_state_in_binding === false, 'optimizer state must remain outside binding');

// -----------------------------------------------------------------------------
// 1. Packaged multi-layer layout is canonical and excludes stateless ReLU.
// -----------------------------------------------------------------------------
const multi = buildMultiLayer();
const programIdentity = multi.graph.programIdentity();
const layout = JSON.parse(m.graphParameterLayout(multi.graph, multi.registry));
const bindingIdentity = m.graphParameterIdentity(multi.graph, multi.registry);
const bindingIdentityParsed = JSON.parse(bindingIdentity);
const flat = Array.from(m.getGraphParametersFlat(multi.graph, multi.registry));
const firstWeights = Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR));
const secondWeights = Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR));

assert(layout.schema === 'burn-research.graph-parameter-layout.v1', 'packaged layout schema mismatch');
assert(layout.ordering === 'unique_first_use_graph_plan', 'packaged layout ordering mismatch');
assert(layout.owners.length === 2, 'Linear -> ReLU -> Linear should expose exactly two trainable owners');
assert(layout.owners[0].layer_id === 201 && layout.owners[1].layer_id === 203, 'owner order must follow unique first use');
assert(layout.owners[0].offset === 0, 'first owner offset must be zero');
assert(layout.owners[0].len === firstWeights.length, 'first owner length mismatch');
assert(layout.owners[1].offset === firstWeights.length, 'second owner offset mismatch');
assert(layout.owners[1].len === secondWeights.length, 'second owner length mismatch');
assert(layout.total_len === firstWeights.length + secondWeights.length, 'graph total candidate length mismatch');
assert(flat.length === layout.total_len, 'packaged flat vector length does not match layout');
assert(
  exactArrayEqual(flat, [...firstWeights, ...secondWeights]),
  'packaged read order does not match canonical owner concatenation',
);
assert(!layout.owners.some((owner) => owner.layer_id === 202), 'stateless ReLU consumed candidate coordinates');
assert(bindingIdentityParsed.program_identity.schema === 'burn-research.program-identity.v1', 'binding identity must embed structural program identity');
assert(bindingIdentityParsed.owners.length === 2, 'binding identity owner cardinality mismatch');

// -----------------------------------------------------------------------------
// 2. Malformed composite candidate fails before any layer mutation.
// -----------------------------------------------------------------------------
const beforeFirst = Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR));
const beforeSecond = Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR));
expectThrow(
  () => m.setGraphParametersFlat(multi.graph, multi.registry, new Float32Array(flat.slice(0, -1))),
  'malformed graph candidate length',
);
assert(
  exactArrayEqual(Array.from(multi.registry.getWeightsFlat(201, LAYER_LINEAR)), beforeFirst),
  'failed composite apply mutated first layer',
);
assert(
  exactArrayEqual(Array.from(multi.registry.getWeightsFlat(203, LAYER_LINEAR)), beforeSecond),
  'failed composite apply mutated second layer',
);

// -----------------------------------------------------------------------------
// 3. Successful composite apply changes execution but not structural/binding identity.
// -----------------------------------------------------------------------------
const candidate = new Float32Array(layout.total_len);
candidate.fill(1.0);
m.setGraphParametersFlat(multi.graph, multi.registry, candidate);
const after = Array.from(m.getGraphParametersFlat(multi.graph, multi.registry));
assert(exactArrayEqual(after, Array.from(candidate)), 'successful composite candidate did not replay exactly');
const output = runVector(multi.graph, multi.registry, [1, 2]);
assert(output.length === 1 && Math.abs(output[0] - 9.0) < 1e-5, 'known all-ones network output mismatch');
assert(multi.graph.programIdentity() === programIdentity, 'mutable candidate changed structural program identity');
assert(m.graphParameterIdentity(multi.graph, multi.registry) === bindingIdentity, 'mutable candidate changed binding identity');
assert(
  m.graphParameterLayout(multi.graph, multi.registry) === JSON.stringify(layout),
  'mutable candidate changed canonical layout',
);
multi.graph.validateRegistryBinding(multi.registry);

// -----------------------------------------------------------------------------
// 4. Repeated graph references deduplicate one trainable owner.
// -----------------------------------------------------------------------------
const repeatBuilder = new m.AgentGraphBuilder(3);
repeatBuilder.addUnary(multi.first, 0, 1);
repeatBuilder.addUnary(multi.first, 1, 2);
repeatBuilder.setOutput(2);
const repeatGraph = repeatBuilder.compile(multi.registry);
const repeatLayout = JSON.parse(m.graphParameterLayout(repeatGraph, multi.registry));
assert(repeatLayout.owners.length === 1, 'repeated trainable layer reference was not deduplicated');
assert(repeatLayout.owners[0].layer_id === 201, 'repeated owner id mismatch');
assert(repeatLayout.total_len === firstWeights.length, 'repeated layer consumed duplicate coordinates');

// -----------------------------------------------------------------------------
// 5. Parameterized layers without a flat bridge fail closed.
// -----------------------------------------------------------------------------
const unsupportedRegistry = new m.LayerRegistry();
const prelu = m.AgentLayerSpec.prelu(301, 2, 0.25);
unsupportedRegistry.initAgentLayer(prelu);
const unsupportedBuilder = new m.AgentGraphBuilder(2);
unsupportedBuilder.addUnary(prelu, 0, 1);
unsupportedBuilder.setOutput(1);
const unsupportedGraph = unsupportedBuilder.compile(unsupportedRegistry);
expectThrow(
  () => m.graphParameterLayout(unsupportedGraph, unsupportedRegistry),
  'PReLU graph parameter binding without flat bridge',
);

// -----------------------------------------------------------------------------
// 6. Structural replacement is rejected before touching another owner.
// -----------------------------------------------------------------------------
const structural = buildMultiLayer();
const structuralFlat = Array.from(m.getGraphParametersFlat(structural.graph, structural.registry));
const structuralFirstBefore = Array.from(structural.registry.getWeightsFlat(201, LAYER_LINEAR));
const replacement = m.AgentLayerSpec.linear(203, 2, 2, true);
structural.registry.initAgentLayer(replacement);
expectThrow(
  () => m.setGraphParametersFlat(structural.graph, structural.registry, new Float32Array(structuralFlat)),
  'structurally stale graph parameter binding',
);
assert(
  exactArrayEqual(Array.from(structural.registry.getWeightsFlat(201, LAYER_LINEAR)), structuralFirstBefore),
  'structural mismatch mutated an earlier compatible owner',
);

// -----------------------------------------------------------------------------
// 7. Existing deterministic ES can consume this boundary without optimizer coupling.
// -----------------------------------------------------------------------------
const esGraph = buildMultiLayer();
const esLayout = JSON.parse(m.graphParameterLayout(esGraph.graph, esGraph.registry));
const es = m.EsOptimizer.strict(esLayout.total_len, 0, 777, 8, 0.1, 0.05);
const batch = Array.from(es.ask());
assert(batch.length === esLayout.total_len * es.batchSize(), 'ES candidate batch dimension does not match graph binding');
const firstCandidate = batch.slice(0, esLayout.total_len);
m.setGraphParametersFlat(esGraph.graph, esGraph.registry, new Float32Array(firstCandidate));
const esOutput = runVector(esGraph.graph, esGraph.registry, [0.5, -1.0]);
assert(esOutput.every(Number.isFinite), 'ES-applied graph produced non-finite output');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'packaged graph parameter-binding v1 research',
  issue: 164,
  layout: {
    ordering: layout.ordering,
    ownerIds: layout.owners.map((owner) => owner.layer_id),
    totalLen: layout.total_len,
    statelessReluCoordinates: 0,
  },
  proofs: {
    exactReadApplyRead: true,
    malformedCompositeAtomic: true,
    structuralMismatchFailsBeforeEarlierOwnerMutation: true,
    repeatedOwnerDeduplicated: true,
    unsupportedParameterizedOwnerFailsClosed: true,
    programIdentityStableAcrossMutableCandidate: true,
    bindingIdentityStableAcrossMutableCandidate: true,
    deterministicEsConsumesBoundary: true,
  },
  separation: {
    optimizerStateInBinding: false,
    programBundleIsCandidateFormat: false,
    autodiffRequired: false,
  },
}, null, 2));
