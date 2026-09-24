import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const moduleUrl = pathToFileURL(path.join(pkgDir, 'operation_contract_registry.mjs')).href;
const {
  loadOperationContractRegistry,
  queryOperationPool,
  resolveDeclarativeGraphPlan,
  OPERATION_REGISTRY_SCHEMAS,
} = await import(moduleUrl);

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

for (const file of [
  'operation_contract_registry.mjs',
  'host-operation-contract-registry.v1.json',
  'agent-layer-catalog.v1.json',
  'agent-layout-contracts.v1.json',
  'agent-input-port.v1.json',
  'agent-fault-contract.v1.json',
  'multi-input-graph-plan.v1.json',
  'multi-input-plan-explain.v1.json',
  'host-support.v1.json',
  'runtime-surface.v1.json',
]) {
  assert(fs.existsSync(path.join(pkgDir, file)), `packaged operation-registry dependency missing: ${file}`);
}

const hostSupport = JSON.parse(fs.readFileSync(path.join(pkgDir, 'host-support.v1.json'), 'utf8'));
assert(hostSupport.verified_hosts?.node?.operation_registry_module === 'operation_contract_registry.mjs', 'Node host does not advertise operation registry module');
assert(hostSupport.verified_hosts?.node?.operation_registry_contract === 'host-operation-contract-registry.v1.json', 'Node host does not advertise operation registry contract');
assert(hostSupport.verified_hosts?.node?.operation_registry_scope?.includes('constructible graph operation type pool'), 'Node host operation registry scope missing');

const registryContract = JSON.parse(fs.readFileSync(path.join(pkgDir, 'host-operation-contract-registry.v1.json'), 'utf8'));
assert(registryContract.schema === 'burn-research.host-operation-contract-registry.v1', 'operation registry host contract schema mismatch');
assert(registryContract.authority?.execution_authorized === false, 'operation registry contract must not authorize execution');
assert(registryContract.pool?.live_registry_instance_inventory?.startsWith('deferred'), 'operation registry contract overclaims live registry inventory');

const runtimeSurface = JSON.parse(fs.readFileSync(path.join(pkgDir, 'runtime-surface.v1.json'), 'utf8'));
assert(runtimeSurface.host_capability_contracts?.contracts?.operation_contract_registry === 'host-operation-contract-registry.v1.json', 'runtime surface does not discover operation registry contract');
for (const file of [
  'agent-layer-catalog.v1.json',
  'agent-layout-contracts.v1.json',
  'agent-input-port.v1.json',
  'agent-fault-contract.v1.json',
  'multi-input-graph-plan.v1.json',
  'multi-input-plan-explain.v1.json',
]) {
  assert(runtimeSurface.artifact_identity?.files?.includes(file), `runtime artifact identity does not bind ${file}`);
}

const registry = loadOperationContractRegistry(pkgDir);
assert(registry.schema === OPERATION_REGISTRY_SCHEMAS.registry, 'registry schema mismatch');
assert(registry.execution_authorized === false && registry.mutation === 'none', 'registry must remain projection-only');
assert(registry.pool_kind === 'constructible_operation_types_not_live_registry_instances', 'pool kind overclaims live registry inventory');
assert(registry.capacity.graph_slots_max === 64, 'graph slot capacity mismatch');
assert(registry.operations.length >= 30, 'operation pool unexpectedly small');
assert(/^sha256:[0-9a-f]{64}$/.test(registry.registry_fingerprint), 'registry fingerprint malformed');

const linear = registry.operations.find(operation => operation.operation_id === 'graph.layer.linear');
assert(linear, 'linear operation missing from registry');
assert(linear.arity === 1 && linear.inputs.length === 1, 'linear arity contract mismatch');
assert(linear.inputs[0].dtype === 'f32' && linear.inputs[0].rank === 4, 'linear tensor contract mismatch');
assert(linear.inputs[0].layout === 'feature_axis1_singleton', 'linear input layout mismatch');
assert(linear.output.layout === 'feature_axis1_singleton', 'linear output layout mismatch');
assert(linear.shape_rule === 'linear_feature_axis1_singleton', 'linear shape rule mismatch');
assert(linear.failure_classes.includes('LAYOUT_INCOMPATIBLE'), 'linear failure taxonomy missing layout incompatibility');
assert(linear.remaining_preflight.some(item => item.includes('LayerRegistry')), 'linear descriptor must preserve registry authority');

const concat = registry.operations.find(operation => operation.operation_id === 'graph.layer.concat');
assert(concat?.axis_contract?.mode === 'parameter', 'concat axis contract must be parameterized');
assert(concat.axis_contract.domain.minimum === 0 && concat.axis_contract.domain.maximum === 3, 'concat rank-4 axis domain mismatch');

const exactPool = queryOperationPool(registry, {operation_id: 'graph.layer.linear', arity: 1});
assert(
  exactPool.status === 'RESOLVED' && exactPool.candidate_count === 1,
  `exact operation pool query should resolve: ${JSON.stringify({
    status: exactPool.status,
    candidate_count: exactPool.candidate_count,
    candidates: exactPool.candidates,
    registry_fingerprint: exactPool.registry_fingerprint,
    linear: {
      operation_id: linear.operation_id,
      constructor: linear.constructor,
      family: linear.family,
      arity: linear.arity,
      input_layout: linear.inputs[0].layout,
      output_layout: linear.output.layout,
    },
  })}`,
);
assert(exactPool.execution_authorized === false, 'pool query must not authorize execution');

const inheritedSelector = Object.create({constructor: 'relu'});
inheritedSelector.operation_id = 'graph.layer.linear';
inheritedSelector.arity = 1;
const inheritedPool = queryOperationPool(registry, inheritedSelector);
assert(inheritedPool.status === 'RESOLVED' && inheritedPool.candidate_count === 1, 'inherited selector properties must not become caller intent');
assert(inheritedPool.candidates[0]?.operation_id === 'graph.layer.linear', 'inherited constructor must not override exact own operation_id');

for (const malformedQuery of [null, [], 'graph.layer.linear', 1, true]) {
  const malformedPool = queryOperationPool(registry, malformedQuery);
  assert(malformedPool.status === 'REJECTED', 'malformed pool query must reject instead of throwing');
  assert(malformedPool.candidate_count === 0, 'malformed pool query must expose no candidates');
  assert(malformedPool.execution_authorized === false && malformedPool.mutation === 'none', 'malformed pool query must remain nonexecuting');
  assert(malformedPool.diagnostics?.some(item => item.code === 'INVALID_PARAMETER'), 'malformed pool query must expose INVALID_PARAMETER');
}

const canonicalSubsetPool = queryOperationPool(registry, {
  operation_id: 'graph.layer.batchNorm',
  input_layout: 'feature_axis1_singleton',
});
assert(canonicalSubsetPool.status === 'RESOLVED', 'feature_axis1_singleton must remain compatible with channel_first consumer according to canonical layout contract');

const conditionalPool = queryOperationPool(registry, {
  operation_id: 'graph.layer.linear',
  input_layout: 'channel_first',
});
assert(conditionalPool.status === 'DEFERRED' || conditionalPool.status === 'REJECTED', 'channel_first -> feature_axis1_singleton must never be overclaimed as statically resolved');

const ambiguousPool = queryOperationPool(registry, {family: 'activation', arity: 1});
assert(ambiguousPool.status === 'AMBIGUOUS' && ambiguousPool.candidate_count > 1, 'activation family should expose an ambiguous pool rather than rank a winner');

const rejectedPool = queryOperationPool(registry, {operation_id: 'graph.layer.does-not-exist'});
assert(rejectedPool.status === 'REJECTED' && rejectedPool.candidate_count === 0, 'unknown operation should reject');

const resolvedPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'observation', role: 'observation', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton'},
  ],
  steps: [
    {
      id: 'project',
      operation: 'graph.layer.linear',
      inputs: ['observation'],
      parameters: {id: 1, in_dim: 4, out_dim: 8, bias: true},
      output: 'features',
      output_role: 'feature',
    },
    {
      id: 'activate',
      operation: 'graph.layer.relu',
      inputs: ['features'],
      parameters: {id: 2},
      output: 'activated',
      output_role: 'feature',
    },
  ],
  output: 'activated',
});
assert(resolvedPlan.status === 'RESOLVED', `exact static graph should resolve, got ${resolvedPlan.status}: ${JSON.stringify(resolvedPlan.diagnostics)}`);
assert(resolvedPlan.execution_authorized === false && resolvedPlan.mutation === 'none', 'resolved plan must remain nonexecuting');
assert(JSON.stringify(resolvedPlan.final_output?.shape) === JSON.stringify([1, 8, 1, 1]), 'resolved plan output shape mismatch');
assert(resolvedPlan.final_output?.role === 'feature', 'resolved plan output role mismatch');

const compatibleChainPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'x', role: 'feature', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton'},
  ],
  steps: [
    {
      id: 'norm',
      operation: 'graph.layer.batchNorm',
      inputs: ['x'],
      parameters: {id: 20, num_features: 4},
      output: 'y',
      output_role: 'feature',
    },
  ],
  output: 'y',
});
assert(compatibleChainPlan.status === 'RESOLVED', `canonical feature-axis subset -> channel_first plan should resolve, got ${compatibleChainPlan.status}`);

const ambiguousPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'x', role: 'feature', shape: [1, 4, 1, 1], layout: 'any_rank4'},
  ],
  steps: [
    {
      id: 'choose_activation',
      selector: {family: 'activation', arity: 1},
      inputs: ['x'],
      parameters: {},
      output: 'y',
      output_role: 'feature',
    },
  ],
  output: 'y',
});
assert(ambiguousPlan.status === 'AMBIGUOUS', `family selector must stay ambiguous, got ${ambiguousPlan.status}`);
assert(ambiguousPlan.step_results[0].candidate_operations.length > 1, 'ambiguous plan lost its candidate pool');

const deferredPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'image', role: 'observation', shape: [1, 3, 16, 16], layout: 'channel_first'},
  ],
  steps: [
    {
      id: 'conv',
      operation: 'graph.layer.conv2d',
      inputs: ['image'],
      parameters: {
        id: 3,
        in_ch: 3,
        out_ch: 8,
        kh: 3,
        kw: 3,
      },
      output: 'conv_out',
      output_role: 'feature',
    },
  ],
  output: 'conv_out',
});
assert(deferredPlan.status === 'DEFERRED', `conv static output should defer to canonical graph/Burn rules, got ${deferredPlan.status}`);
assert(deferredPlan.step_results[0].selected_operation === 'graph.layer.conv2d', 'deferred plan must retain exact selected operation identity');

const layoutRejectedPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'image', role: 'observation', shape: [1, 3, 16, 16], layout: 'channel_first'},
  ],
  steps: [
    {
      id: 'bad_linear',
      operation: 'graph.layer.linear',
      inputs: ['image'],
      parameters: {id: 4, in_dim: 3, out_dim: 8, bias: true},
      output: 'bad',
      output_role: 'feature',
    },
  ],
  output: 'bad',
});
assert(layoutRejectedPlan.status === 'REJECTED', 'known channel_first -> linear feature-axis shape mismatch must reject');
assert(layoutRejectedPlan.diagnostics.some(item => item.code === 'LAYOUT_INCOMPATIBLE' || item.code === 'SHAPE_INCOMPATIBLE'), 'layout rejection diagnostic missing');

const axisRejectedPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'a', role: 'feature', shape: [1, 2, 1, 1], layout: 'any_rank4'},
    {id: 'b', role: 'feature', shape: [1, 3, 1, 1], layout: 'any_rank4'},
  ],
  steps: [
    {
      id: 'concat',
      operation: 'graph.layer.concat',
      inputs: ['a', 'b'],
      parameters: {id: 5, dim: 4},
      output: 'joined',
      output_role: 'feature',
    },
  ],
  output: 'joined',
});
assert(axisRejectedPlan.status === 'REJECTED', 'rank-4 concat dim=4 must reject before execution');
assert(axisRejectedPlan.diagnostics.some(item => item.code === 'AXIS_OUT_OF_RANGE'), 'axis rejection diagnostic missing');

const missingParameterPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: [
    {id: 'x', role: 'feature', shape: [1, 4, 1, 1], layout: 'feature_axis1_singleton'},
  ],
  steps: [
    {
      id: 'linear',
      operation: 'graph.layer.linear',
      inputs: ['x'],
      parameters: {id: 6, in_dim: 4, bias: true},
      output: 'y',
      output_role: 'feature',
    },
  ],
  output: 'y',
});
assert(missingParameterPlan.status === 'AMBIGUOUS', 'missing required out_dim should be ambiguous caller information, not execution failure');

const capacityInputs = Array.from({length: 64}, (_, index) => ({
  id: `input_${index}`,
  role: 'feature',
  shape: [1, 1, 1, 1],
  layout: 'any_rank4',
}));
const capacityRejectedPlan = resolveDeclarativeGraphPlan(registry, {
  schema: 'burn-research.declarative-graph-plan.v1',
  inputs: capacityInputs,
  steps: [
    {
      id: 'extra_value',
      operation: 'graph.layer.relu',
      inputs: ['input_0'],
      parameters: {id: 7},
      output: 'value_65',
      output_role: 'feature',
    },
  ],
  output: 'value_65',
});
assert(capacityRejectedPlan.status === 'REJECTED', 'more than 64 distinct graph values must reject at planning boundary');
assert(capacityRejectedPlan.diagnostics.some(item => item.code === 'CAPACITY_EXCEEDED'), 'capacity rejection diagnostic missing');

console.log(JSON.stringify({
  verdict: 'PASS',
  schema: registry.schema,
  registry_fingerprint: registry.registry_fingerprint,
  operation_count: registry.operations.length,
  canonical_subset_pool: canonicalSubsetPool.status,
  statuses: {
    resolved: resolvedPlan.status,
    compatible_chain: compatibleChainPlan.status,
    ambiguous: ambiguousPlan.status,
    deferred: deferredPlan.status,
    rejected_layout: layoutRejectedPlan.status,
    rejected_axis: axisRejectedPlan.status,
    rejected_capacity: capacityRejectedPlan.status,
  },
  execution_authorized: false,
  live_registry_instance_inventory: false,
}));