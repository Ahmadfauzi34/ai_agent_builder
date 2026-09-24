import {createHash} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const REGISTRY_SCHEMA = 'burn-research.operation-contract-registry.v1';
const POOL_SCHEMA = 'burn-research.operation-pool-query.v1';
const PLAN_SCHEMA = 'burn-research.declarative-graph-plan.v1';
const RESOLUTION_SCHEMA = 'burn-research.declarative-graph-plan-resolution.v1';

const CANONICAL_ROLES = [
  'observation',
  'state',
  'feature',
  'candidate',
  'parameter',
  'reward',
  'context',
];

const STATUS_PRIORITY = {
  RESOLVED: 0,
  DEFERRED: 1,
  AMBIGUOUS: 2,
  REJECTED: 3,
};

function stableValue(value) {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.keys(value).sort().map(key => [key, stableValue(value[key])]),
    );
  }
  return value;
}

function sha256Json(value) {
  return `sha256:${createHash('sha256').update(JSON.stringify(stableValue(value))).digest('hex')}`;
}

function loadJson(baseDir, file) {
  const filePath = path.join(baseDir, file);
  if (!fs.existsSync(filePath)) throw new Error(`operation registry source contract is missing: ${filePath}`);
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function roleValid(role) {
  return CANONICAL_ROLES.includes(role)
    || (typeof role === 'string' && /^x-[a-z0-9][a-z0-9._-]*$/.test(role));
}

function normalizeStatus(statuses) {
  let result = 'RESOLVED';
  for (const status of statuses) {
    if ((STATUS_PRIORITY[status] ?? -1) > STATUS_PRIORITY[result]) result = status;
  }
  return result;
}

function descriptorShapeRule(constructor) {
  if (['relu', 'gelu', 'sigmoid', 'tanh', 'hardSwish', 'leakyRelu', 'prelu', 'hardSigmoid', 'softplus', 'mish', 'softmax', 'logSoftmax', 'batchNorm', 'groupNorm', 'instanceNorm', 'layerNorm', 'rmsNorm', 'shiftUp', 'shiftDown', 'shiftLeft', 'shiftRight', 'featureNorm'].includes(constructor)) {
    return 'preserve_input';
  }
  if (['add', 'sub', 'mul'].includes(constructor)) return 'binary_exact_shape_match';
  if (constructor === 'matmul') return 'batched_matmul_last_two_axes';
  if (constructor === 'concat') return 'concat_parameter_axis';
  if (constructor === 'linear') return 'linear_feature_axis1_singleton';
  return 'runtime_deferred';
}

function descriptorAxisContract(constructor, parameters, inputLayoutDefinition, outputLayoutDefinition) {
  const dim = parameters.find(parameter => parameter.name === 'dim');
  if (dim) {
    return {
      mode: 'parameter',
      parameter: 'dim',
      domain: {minimum: 0, maximum: 3},
      source: 'rank4_tensor_bridge',
    };
  }
  if (constructor === 'matmul') {
    return {
      mode: 'fixed_semantics',
      semantics: 'batched_matmul_over_last_two_axes',
      source: 'multi-input-plan-explain.v1',
    };
  }
  if (constructor === 'featureNorm') {
    return {
      mode: 'fixed_semantics',
      semantics: 'l2_normalize_feature_axis_1',
      axis: 1,
      source: 'agent-layout-contracts.v1',
    };
  }
  return {
    mode: 'layout_defined_or_none',
    input_axes: Object.fromEntries(Object.entries(inputLayoutDefinition ?? {}).filter(([key]) => key.endsWith('_axis') || key === 'singleton_axis')),
    output_axes: Object.fromEntries(Object.entries(outputLayoutDefinition ?? {}).filter(([key]) => key.endsWith('_axis') || key === 'singleton_axis')),
  };
}

function layoutCompatibility(actualLayout, expectedLayout, layoutContract) {
  if (!actualLayout || actualLayout === 'unknown') return 'deferred';
  if (expectedLayout === 'any_rank4') return 'compatible';
  if (expectedLayout === 'dynamic') return 'deferred';
  if (actualLayout === 'dynamic' || actualLayout === 'preserve_input' || actualLayout === 'any_rank4') {
    return actualLayout === expectedLayout ? 'compatible' : 'deferred';
  }
  if (actualLayout === expectedLayout) return 'compatible';
  const compatibility = layoutContract.compatibility?.[actualLayout];
  if (compatibility?.accepts_consumers?.includes(expectedLayout)) return 'compatible';
  if (compatibility?.conditional_consumers?.includes(expectedLayout)) return 'deferred';
  return 'incompatible';
}

function shapeCompatibleWithLayout(shape, layout) {
  if (!Array.isArray(shape) || shape.length !== 4
    || shape.some(value => !Number.isInteger(value) || value < 1 || value > 0xffffffff)) {
    return 'incompatible';
  }
  if (!layout || layout === 'unknown' || layout === 'dynamic') return 'deferred';
  if (['feature_axis1_singleton', 'token_ids_axis1_singleton'].includes(layout)) {
    return shape[2] === 1 && shape[3] === 1 ? 'compatible' : 'incompatible';
  }
  if (['channel_first_singleton_width', 'sequence_feature_axis2_singleton_width'].includes(layout)) {
    return shape[3] === 1 ? 'compatible' : 'incompatible';
  }
  return 'compatible';
}

function outputLayout(operation, inputValues) {
  const declared = operation.output.layout;
  if (declared === 'preserve_input') return inputValues[0]?.layout ?? 'unknown';
  if (declared === 'dynamic') return 'unknown';
  return declared;
}

function validateParameterType(value, type) {
  const normalized = type.replace(/^Option<(.+)>$/, '$1');
  if (value === undefined) return true;
  if (normalized === 'u32') return Number.isInteger(value) && value >= 0 && value <= 0xffffffff;
  if (normalized === 'f64') return typeof value === 'number' && Number.isFinite(value);
  if (normalized === 'bool') return typeof value === 'boolean';
  if (normalized === 'string') return typeof value === 'string';
  return null;
}

function propagateShape(operation, inputValues, parameters) {
  const rule = operation.shape_rule;
  const first = inputValues[0]?.shape ?? null;
  const second = inputValues[1]?.shape ?? null;
  if (!first) return {status: 'DEFERRED', shape: null, reason: 'input shape is not statically known'};

  if (rule === 'preserve_input') return {status: 'RESOLVED', shape: [...first]};

  if (rule === 'binary_exact_shape_match') {
    if (!second) return {status: 'DEFERRED', shape: null, reason: 'second input shape is not statically known'};
    if (first.some((value, index) => value !== second[index])) {
      return {status: 'REJECTED', shape: null, code: 'SHAPE_INCOMPATIBLE', reason: 'binary elementwise inputs require exact shape equality'};
    }
    return {status: 'RESOLVED', shape: [...first]};
  }

  if (rule === 'batched_matmul_last_two_axes') {
    if (!second) return {status: 'DEFERRED', shape: null, reason: 'second input shape is not statically known'};
    if (first[0] !== second[0] || first[1] !== second[1] || first[3] !== second[2]) {
      return {status: 'REJECTED', shape: null, code: 'SHAPE_INCOMPATIBLE', reason: 'batched matmul requires matching batch/channel axes and contracted dimension'};
    }
    return {status: 'RESOLVED', shape: [first[0], first[1], first[2], second[3]]};
  }

  if (rule === 'concat_parameter_axis') {
    if (!second) return {status: 'DEFERRED', shape: null, reason: 'second input shape is not statically known'};
    const axis = parameters.dim;
    if (!Number.isInteger(axis) || axis < 0 || axis > 3) {
      return {status: 'REJECTED', shape: null, code: 'AXIS_OUT_OF_RANGE', reason: 'concat dim must be an integer in rank-4 axis range 0..3'};
    }
    for (let index = 0; index < 4; index++) {
      if (index !== axis && first[index] !== second[index]) {
        return {status: 'REJECTED', shape: null, code: 'SHAPE_INCOMPATIBLE', reason: 'concat non-axis dimensions must match'};
      }
    }
    const extent = first[axis] + second[axis];
    if (!Number.isSafeInteger(extent) || extent > 0xffffffff) {
      return {status: 'REJECTED', shape: null, code: 'CAPACITY_EXCEEDED', reason: 'concat output extent exceeds u32 rank-4 bridge capacity'};
    }
    const shape = [...first];
    shape[axis] = extent;
    return {status: 'RESOLVED', shape};
  }

  if (rule === 'linear_feature_axis1_singleton') {
    const inDim = parameters.in_dim;
    const outDim = parameters.out_dim;
    if (!Number.isInteger(inDim) || !Number.isInteger(outDim)) {
      return {status: 'AMBIGUOUS', shape: null, reason: 'linear in_dim and out_dim must be provided'};
    }
    if (first[1] !== inDim || first[2] !== 1 || first[3] !== 1) {
      return {status: 'REJECTED', shape: null, code: 'SHAPE_INCOMPATIBLE', reason: 'linear input must be [B,in_dim,1,1]'};
    }
    if (outDim < 1 || outDim > 0xffffffff) {
      return {status: 'REJECTED', shape: null, code: 'INVALID_PARAMETER', reason: 'linear out_dim must be within 1..=u32::MAX'};
    }
    return {status: 'RESOLVED', shape: [first[0], outDim, 1, 1]};
  }

  return {status: 'DEFERRED', shape: null, reason: 'shape rule remains authoritative at canonical graph/Burn runtime'};
}

export function loadOperationContractRegistry(baseDir = MODULE_DIR) {
  const layerCatalog = loadJson(baseDir, 'agent-layer-catalog.v1.json');
  const layoutContract = loadJson(baseDir, 'agent-layout-contracts.v1.json');
  const inputPort = loadJson(baseDir, 'agent-input-port.v1.json');
  const faultContract = loadJson(baseDir, 'agent-fault-contract.v1.json');
  const graphPlan = loadJson(baseDir, 'multi-input-graph-plan.v1.json');
  const planExplain = loadJson(baseDir, 'multi-input-plan-explain.v1.json');

  const operations = Object.entries(layerCatalog.constructors).map(([constructor, source]) => {
    const layout = layoutContract.constructors?.[constructor] ?? {input: 'dynamic', output: 'dynamic'};
    const inputLayoutDefinition = layoutContract.layouts?.[layout.input] ?? {};
    const outputLayoutDefinition = layoutContract.layouts?.[layout.output] ?? {};
    const arity = source.arity;
    const positions = arity === 2 ? ['left', 'right'] : ['input'];
    const parameters = source.parameters.map(parameter => ({...parameter}));
    const shapeRule = descriptorShapeRule(constructor);
    const failureClasses = [
      'INVALID_PARAMETER',
      'INPUT_ARITY_MISMATCH',
      'LAYOUT_INCOMPATIBLE',
      'SHAPE_INCOMPATIBLE',
      'CAPACITY_EXCEEDED',
      'REGISTRY_BINDING_DEFERRED',
      'RUNTIME_OPERATOR_DEFERRED',
    ];

    return {
      operation_id: `graph.layer.${constructor}`,
      namespace: 'graph.layer',
      constructor,
      family: source.family,
      arity,
      constructor_signature: source.signature,
      constructor_fallible: source.fallible,
      parameters,
      inputs: positions.map(position => ({
        position,
        kind: 'tensor',
        dtype: 'f32',
        rank: 4,
        layout: layout.input,
        layout_axes: inputLayoutDefinition,
        semantic_role_policy: 'caller_declared',
        allowed_roles: [...CANONICAL_ROLES, 'x-*'],
      })),
      output: {
        kind: 'tensor',
        dtype: 'f32',
        rank: 4,
        layout: layout.output,
        layout_axes: outputLayoutDefinition,
        semantic_role_policy: 'caller_declared',
        allowed_roles: [...CANONICAL_ROLES, 'x-*'],
      },
      axis_contract: descriptorAxisContract(constructor, parameters, inputLayoutDefinition, outputLayoutDefinition),
      shape_rule: shapeRule,
      capacity: {
        graph_slots_max: graphPlan.invariants.graph_slots_max,
        tensor_rank: graphPlan.invariants.tensor_rank,
        tensor_dtype: graphPlan.invariants.tensor_dtype,
      },
      failure_classes: failureClasses,
      remaining_preflight: [
        `AgentLayerSpec.${constructor} canonical constructor validation`,
        'LayerRegistry initialization or exact identity binding',
        'CompiledMultiInputGraph.preflight when using runtime tensors',
        'Burn numerical/operator execution',
      ],
      sources: {
        constructor: 'agent-layer-catalog.v1.json',
        layout: 'agent-layout-contracts.v1.json',
        semantic_roles: 'agent-input-port.v1.json',
        static_shape: 'multi-input-plan-explain.v1.json',
        graph_capacity: 'multi-input-graph-plan.v1.json',
      },
    };
  }).sort((left, right) => left.operation_id.localeCompare(right.operation_id));

  const registry = {
    schema: REGISTRY_SCHEMA,
    version: 1,
    role: 'read_only_constructible_operation_pool',
    execution_authorized: false,
    mutation: 'none',
    source_contracts: {
      layer_catalog: layerCatalog.schema_id,
      layout_contract: layoutContract.schema_id,
      semantic_input_port: inputPort.schema_id,
      fault_contract: faultContract.schema_id,
      multi_input_graph: graphPlan.schema_id,
      plan_explain: planExplain.schema_id,
    },
    canonical_roles: [...CANONICAL_ROLES],
    capacity: {
      graph_slots_max: graphPlan.invariants.graph_slots_max,
      operation_type_count: operations.length,
    },
    pool_kind: 'constructible_operation_types_not_live_registry_instances',
    normalized_failure_classes: [
      'OPERATION_NOT_FOUND',
      'OPERATION_SELECTION_AMBIGUOUS',
      'INPUT_ARITY_MISMATCH',
      'DEPENDENCY_UNRESOLVED',
      'INVALID_PARAMETER',
      'AXIS_OUT_OF_RANGE',
      'SEMANTIC_ROLE_UNRESOLVED',
      'LAYOUT_INCOMPATIBLE',
      'SHAPE_INCOMPATIBLE',
      'CAPACITY_EXCEEDED',
      'REGISTRY_BINDING_DEFERRED',
      'RUNTIME_OPERATOR_DEFERRED',
    ],
    operations,
  };
  registry.registry_fingerprint = sha256Json(registry);
  return registry;
}

export function queryOperationPool(registry, query = {}) {
  if (!registry || registry.schema !== REGISTRY_SCHEMA || !Array.isArray(registry.operations)) {
    throw new Error('queryOperationPool requires burn-research.operation-contract-registry.v1');
  }
  if (query.semantic_role !== undefined && !roleValid(query.semantic_role)) {
    return {
      schema: POOL_SCHEMA,
      status: 'REJECTED',
      execution_authorized: false,
      mutation: 'none',
      candidate_count: 0,
      candidates: [],
      diagnostics: [{code: 'SEMANTIC_ROLE_UNRESOLVED', class: 'semantic', message: `invalid semantic role ${query.semantic_role}`}],
    };
  }

  const candidates = [];
  for (const operation of registry.operations) {
    if (query.operation_id !== undefined && operation.operation_id !== query.operation_id) continue;
    if (query.constructor !== undefined && operation.constructor !== query.constructor) continue;
    if (query.family !== undefined && operation.family !== query.family) continue;
    if (query.arity !== undefined && operation.arity !== query.arity) continue;

    let compatibility = 'compatible';
    if (query.input_layout !== undefined) {
      const relation = layoutCompatibility(query.input_layout, operation.inputs[0].layout, {
        compatibility: query.__layout_compatibility ?? {},
      });
      compatibility = relation;
    }
    if (query.output_layout !== undefined) {
      const actual = operation.output.layout;
      if (actual === 'dynamic' || actual === 'preserve_input') compatibility = compatibility === 'incompatible' ? compatibility : 'deferred';
      else if (actual !== query.output_layout) compatibility = 'incompatible';
    }
    if (compatibility === 'incompatible') continue;

    candidates.push({
      operation_id: operation.operation_id,
      constructor: operation.constructor,
      family: operation.family,
      arity: operation.arity,
      input_layout: operation.inputs[0].layout,
      output_layout: operation.output.layout,
      compatibility,
      axis_contract: operation.axis_contract,
      shape_rule: operation.shape_rule,
    });
  }

  let status;
  if (candidates.length === 0) status = 'REJECTED';
  else if (candidates.length > 1) status = 'AMBIGUOUS';
  else status = candidates[0].compatibility === 'deferred' ? 'DEFERRED' : 'RESOLVED';

  return {
    schema: POOL_SCHEMA,
    status,
    execution_authorized: false,
    mutation: 'none',
    registry_fingerprint: registry.registry_fingerprint,
    candidate_count: candidates.length,
    candidates,
    diagnostics: candidates.length === 0
      ? [{code: 'OPERATION_NOT_FOUND', class: 'selection', message: 'no operation satisfies the supplied pool query'}]
      : [],
  };
}

function diagnostic(code, className, message, stepId = null) {
  return {code, class: className, step_id: stepId, message};
}

export function resolveDeclarativeGraphPlan(registry, plan) {
  if (!registry || registry.schema !== REGISTRY_SCHEMA) {
    throw new Error('resolveDeclarativeGraphPlan requires burn-research.operation-contract-registry.v1');
  }

  const diagnostics = [];
  const statuses = [];
  const stepResults = [];
  const values = new Map();
  const graphSlotsMax = registry.capacity.graph_slots_max;

  if (!plan || typeof plan !== 'object' || Array.isArray(plan)) {
    return {
      schema: RESOLUTION_SCHEMA,
      status: 'REJECTED',
      execution_authorized: false,
      mutation: 'none',
      diagnostics: [diagnostic('INVALID_PARAMETER', 'plan', 'plan must be an object')],
      step_results: [],
    };
  }
  if (plan.schema !== PLAN_SCHEMA) {
    diagnostics.push(diagnostic('INVALID_PARAMETER', 'plan', `plan.schema must be ${PLAN_SCHEMA}`));
    statuses.push('REJECTED');
  }
  if (!Array.isArray(plan.inputs) || plan.inputs.length === 0 || !Array.isArray(plan.steps)) {
    diagnostics.push(diagnostic('INVALID_PARAMETER', 'plan', 'plan.inputs must be a non-empty array and plan.steps must be an array'));
    statuses.push('REJECTED');
  }

  for (const input of Array.isArray(plan.inputs) ? plan.inputs : []) {
    if (!input || typeof input.id !== 'string' || input.id.length === 0 || values.has(input.id)) {
      diagnostics.push(diagnostic('INVALID_PARAMETER', 'input', 'input id must be a unique non-empty string'));
      statuses.push('REJECTED');
      continue;
    }
    if (!roleValid(input.role)) {
      diagnostics.push(diagnostic('SEMANTIC_ROLE_UNRESOLVED', 'semantic', `input ${input.id} has invalid role ${input.role ?? '<missing>'}`));
      statuses.push(input.role === undefined ? 'AMBIGUOUS' : 'REJECTED');
    }
    const shapeStatus = shapeCompatibleWithLayout(input.shape, input.layout);
    if (shapeStatus === 'incompatible') {
      diagnostics.push(diagnostic('SHAPE_INCOMPATIBLE', 'shape', `input ${input.id} shape is invalid for rank-4 layout ${input.layout}`));
      statuses.push('REJECTED');
    } else if (shapeStatus === 'deferred') {
      statuses.push('DEFERRED');
    }
    values.set(input.id, {
      shape: Array.isArray(input.shape) && input.shape.length === 4 ? [...input.shape] : null,
      layout: input.layout ?? 'unknown',
      role: input.role ?? null,
      producer: 'external_input',
    });
  }

  for (const step of Array.isArray(plan.steps) ? plan.steps : []) {
    const stepStatuses = [];
    const stepDiagnostics = [];
    const stepId = typeof step?.id === 'string' ? step.id : null;
    const inputIds = Array.isArray(step?.inputs) ? step.inputs : [];
    const inputValues = inputIds.map(id => values.get(id));

    if (!stepId) {
      stepDiagnostics.push(diagnostic('INVALID_PARAMETER', 'step', 'step id must be a non-empty string'));
      stepStatuses.push('REJECTED');
    }
    if (![1, 2].includes(inputIds.length)) {
      stepDiagnostics.push(diagnostic('INPUT_ARITY_MISMATCH', 'graph', 'each v1 declarative graph step must have one or two inputs', stepId));
      stepStatuses.push('REJECTED');
    }
    for (let index = 0; index < inputIds.length; index++) {
      if (!inputValues[index]) {
        stepDiagnostics.push(diagnostic('DEPENDENCY_UNRESOLVED', 'graph', `step input ${inputIds[index]} is not a declared input or earlier step output`, stepId));
        stepStatuses.push('REJECTED');
      }
    }

    let candidates = [];
    if (typeof step?.operation === 'string') {
      const operation = registry.operations.find(candidate => candidate.operation_id === step.operation);
      if (!operation) {
        stepDiagnostics.push(diagnostic('OPERATION_NOT_FOUND', 'selection', `unknown operation ${step.operation}`, stepId));
        stepStatuses.push('REJECTED');
      } else candidates = [operation];
    } else if (step?.selector && typeof step.selector === 'object' && !Array.isArray(step.selector)) {
      const pool = registry.operations.filter(operation => {
        if (step.selector.constructor !== undefined && operation.constructor !== step.selector.constructor) return false;
        if (step.selector.family !== undefined && operation.family !== step.selector.family) return false;
        if (step.selector.arity !== undefined && operation.arity !== step.selector.arity) return false;
        if (operation.arity !== inputIds.length) return false;
        return true;
      });
      candidates = pool;
      if (pool.length === 0) {
        stepDiagnostics.push(diagnostic('OPERATION_NOT_FOUND', 'selection', 'step selector matches no operation', stepId));
        stepStatuses.push('REJECTED');
      } else if (pool.length > 1) {
        stepDiagnostics.push(diagnostic('OPERATION_SELECTION_AMBIGUOUS', 'selection', `step selector matches ${pool.length} operations`, stepId));
        stepStatuses.push('AMBIGUOUS');
      }
    } else {
      stepDiagnostics.push(diagnostic('OPERATION_SELECTION_AMBIGUOUS', 'selection', 'step requires exact operation or selector', stepId));
      stepStatuses.push('AMBIGUOUS');
    }

    const selected = candidates.length === 1 ? candidates[0] : null;
    if (selected && selected.arity !== inputIds.length) {
      stepDiagnostics.push(diagnostic('INPUT_ARITY_MISMATCH', 'graph', `operation ${selected.operation_id} requires arity ${selected.arity}`, stepId));
      stepStatuses.push('REJECTED');
    }

    const parameters = step?.parameters && typeof step.parameters === 'object' && !Array.isArray(step.parameters)
      ? step.parameters : {};
    if (selected) {
      const allowedNames = new Set(selected.parameters.map(parameter => parameter.name));
      for (const key of Object.keys(parameters)) {
        if (!allowedNames.has(key)) {
          stepDiagnostics.push(diagnostic('INVALID_PARAMETER', 'parameter', `operation ${selected.operation_id} has no parameter ${key}`, stepId));
          stepStatuses.push('REJECTED');
        }
      }
      for (const parameter of selected.parameters) {
        const present = Object.prototype.hasOwnProperty.call(parameters, parameter.name);
        if (!present && !parameter.optional) {
          stepDiagnostics.push(diagnostic('INVALID_PARAMETER', 'parameter', `required parameter ${parameter.name} is missing`, stepId));
          stepStatuses.push('AMBIGUOUS');
          continue;
        }
        if (!present) continue;
        const typeStatus = validateParameterType(parameters[parameter.name], parameter.type);
        if (typeStatus === false) {
          stepDiagnostics.push(diagnostic('INVALID_PARAMETER', 'parameter', `parameter ${parameter.name} does not satisfy ${parameter.type}`, stepId));
          stepStatuses.push('REJECTED');
        } else if (typeStatus === null) {
          stepDiagnostics.push(diagnostic('INVALID_PARAMETER', 'parameter', `parameter type ${parameter.type} is not modeled by the v1 planner`, stepId));
          stepStatuses.push('DEFERRED');
        }
      }
      if (selected.axis_contract.mode === 'parameter') {
        const axis = parameters[selected.axis_contract.parameter];
        if (axis !== undefined && (!Number.isInteger(axis)
          || axis < selected.axis_contract.domain.minimum
          || axis > selected.axis_contract.domain.maximum)) {
          stepDiagnostics.push(diagnostic('AXIS_OUT_OF_RANGE', 'axis', `axis parameter ${selected.axis_contract.parameter} must be ${selected.axis_contract.domain.minimum}..${selected.axis_contract.domain.maximum}`, stepId));
          stepStatuses.push('REJECTED');
        }
      }

      for (let index = 0; index < inputValues.length; index++) {
        const value = inputValues[index];
        if (!value) continue;
        const expectedLayout = selected.inputs[Math.min(index, selected.inputs.length - 1)].layout;
        const relation = layoutCompatibility(value.layout, expectedLayout, {
          compatibility: {},
        });
        if (relation === 'incompatible') {
          stepDiagnostics.push(diagnostic('LAYOUT_INCOMPATIBLE', 'layout', `input ${inputIds[index]} layout ${value.layout} is incompatible with ${selected.operation_id} expected layout ${expectedLayout}`, stepId));
          stepStatuses.push('REJECTED');
        } else if (relation === 'deferred') {
          stepDiagnostics.push(diagnostic('RUNTIME_OPERATOR_DEFERRED', 'layout', `input ${inputIds[index]} layout compatibility with ${selected.operation_id} is not statically proven`, stepId));
          stepStatuses.push('DEFERRED');
        }
      }
    }

    let outputRole = step?.output_role;
    if (!roleValid(outputRole)) {
      stepDiagnostics.push(diagnostic('SEMANTIC_ROLE_UNRESOLVED', 'semantic', `step output_role must be canonical or x-* extension`, stepId));
      stepStatuses.push(outputRole === undefined ? 'AMBIGUOUS' : 'REJECTED');
      outputRole = null;
    }

    let propagated = {status: 'DEFERRED', shape: null, reason: 'operation selection unresolved'};
    if (selected && inputValues.every(Boolean)) {
      propagated = propagateShape(selected, inputValues, parameters);
      stepStatuses.push(propagated.status);
      if (propagated.status !== 'RESOLVED') {
        stepDiagnostics.push(diagnostic(
          propagated.code ?? 'RUNTIME_OPERATOR_DEFERRED',
          propagated.status === 'REJECTED' ? 'shape' : 'runtime_deferred',
          propagated.reason,
          stepId,
        ));
      }
    }

    const stepStatus = normalizeStatus(stepStatuses.length ? stepStatuses : ['RESOLVED']);
    statuses.push(stepStatus);
    diagnostics.push(...stepDiagnostics);

    const outputId = typeof step?.output === 'string' && step.output.length > 0 ? step.output : null;
    if (!outputId) {
      const item = diagnostic('INVALID_PARAMETER', 'graph', 'step output must be a non-empty string', stepId);
      diagnostics.push(item);
      stepDiagnostics.push(item);
      statuses.push('REJECTED');
    } else if (values.has(outputId)) {
      const item = diagnostic('INVALID_PARAMETER', 'graph', `step output ${outputId} collides with an existing value`, stepId);
      diagnostics.push(item);
      stepDiagnostics.push(item);
      statuses.push('REJECTED');
    } else {
      values.set(outputId, {
        shape: propagated.shape,
        layout: selected ? outputLayout(selected, inputValues) : 'unknown',
        role: outputRole,
        producer: stepId,
      });
    }

    stepResults.push({
      step_id: stepId,
      status: stepStatus,
      selected_operation: selected?.operation_id ?? null,
      candidate_operations: candidates.map(candidate => candidate.operation_id),
      input_values: [...inputIds],
      output_value: outputId,
      output_contract: outputId ? values.get(outputId) ?? null : null,
      diagnostics: stepDiagnostics,
      remaining_preflight: selected?.remaining_preflight ?? [],
    });
  }

  if (values.size > graphSlotsMax) {
    diagnostics.push(diagnostic('CAPACITY_EXCEEDED', 'capacity', `declarative graph exposes ${values.size} distinct values, exceeding graph slot capacity ${graphSlotsMax}`));
    statuses.push('REJECTED');
  }

  let finalOutput = null;
  if (typeof plan.output !== 'string' || plan.output.length === 0) {
    diagnostics.push(diagnostic('DEPENDENCY_UNRESOLVED', 'graph', 'plan.output must name a final value'));
    statuses.push('AMBIGUOUS');
  } else if (!values.has(plan.output)) {
    diagnostics.push(diagnostic('DEPENDENCY_UNRESOLVED', 'graph', `plan.output ${plan.output} is not produced`));
    statuses.push('REJECTED');
  } else {
    finalOutput = {id: plan.output, ...values.get(plan.output)};
  }

  const status = normalizeStatus(statuses.length ? statuses : ['RESOLVED']);
  const resolution = {
    schema: RESOLUTION_SCHEMA,
    status,
    execution_authorized: false,
    mutation: 'none',
    registry_fingerprint: registry.registry_fingerprint,
    plan_schema: PLAN_SCHEMA,
    graph_slots_max: graphSlotsMax,
    value_count: values.size,
    step_results: stepResults,
    final_output: finalOutput,
    diagnostics,
    status_semantics: {
      RESOLVED: 'exact operation identities and modeled static contracts are resolved; authoritative preflight still applies',
      AMBIGUOUS: 'multiple candidates remain or required caller/agent information is missing',
      DEFERRED: 'operation identity is known but canonical constructor, registry, shape/layout, or Burn runtime proof remains necessary',
      REJECTED: 'known contract violation prevents admissible execution planning',
    },
  };
  resolution.resolution_fingerprint = sha256Json({
    registry_fingerprint: registry.registry_fingerprint,
    plan,
    status,
    step_results: stepResults,
    final_output: finalOutput,
  });
  return resolution;
}

export const OPERATION_REGISTRY_SCHEMAS = Object.freeze({
  registry: REGISTRY_SCHEMA,
  pool: POOL_SCHEMA,
  plan: PLAN_SCHEMA,
  resolution: RESOLUTION_SCHEMA,
});
