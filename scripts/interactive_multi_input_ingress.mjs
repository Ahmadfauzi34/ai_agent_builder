import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {createHash} from 'node:crypto';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {manifestDigest, SIGNED_INPUT_CLAIM_SCHEMA_V1, SIGNED_INPUT_CLAIM_SCHEMA_V2, SignedIngressVerifier} from './ingress_provenance.mjs';
import {decodeF32Base64, encodedF32Matches, f32ValueBytes, f32ValueDigest, receiptMatchesOutput} from './ingress_execution_receipt.mjs';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const defaultPackageDir = fs.existsSync(path.join(scriptDir, 'node.mjs')) ? scriptDir : 'pkg';
const packageDir = path.resolve(process.argv[2] ?? defaultPackageDir);
const startupOptions = process.argv.slice(6);
if (startupOptions.length > 3 || new Set(startupOptions).size !== startupOptions.length
  || startupOptions.some(option => !['--allow-state-checkpoint-export', '--allow-checkpoint-restore', '--require-state-bound-inputs'].includes(option))) {
  throw new Error('unknown or duplicate trusted runner startup option');
}
const allowStateCheckpointExport = startupOptions.includes('--allow-state-checkpoint-export');
const allowCheckpointRestore = startupOptions.includes('--allow-checkpoint-restore');
const requireStateBoundInputs = startupOptions.includes('--require-state-bound-inputs');
// A trusted host owns this startup argument; JSON Lines commands cannot replace keys.
if (!process.argv[3] && (process.argv[4] || process.argv[5])) throw new Error('durable ingress requires a host trust policy');
const provenanceVerifier = process.argv[3] ? new SignedIngressVerifier(path.resolve(process.argv[3]), {
  ledgerPath: process.argv[4] ? path.resolve(process.argv[4]) : undefined,
  subject: process.argv[5],
}) : null;
if ((allowStateCheckpointExport || allowCheckpointRestore) && !provenanceVerifier?.ledger) {
  throw new Error('checkpoint export and restore require durable signed ingress');
}
if (requireStateBoundInputs && !provenanceVerifier?.allIssuersSupportStateBoundInputClaims) {
  throw new Error('state-bound input requirement needs every configured issuer to explicitly allow signed claim v2');
}
const {loadBurnRuntime} = await import(pathToFileURL(path.join(packageDir, 'node.mjs')).href);
const wasm = await loadBurnRuntime(packageDir);

// ---- Structured error envelope (complaint #02) ----
// Frozen taxonomy so callers can branch on `code` without regexing `message`.
// `error` in the JSON response stays a plain human string for backward
// compatibility; the machine-readable envelope rides alongside as
// `error_envelope: {code, phase, message, execution_started, ...details}`.
// `execution_started` is false for every pre-execution failure; the run/trace
// path flips it once the WASM call begins.
const ERROR_CODES = Object.freeze({
  unknown_operation: 'op not recognized',
  session_required: 'no active graph session',
  invalid_argument: 'malformed command argument',
  unknown_layer_constructor: 'layer constructor not in capabilities',
  missing_layer: 'step references a layer index that was not created',
  invalid_step: 'step is neither a valid unary nor binary step',
  invalid_port_shape: 'port.shape must have four dimensions',
  missing_required_field: 'a required field or port mapping is absent',
  missing_logical_port: 'slot has no logical port mapping',
  provenance_preflight_failed: 'host provenance gate rejected execution',
  policy_violation: 'host startup policy forbids this operation',
  source_mismatch: 'input source differs from the logical port expectation',
  invalid_handoff: 'state handoff preconditions not met',
  handoff_payload_mismatch: 'handoff bytes differ from the parent receipt output',
  stale_claim_rebind: 'signed claim cannot rebind an unchanged input',
  stale_state_claim: 'signed claim checkpoint digest is stale',
  restore_identity_mismatch: 'restored graph identity differs from receipt',
  restore_manifest_mismatch: 'restored manifest differs from receipt',
  ledger_required: 'durable host ledger required',
  trace_failed: 'execution trace faulted before completion',
  preflight_failed: 'ingress preflight rejected execution',
  digest_mismatch: 'host recomputation digest differs',
  internal_error: 'unclassified failure (often from the WASM layer)',
  // Complaint #07: vocabulary errors name the actual value and the valid choices.
  invalid_enum: 'value is not in the supported vocabulary',
  // Complaint #08: the two multi-input plan invariants fail with different codes.
  plan_minimum_ports: 'multi-input plan declares fewer ports than the minimum',
  plan_unused_ports: 'multi-input plan declares ports the graph never consumes',
});

class HostError extends Error {
  constructor(code, phase, message, details = {}) {
    super(message);
    this.name = 'HostError';
    this.code = code;
    this.phase = phase;
    this.details = details;
    this.executionStarted = details.execution_started ?? false;
  }
}

function fail(code, phase, message, details = {}) {
  if (!ERROR_CODES[code]) throw new Error(`unknown error code: ${code}`);
  return new HostError(code, phase, message, details);
}

// Complaint #07: role/layout vocabulary, mirrored from the WASM core
// (src/input_port.rs CANONICAL_ROLES/role_valid, src/contracts.rs LayoutTag).
// The host validates enum MEMBERSHIP only; shape-vs-layout constraints stay
// authoritative in the WASM, so the host can never reject what the WASM accepts.
const HOST_VOCABULARY_VERSION = 'burn-research.host-vocabulary.v1';
const CANONICAL_INPUT_ROLES = Object.freeze([
  'observation', 'state', 'feature', 'candidate', 'parameter', 'reward', 'context']);
const ROLE_EXTENSION_NAMESPACE = Object.freeze({
  prefix: 'x-',
  pattern: '^x-[a-z0-9\\-_.]+$',
  max_bytes: 64,
  description: 'extension roles must start with "x-" followed by lowercase alphanumerics, "-", "_" or "."',
});
const EXTERNAL_LAYOUTS = Object.freeze([
  {name: 'unknown', shape_constraint: 'none; any rank-4 shape with all dimensions > 0',
   description: 'no layout assertion on the external tensor'},
  {name: 'any_rank4', shape_constraint: 'none; any rank-4 shape with all dimensions > 0',
   description: 'any rank-4 external tensor'},
  {name: 'feature_axis1_singleton', shape_constraint: 'shape[2] == 1 and shape[3] == 1',
   description: 'feature vector on axis 1, e.g. shape [N, C, 1, 1]'},
  {name: 'channel_first', shape_constraint: 'none; any rank-4 shape with all dimensions > 0',
   description: 'channel-first semantics (NCHW); not enforced beyond rank-4'},
  {name: 'channel_first_singleton_width', shape_constraint: 'shape[3] == 1',
   description: 'channel-first with singleton width, e.g. shape [N, C, H, 1]'},
  {name: 'feature_last', shape_constraint: 'none; any rank-4 shape with all dimensions > 0',
   description: 'feature-last semantics; not enforced beyond rank-4'},
  {name: 'token_ids_axis1_singleton', shape_constraint: 'shape[2] == 1 and shape[3] == 1',
   description: 'token ids on axis 1, e.g. shape [N, T, 1, 1]'},
  {name: 'sequence_feature_axis2_singleton_width', shape_constraint: 'shape[3] == 1',
   description: 'sequence/feature tensor, e.g. shape [N, S, F, 1]'},
]);
const REJECTED_EXTERNAL_LAYOUTS = Object.freeze([
  {name: 'preserve_input', reason: 'relational/internal layout; cannot describe an external tensor'},
  {name: 'dynamic', reason: 'relational/internal layout; cannot describe an external tensor'},
]);
// Complaint #08: multi-input plan invariants (mirrors
// MultiInputGraphPlan::validate_for_compile and the read-before-write check).
const MINIMUM_PLAN_PORTS = 2;

function roleValid(role) {
  if (typeof role !== 'string' || role.length === 0 || Buffer.byteLength(role) > ROLE_EXTENSION_NAMESPACE.max_bytes)
    return false;
  if (CANONICAL_INPUT_ROLES.includes(role)) return true;
  return role.startsWith(ROLE_EXTENSION_NAMESPACE.prefix)
    && Buffer.byteLength(role) > ROLE_EXTENSION_NAMESPACE.prefix.length
    && /^[a-z0-9\-_.]+$/.test(role.slice(ROLE_EXTENSION_NAMESPACE.prefix.length));
}

function externalLayoutValid(layout) {
  return typeof layout === 'string' && EXTERNAL_LAYOUTS.some(entry => entry.name === layout);
}

function vocabularyDocument() {
  const term = (canonical_name, extra) => ({canonical_name, aliases: [],
    since_version: HOST_VOCABULARY_VERSION, ...extra});
  return {
    schema: HOST_VOCABULARY_VERSION,
    since_version_semantics: 'the vocabulary-document version in which the term was first documented; '
      + 'the underlying WASM vocabulary predates host documentation',
    input_roles: CANONICAL_INPUT_ROLES.map(name => term(name, {
      constraints: `non-empty, at most ${ROLE_EXTENSION_NAMESPACE.max_bytes} bytes`,
    })),
    role_extension_namespace: ROLE_EXTENSION_NAMESPACE,
    layouts: EXTERNAL_LAYOUTS.map(entry => term(entry.name, {
      constraints: entry.shape_constraint,
      description: entry.description,
    })),
    layouts_rejected_for_external_declaration: REJECTED_EXTERNAL_LAYOUTS,
  };
}

function assertValidRole(role, phase, extra = {}) {
  if (roleValid(role)) return;
  throw fail('invalid_enum', phase, `unsupported input role ${JSON.stringify(role)}`, {
    field: 'role', actual_value: role, allowed_values: [...CANONICAL_INPUT_ROLES],
    extension_namespace: ROLE_EXTENSION_NAMESPACE,
    capability_ref: 'capabilities.vocabulary',
    remediation_hint: 'use a canonical role from capabilities.vocabulary.input_roles or an x- extension role',
    ...extra});
}

function assertValidLayout(layout, phase, extra = {}) {
  if (externalLayoutValid(layout)) return;
  const rejected = REJECTED_EXTERNAL_LAYOUTS.find(entry => entry.name === layout);
  throw fail('invalid_enum', phase, `unsupported external layout ${JSON.stringify(layout)}`, {
    field: 'layout', actual_value: layout,
    allowed_values: EXTERNAL_LAYOUTS.map(entry => entry.name),
    ...(rejected ? {rejection_reason: rejected.reason} : {}),
    capability_ref: 'capabilities.vocabulary',
    remediation_hint: 'use a layout from capabilities.vocabulary.layouts; relational layouts cannot describe external tensors',
    ...extra});
}

// Complaint #08: host-side port-consumption analysis replicating the WASM
// read-before-write rule. A declared slot counts as consumed when a graph step
// reads it before any step writes it, or when it is the output slot and no
// step writes it. The WASM stays authoritative for deeper ordering issues.
function analyzePlanPorts(numSlots, steps, ports, outputSlot) {
  const declared = requireArray(ports, 'ports').map(port => port.slot);
  const declaredSet = new Set(declared);
  const written = new Set();
  const consumed = new Set();
  for (const step of requireArray(steps, 'steps')) {
    const slots = requireArray(step.slots, 'step.slots');
    const inputs = step.kind === 'binary' ? slots.slice(0, 2) : slots.slice(0, 1);
    const out = step.kind === 'binary' ? slots[2] : slots[1];
    for (const slot of inputs) {
      if (!written.has(slot) && declaredSet.has(slot)) consumed.add(slot);
    }
    if (out !== undefined) written.add(out);
  }
  if (!written.has(outputSlot) && declaredSet.has(outputSlot)) consumed.add(outputSlot);
  const bySlot = (a, b) => a - b;
  const consumedPorts = [...consumed].sort(bySlot);
  const unusedPorts = [...declared].filter(slot => !consumed.has(slot)).sort(bySlot);
  return {
    minimum_ports: MINIMUM_PLAN_PORTS,
    declared_ports: [...declared].sort(bySlot),
    consumed_ports: consumedPorts,
    unused_ports: unusedPorts,
  };
}

function planConstraintsDocument() {
  return {
    minimum_ports: MINIMUM_PLAN_PORTS,
    minimum_ports_code: 'plan_minimum_ports',
    all_declared_ports_must_be_consumed: true,
    unused_port_code: 'plan_unused_ports',
    consumed_definition: 'a declared slot counts as consumed when a graph step reads it before any '
      + 'step writes it, or when it is the output slot and no step writes it',
    declared_ports_are_external_inputs: true,
    validator: 'op validatePlan reports declared_ports, consumed_ports, unused_ports and minimum_ports without creating a session',
  };
}

// Complaint #06: canonical layer-type identity, mirrored from src/protocol.rs
// (LAYER_* type codes) and src/agent.rs (AgentLayerSpec constructors). The WASM
// core spells the type code as decimal in explain/trace steps (`layer_type`) and
// as two lowercase hex digits in init fingerprints (`type=13`); every surface
// below resolves both spellings to the same canonical name, and the legacy
// spellings stay as compatibility fields.
const LAYER_TYPE_NAMES = Object.freeze({
  1: 'linear', 2: 'norm', 3: 'conv', 4: 'activation', 5: 'embedding', 6: 'pool',
  16: 'shift', 17: 'ghost', 18: 'seblock', 19: 'binary', 20: 'feature_norm'});
const LAYER_TYPE_CONSTRUCTORS = Object.freeze({
  1: ['linear'],
  2: ['batchNorm', 'groupNorm', 'instanceNorm', 'layerNorm', 'rmsNorm'],
  3: ['conv1d', 'conv2d', 'convTranspose2d'],
  4: ['relu', 'gelu', 'sigmoid', 'tanh', 'hardSwish', 'leakyRelu', 'prelu',
      'swiGlu', 'hardSigmoid', 'softplus', 'mish', 'softmax', 'logSoftmax', 'glu'],
  5: ['embedding'],
  6: ['maxPool1d', 'maxPool2d', 'avgPool1d', 'avgPool2d', 'adaptiveAvgPool2d'],
  16: ['shiftUp', 'shiftDown', 'shiftLeft', 'shiftRight'],
  17: ['ghost'],
  18: ['seBlock'],
  19: ['add', 'sub', 'mul', 'matmul', 'concat'],
  20: ['featureNorm']});
const LAYER_TYPE_CODES = Object.freeze(Object.keys(LAYER_TYPE_NAMES).map(Number).sort((a, b) => a - b));

// Canonical identity for one layer-type code as spelled on a given surface.
// `encoding` is 'decimal' (explain/trace `layer_type`) or 'hex' (init
// fingerprint `type=..`); the code is always reported decimal.
function layerTypeIdentity(code, encoding) {
  const numeric = Number(code);
  return {layer_type_code: numeric,
    layer_type_name: LAYER_TYPE_NAMES[numeric] ?? 'unknown',
    layer_type_encoding: encoding};
}

// Attach the canonical identity next to a WASM step's decimal `layer_type`.
// The original `layer_type` field is kept for compatibility.
function annotateStepLayerIdentity(step) {
  if (!step || typeof step !== 'object' || !Number.isInteger(step.layer_type)) return step;
  return {...step, ...layerTypeIdentity(step.layer_type, 'decimal')};
}

// Parse the Rust init-fingerprint format `type=%02x;id=%d;...` (see
// registry.rs LayerInitIdentity::fingerprint) and attach the canonical
// identity next to each fingerprint string.
function annotateProgramIdentity(programIdentity) {
  if (!programIdentity || typeof programIdentity !== 'object') return programIdentity;
  const fingerprints = programIdentity.layer_init_fingerprints;
  if (!Array.isArray(fingerprints)) return programIdentity;
  return {...programIdentity,
    layer_type_identities: fingerprints.map(fingerprint => {
      const match = /^type=([0-9a-fA-F]{2});id=(\d+);/.exec(fingerprint ?? '');
      const code = match ? parseInt(match[1], 16) : NaN;
      return {fingerprint, layer_id: match ? Number(match[2]) : null,
        ...layerTypeIdentity(code, 'hex')};
    })};
}

function layerTypeCatalog() {
  const signatures = JSON.parse(wasm.agentCapabilities()).agent_facade.constructor_signatures ?? {};
  return {
    schema: 'burn-research.host-layer-type-catalog.v1',
    source: 'mirrored from src/protocol.rs LAYER_* constants and src/agent.rs AgentLayerSpec constructors; '
      + 'the WASM core remains authoritative for unknown codes',
    types: LAYER_TYPE_CODES.map(code => {
      const constructors = LAYER_TYPE_CONSTRUCTORS[code];
      const parameterSchemas = Object.fromEntries(constructors
        .filter(name => typeof signatures[name] === 'string')
        .map(name => [name, signatures[name]]));
      return {code, code_hex: code.toString(16).padStart(2, '0'),
        name: LAYER_TYPE_NAMES[code], constructors,
        encodings: {decimal: 'explain/trace step layer_type; canonical code spelling',
          hex: 'init fingerprint type=%02x; zero-padded lowercase hex'},
        ...(Object.keys(parameterSchemas).length ? {parameter_schemas: parameterSchemas} : {})};
    }),
  };
}

// Complaint #06 + #10 presentation for plan explanations: every step carries
// the canonical layer-type identity next to its decimal `layer_type`, init
// fingerprints resolve to the same names, and the misleading hardcoded
// `execution_authorized` is removed (the execution gate is `ready`).
function presentPlanExplanation(explanation) {
  const {execution_authorized: _removed, steps, program_identity, ...rest} = explanation ?? {};
  return {...rest,
    ...(Array.isArray(steps) ? {steps: steps.map(annotateStepLayerIdentity)} : {}),
    ...(program_identity ? {program_identity: annotateProgramIdentity(program_identity)} : {}),
    execution_state_semantics: EXECUTION_STATE_SEMANTICS};
}

// Complaint #06 + #10 presentation for execution traces: same treatment as
// plan explanations; the WASM detail structure is otherwise preserved.
function presentExecutionTrace(executionTrace) {
  if (!executionTrace || typeof executionTrace !== 'object') return executionTrace;
  const {execution_authorized: _removed, steps, program_identity, ...rest} = executionTrace;
  return {...rest,
    ...(Array.isArray(steps) ? {steps: steps.map(annotateStepLayerIdentity)} : {}),
    ...(program_identity ? {program_identity: annotateProgramIdentity(program_identity)} : {})};
}

function assessPlanPorts(command, phase) {
  const analysis = analyzePlanPorts(command.numSlots, command.steps, command.ports, command.outputSlot);
  if (analysis.declared_ports.length < analysis.minimum_ports)
    throw fail('plan_minimum_ports', phase,
      `multi-input plan declares ${analysis.declared_ports.length} external port(s); minimum is ${analysis.minimum_ports}`,
      {...analysis, capability_ref: 'capabilities.plan_constraints',
       remediation_hint: 'declare at least two external input ports in create.ports'});
  if (analysis.unused_ports.length > 0)
    throw fail('plan_unused_ports', phase,
      `multi-input plan declares port(s) the graph never consumes: ${analysis.unused_ports.join(', ')}`,
      {...analysis, capability_ref: 'capabilities.plan_constraints',
       remediation_hint: 'every declared port must be read by a graph step before any write, or be the unwritten output slot'});
  return analysis;
}

// Complaint #09: uniform verification summary across all verify outcomes.
// verdict is 'pass' | 'fail'; null metrics always carry a reason. The WASM
// detail structure (result.reference.verification) is preserved unchanged.
function summarizeVerification(verification) {
  if (!verification || typeof verification.passed !== 'boolean') {
    return {verdict: 'fail', reason: 'no_verification_detail', shape_matches: null,
      compared_elements: null, max_abs_error: null, max_rel_error: null, first_failure: null};
  }
  return {
    verdict: verification.passed ? 'pass' : 'fail',
    reason: verification.passed ? null : 'tolerance_exceeded',
    shape_matches: true,
    compared_elements: verification.len ?? null,
    max_abs_error: verification.max_abs_error ?? null,
    max_rel_error: verification.max_rel_error ?? null,
    first_failure: verification.first_failure ?? null,
  };
}

// A candidate/output length mismatch is a verification verdict
// (reason shape_mismatch), not an op failure. Any other WASM error rethrows.
function shapeMismatchSummary(error, candidateElements) {
  const message = String(error?.message ?? error ?? '');
  if (!/length mismatch/i.test(message)) return null;
  const lengths = message.match(/expected (\d+), got (\d+)/i);
  return {
    verdict: 'fail',
    reason: 'shape_mismatch',
    shape_matches: false,
    compared_elements: null,
    max_abs_error: null,
    max_rel_error: null,
    first_failure: null,
    ...(lengths
      ? {expected_elements: Number(lengths[1]), candidate_elements: Number(lengths[2])}
      : {candidate_elements: candidateElements}),
  };
}

// Non-mutating ingress preflight gate. The manifest status `ready` is the true
// execution gate: it covers graph preflight AND runtime backing (logical port
// mappings, contract, registry binding). Throws HostError
// (execution_started=false) when the ingress is not ready, so the host never
// mislabels a pre-execution rejection as an execution failure.
function assertIngressPreflightReady(s, op) {
  let status;
  try {
    status = ingressStatusJson(s);
  } catch (error) {
    throw fail('internal_error', op,
      `ingress status probe failed: ${String(error?.message ?? error)}`,
      {execution_started: false});
  }
  if (status && status.ready === true) return;
  const notReady = sessionBlockers(status);
  const gaps = logicalPortGaps(status);
  if (gaps.length) {
    throw fail('missing_required_field', op,
      'logicalPorts is required: every required input slot must be mapped to a logical port before run', {
        execution_started: false,
        path: 'logicalPorts',
        required: true,
        missing_ports: gaps,
        not_ready_ports: notReady,
        remediation_hint: 'declare the slots in create.logicalPorts or map them with op=map, then bind and run',
      });
  }
  throw fail('preflight_failed', op, 'ingress preflight failed; execution was not started', {
    execution_started: false,
    graph_preflight_ready: status?.graph_preflight?.ready ?? null,
    runtime_coverage_complete: status?.runtime_coverage_complete ?? null,
    registry_binding_current: status?.registry_binding_current ?? null,
    not_ready_ports: notReady,
    remediation_hint: 'map every required slot to a logical port (create.logicalPorts or op=map) and bind all required inputs (op=bind) before run/trace',
  });
}

// Required input slots that have no logical-port mapping (no runtime
// backing). These must be declared via create.logicalPorts (or mapped later
// with op=map) before run/trace can execute.
function logicalPortGaps(status) {
  const ports = Array.isArray(status?.ports) ? status.ports : [];
  return ports
    .filter(p => p && p.required === true && (p.logical_port_id == null || p.status === 'runtime_backing_missing'))
    .map(p => ({slot: p.slot, role: p.role ?? null, layout: p.layout ?? null, status: p.status}));
}

// Session-level blockers: required ports that keep the session from executing.
// Shared by the run/trace preflight gate, the bind response, and validate.
// Session-level port statuses observed from the WASM manifest:
//   good: ready, contract_ready, runtime_backing_current
//   blocking: runtime_backing_missing, runtime_input_unbound, source_mismatch,
//             contract_mismatch, deferred_no_runtime_backing, ...
function sessionBlockers(status) {
  const GOOD = new Set(['ready', 'contract_ready', 'runtime_backing_current']);
  const ports = Array.isArray(status?.ports) ? status.ports : [];
  return ports
    .filter(p => p && !GOOD.has(p.status))
    .map(p => ({slot: p.slot, logical_port_id: p.logical_port_id ?? null,
      status: p.status, required: p.required ?? false}));
}

// Binding lifecycle (complaint #03):
// - accepted:   the bind op passed local validation and the value is stored.
// - resolvable: the binding resolves to a live logical port with a satisfied
//               contract (mapped, source matches, role/layout/shape agree).
// - executable: the session as a whole is ready to execute.
// accepted=true with resolvable=false is the honest signal for "stored but
// not usable yet" (unmapped slot, source mismatch, contract mismatch).
function assessBinding(session, slotPort) {
  const port = slotPort?.port;
  const resolvable = !!port && port.logical_port_id != null && port.status === 'runtime_backing_current';
  const status = ingressStatusJson(session);
  return {resolvable, executable: status.ready === true, blockers: sessionBlockers(status)};
}

// Explicit preflight summary for the create response: names the `logicalPorts`
// field and the required slots still missing a mapping, so the caller is told
// upfront instead of discovering it at run time.
function logicalPortsPreflight(status) {
  const gaps = logicalPortGaps(status);
  return {
    ready: status?.ready === true,
    missing_required_fields: gaps.length ? ['logicalPorts'] : [],
    unmapped_required_slots: gaps,
    ...(gaps.length ? {remediation_hint:
      'declare every required slot in create.logicalPorts, or map it later with op=map, before run/trace'} : {}),
  };
}

// Execution state semantics (complaint #05). The compiled graph reads live
// weights from the registry on every run; there is no snapshot mode. This is
// declared identically on capabilities, explain, and every run/trace result.
const EXECUTION_STATE_SEMANTICS = 'live_registry';

// layerMeta: [{id, type}] per layer index. Built from the create command
// (args[0] is the registry layer id by AgentLayerSpec convention) or parsed
// back from program_identity.layer_init_fingerprints after a bundle restore
// (Rust emits `type={:02x};id={};...`, see registry.rs LayerInitIdentity).
function layerMetaFromFingerprints(fingerprints) {
  return fingerprints.map(fp => {
    const m = /^type=([0-9a-fA-F]{2});id=(\d+);/.exec(fp);
    if (!m) throw fail('internal_error', 'restore', 'unparseable layer init fingerprint', {fingerprint: fp});
    return {id: Number(m[2]), type: parseInt(m[1], 16)};
  });
}

function initWeightTracking(next, layerMeta) {
  next.layerMeta = layerMeta;
  next.registryRevision = 0;
  next.layerWeightRevisions = layerMeta.map(() => 0);
}

// Layer types whose weights are readable/mutable via get/setWeightsFlat
// (protocol.rs: linear 0x01, norm 0x02, conv 0x03, embedding 0x05). Stateless
// ops (activation, binary/add, pooling, ...) carry no weights and contribute
// nothing to the digest.
const WEIGHTED_LAYER_TYPES = new Set([0x01, 0x02, 0x03, 0x05]);

// sha256 over the live weights of every weight-bearing layer in layer order:
// the state that was actually executed (complaint #04). Each layer's digest
// input is framed with (type, id, length) so weight blocks cannot shift
// across layers undetected. Computed from the registry at call time, never
// cached, so it always reflects the executed weights.
function computeStateDigest(s) {
  const hash = createHash('sha256');
  for (const meta of s.layerMeta) {
    if (!WEIGHTED_LAYER_TYPES.has(meta.type)) continue;
    const w = s.registry.getWeightsFlat(meta.id, meta.type);
    const frame = Buffer.alloc(8);
    frame.writeUInt8(meta.type, 0);
    frame.writeUInt32LE(meta.id, 1);
    frame.writeUInt16LE(w.length, 5);
    hash.update(frame);
    hash.update(Buffer.from(w.buffer, w.byteOffset, w.byteLength));
  }
  return `sha256:${hash.digest('hex')}`;
}

function executionState(s) {
  return {
    execution_state_semantics: EXECUTION_STATE_SEMANTICS,
    registry_revision: s.registryRevision,
    layer_weight_revisions: [...s.layerWeightRevisions],
    state_digest: computeStateDigest(s),
  };
}

function errorEnvelope(error, fallbackPhase) {
  if (error instanceof HostError) {
    const {code, phase, message, details, executionStarted} = error;
    const {execution_started: _ignored, ...rest} = details;
    return {code, phase, message, execution_started: executionStarted, ...rest};
  }
  return {code: 'internal_error', phase: fallbackPhase ?? 'unknown',
    message: String(error?.message ?? error), execution_started: false};
}
const supportedConstructors = new Set(JSON.parse(wasm.agentCapabilities()).agent_facade.constructors);
let session;

function requireSession() {
  if (!session) throw fail('session_required', 'session', 'create a graph session first',
    {remediation_hint: 'send op=create before any session operation'});
  return session;
}

function release(value) {
  value?.free?.();
}

function releaseSession(value) {
  if (!value) return;
  for (const resource of [value.ingress, value.bundle, value.graph, value.plan, value.builder, ...value.layers.slice().reverse(), value.registry]) {
    release(resource);
  }
}

function exportStateCheckpoint(value) {
  const bytes = wasm.exportMultiInputProgramBundle(value.graph, value.registry, true);
  const checkpointBytesSha256 = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  return {bytes, checkpoint_bytes_sha256: checkpointBytesSha256};
}

function requireArray(value, name) {
  if (!Array.isArray(value)) throw fail('invalid_argument', 'input', `${name} must be an array`,
    {path: name, expected: 'array'});
  return value;
}

function manifestContext(value, slot, stateBound = false) {
  const manifest = manifestJson(value);
  const port = manifest.ports.find(port => port.backing === 'graph_input_slot' && port.slot === slot);
  if (!port) throw fail('missing_logical_port', 'preflight', `slot ${slot} has no logical port mapping`,
    {slot, remediation_hint: 'map the slot with op=map or declare it in create.logicalPorts'});
  return {
    plan_hex: manifest.plan_hex,
    manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest),
    logical_port_id: port.logical_port_id,
    expected_source: port.expected_source,
    ...(stateBound ? {active_state_checkpoint_bytes_sha256: exportStateCheckpoint(value).checkpoint_bytes_sha256} : {}),
  };
}

// Complaint #10: the execution gate, stated once. `ready` is the documented
// gate: in signed modes host_provenance.ready covers ingress readiness AND
// per-port signature verification; in caller_declared mode there are no
// signatures, so execution is gated by the ingress status.ready field instead.
// The old `execution_authorized` field (hardcoded false everywhere, including
// inside WASM status blobs) is removed: it named an allow/deny decision it
// never made, and misled callers into blocking legitimate runs.
const EXECUTION_GATE = Object.freeze({
  field: 'ready',
  meaning: 'host_provenance.ready === true is necessary and sufficient for the host to permit execution. '
    + 'In signed modes it covers ingress readiness and verification of every bound input claim; '
    + 'in caller_declared mode host_provenance.ready is null and execution is gated by the ingress status.ready field instead.',
  supersedes: 'execution_authorized (removed; it was hardcoded false and never represented a gate decision)',
});

// Presentation helper for the WASM ingress status blob: parse once and strip
// the misleading hardcoded `execution_authorized` field wherever it appears
// (top level and inside the nested graph_preflight report). The documented
// execution gate is `ready` (see EXECUTION_GATE).
function stripExecutionAuthorized(value) {
  if (Array.isArray(value)) { for (const item of value) stripExecutionAuthorized(item); return; }
  if (value && typeof value === 'object') {
    delete value.execution_authorized;
    for (const key of Object.keys(value)) stripExecutionAuthorized(value[key]);
  }
}
function ingressStatusJson(value) {
  const status = JSON.parse(value.ingress.status(value.registry, value.graph, value.bundle));
  stripExecutionAuthorized(status);
  return status;
}
// The WASM ingress manifest in its canonical host-presented form: parsed once
// with the misleading hardcoded `execution_authorized` stripped (complaint
// #10). ALL host-side uses — client-visible presentation, signed-claim digest
// computation, receipt binding, checkpoint retention — go through this helper,
// so digests a client computes over the presented manifest always match the
// digests the host verifies.
function manifestJson(value) {
  const manifest = JSON.parse(value.ingress.toJSON());
  stripExecutionAuthorized(manifest);
  return manifest;
}

function provenanceDenialReason(ingressReady, ports) {
  if (!ingressReady) return 'ingress_not_ready';
  const bad = ports.find(port => port.status !== 'host_signature_verified');
  return bad ? bad.status : null;
}

function provenanceStatus(value) {
  if (!provenanceVerifier) return {
    mode: 'caller_declared',
    authority_mode: 'caller_declared',
    ready: null,
    execution_gate: EXECUTION_GATE,
    enforcement_effect: 'none: caller-declared inputs carry no signatures; the host checks structural readiness only',
    claim_verification: null,
    denial_reason: null,
    host_authority_attested: false,
  };
  const ingress = ingressStatusJson(value);
  const currentManifestSha = manifestDigest(manifestJson(value));
  const stateBoundInputClaimsSupported = provenanceVerifier.stateBoundInputClaimsSupported;
  const activeStateCheckpointBytesSha256 = stateBoundInputClaimsSupported || requireStateBoundInputs
    ? exportStateCheckpoint(value).checkpoint_bytes_sha256 : null;
  const ports = ingress.ports.filter(port => Number.isInteger(port.slot)).map(port => {
    const claim = value.proofs.get(port.slot);
    const current = Boolean(claim && (!provenanceVerifier.hostSubject || claim.subject === provenanceVerifier.hostSubject)
      && claim.manifest_sha256 === currentManifestSha
      && (!requireStateBoundInputs || claim.schema === SIGNED_INPUT_CLAIM_SCHEMA_V2)
      && (claim.schema !== SIGNED_INPUT_CLAIM_SCHEMA_V2
        || claim.active_state_checkpoint_bytes_sha256 === activeStateCheckpointBytesSha256)
      && claim.slot === port.slot && claim.source === port.actual_source
      && claim.revision === String(port.revision) && port.status === 'runtime_backing_current');
    return {slot: port.slot, status: current ? 'host_signature_verified' : claim ? 'stale_or_unbound' : 'missing_signed_claim',
      source: claim?.source ?? null, subject: claim?.subject ?? null, key_id: claim?.key_id ?? null};
  });
  const ready = ingress.ready && ports.every(port => port.status === 'host_signature_verified');
  const verifiedPorts = ports.filter(port => port.status === 'host_signature_verified').length;
  return {
    mode: provenanceVerifier.mode,
    authority_mode: provenanceVerifier.mode,
    ready,
    execution_gate: EXECUTION_GATE,
    enforcement_effect: 'deny-by-default: unsigned, forged, replayed, stale, or cross-slot claims are rejected; '
      + 'execution requires host_provenance.ready === true',
    claim_verification: {verified_ports: verifiedPorts, total_ports: ports.length,
      all_verified: verifiedPorts === ports.length, ports},
    denial_reason: provenanceDenialReason(ingress.ready, ports),
    host_authority_attested: ready,
    replay_scope: provenanceVerifier.replayScope,
    host_subject: provenanceVerifier.hostSubject,
    state_bound_input_claims_supported: stateBoundInputClaimsSupported,
    state_bound_input_claims_required: requireStateBoundInputs,
    ...(activeStateCheckpointBytesSha256 ? {active_state_checkpoint_bytes_sha256: activeStateCheckpointBytesSha256} : {}),
    ports};
}

function requireProvenance(value) {
  const status = provenanceStatus(value);
  if (provenanceVerifier && !status.ready) throw fail('provenance_preflight_failed', 'preflight',
    'host provenance preflight failed; execution was not started', {execution_started: false});
  return status;
}

function withCurrentProvenance(value, execute) {
  return provenanceVerifier ? provenanceVerifier.withCurrent(value.proofs.values(), execute) : execute();
}

function assertCurrentStateBoundClaims(value) {
  const claims = [...value.proofs.values()].filter(claim => claim.schema === SIGNED_INPUT_CLAIM_SCHEMA_V2);
  if (requireStateBoundInputs && claims.length !== value.proofs.size) {
    throw fail('policy_violation', 'preflight',
      'host policy requires every signed input to use state-bound claim v2',
      {policy: 'state_bound_claim_v2_required', remediation_hint: 'sign inputs with claim schema v2'});
  }
  if (!claims.length) return;
  const active = exportStateCheckpoint(value).checkpoint_bytes_sha256;
  if (claims.some(claim => claim.active_state_checkpoint_bytes_sha256 !== active)) {
    throw fail('stale_state_claim', 'preflight',
      'signed claim checkpoint digest differs from the active program state');
  }
}

function createSession(command) {
  // Complaint #05: the only supported execution state semantics is live_registry.
  // Reject anything else explicitly instead of silently running live anyway.
  if (command.execution_state_semantics !== undefined
      && command.execution_state_semantics !== EXECUTION_STATE_SEMANTICS)
    throw fail('invalid_argument', 'create',
      `unsupported execution_state_semantics: ${command.execution_state_semantics}`,
      {supported_execution_state_semantics: [EXECUTION_STATE_SEMANTICS],
       remediation_hint: 'the compiled graph always reads live registry weights; snapshot mode is not supported'});
  const next = {registry: new wasm.LayerRegistry(), layers: [], proofs: new Map(), handoffs: new Map()};
  try {
    const layerDecls = requireArray(command.layers, 'layers');
    const layerMeta = [];
    for (const layer of layerDecls) {
      if (!supportedConstructors.has(layer.constructor)) throw fail('unknown_layer_constructor', 'create',
        `unknown typed layer constructor: ${layer.constructor}`,
        {constructor: layer.constructor, remediation_hint: 'see capabilities.agent.agent_facade.constructors'});
      const spec = wasm.AgentLayerSpec[layer.constructor](...requireArray(layer.args, 'layer.args'));
      next.registry.initAgentLayer(spec);
      next.layers.push(spec);
      const id = layer.args[0];
      if (!Number.isInteger(id) || id < 0) throw fail('invalid_argument', 'create',
        'layer.args[0] must be the non-negative registry layer id', {constructor: layer.constructor});
      layerMeta.push({id, type: spec.layerType()});
    }
    initWeightTracking(next, layerMeta);
    next.builder = new wasm.AgentGraphBuilder(command.numSlots);
    for (const step of requireArray(command.steps, 'steps')) {
      const spec = next.layers[step.layer];
      if (!spec) throw fail('missing_layer', 'create', `missing layer at index ${step.layer}`, {layer_index: step.layer});
      const slots = requireArray(step.slots, 'step.slots');
      if (step.kind === 'unary' && slots.length === 2) next.builder.addUnary(spec, ...slots);
      else if (step.kind === 'binary' && slots.length === 3) next.builder.addBinary(spec, ...slots);
      else throw fail('invalid_step', 'create', 'step.kind and step.slots must describe a unary or binary step',
        {step_kind: step.kind, slots: step.slots});
    }
    next.builder.setOutput(command.outputSlot);
    // Complaint #08: fail the two plan invariants with distinct codes before
    // the WASM layer collapses them into one generic error.
    assessPlanPorts(command, 'create');
    next.plan = next.builder.multiInputPlanV1();
    for (const port of requireArray(command.ports, 'ports')) {
      // Complaint #07: name the actual value and the valid choices up front.
      assertValidRole(port.role, 'create', {slot: port.slot});
      assertValidLayout(port.layout, 'create', {slot: port.slot});
      const shape = requireArray(port.shape, 'port.shape');
      if (shape.length !== 4) throw fail('invalid_port_shape', 'create', 'port.shape must contain four dimensions',
        {slot: port.slot, shape});
      next.plan.addInputPort(port.slot, port.role, ...shape, port.layout, port.requireFingerprint ?? false, BigInt(port.minimumRevision ?? 0));
    }
    next.graph = next.registry.compileMultiInputGraph(next.plan);
    next.bundle = new wasm.MultiInputInputBundle(next.plan);
    next.ingress = new wasm.SemanticIngressManifestV2(next.plan);
    for (const port of command.logicalPorts ?? []) next.ingress.addRuntimePort(port.id, port.slot, port.source);
    for (const port of command.deferredPorts ?? []) next.ingress.addDeferredPort(port.id, port.role, port.required ?? false);
  } catch (error) {
    releaseSession(next);
    throw error;
  }
  const old = session;
  session = next;
  releaseSession(old);
  const status = ingressStatusJson(next);
  return {
    manifest: manifestJson(next),
    program_identity: JSON.parse(next.graph.programIdentity()),
    status,
    preflight: logicalPortsPreflight(status),
    host_provenance: provenanceStatus(next),
  };
}

function restoreSession(receiptId) {
  if (!allowCheckpointRestore) throw fail('policy_violation', 'restore',
      'checkpoint restore is disabled by host startup policy',
      {policy: 'checkpoint_restore_disabled', remediation_hint: 'restart the host with --allow-checkpoint-restore and durable signed ingress'});
  const {receipt, bytes, manifest: retainedManifest} = provenanceVerifier.getCheckpoint(receiptId);
  // The retained manifest is stored in the canonical presented form (the
  // misleading `execution_authorized` field is stripped at retention time,
  // see manifestJson). Ledgers written before that change predate any merge
  // of this branch and are not supported across the migration.
  const retainedStripped = JSON.parse(JSON.stringify(retainedManifest));
  stripExecutionAuthorized(retainedStripped);
  const next = {registry: new wasm.LayerRegistry(), layers: [], proofs: new Map(), handoffs: new Map()};
  try {
    next.graph = wasm.importMultiInputProgramBundle(next.registry, bytes);
    const restoredIdentity = JSON.parse(next.graph.programIdentity());
    if (JSON.stringify(restoredIdentity) !== JSON.stringify(receipt.program_identity)) {
      throw fail('restore_identity_mismatch', 'restore', 'restored graph structural identity differs from receipt');
    }
    initWeightTracking(next,
      layerMetaFromFingerprints(restoredIdentity.layer_init_fingerprints ?? []));
    next.plan = wasm.MultiInputGraphPlan.fromBytes(next.graph.inputPlanV1());
    next.bundle = new wasm.MultiInputInputBundle(next.plan);
    next.ingress = new wasm.SemanticIngressManifestV2(next.plan);
    for (const port of requireArray(retainedStripped.ports, 'checkpoint manifest ports')) {
      if (port.backing === 'graph_input_slot') next.ingress.addRuntimePort(port.logical_port_id, port.slot, port.expected_source);
      else if (port.backing === 'deferred') next.ingress.addDeferredPort(port.logical_port_id, port.role, port.required);
      else throw fail('invalid_argument', 'restore', 'unknown checkpoint manifest port backing', {backing: port.backing});
    }
    if (JSON.stringify(manifestJson(next)) !== JSON.stringify(retainedStripped)
      || manifestDigest(retainedStripped) !== receipt.manifest_sha256) {
      throw fail('restore_manifest_mismatch', 'restore', 'restored manifest differs from receipt');
    }
    next.restoreEvent = provenanceVerifier.recordRestore(receiptId, JSON.parse(next.graph.programIdentity()),
      manifestDigest(retainedStripped), receipt.state_checkpoint_bytes_sha256);
    next.stateParentReceiptId = receiptId;
  } catch (error) {
    releaseSession(next);
    throw error;
  }
  const old = session;
  session = next;
  releaseSession(old);
  return {restored_from_receipt_id: receipt.receipt_id, restore_event_id: next.restoreEvent.restore_id,
    checkpoint_bytes_sha256: receipt.state_checkpoint_bytes_sha256,
    manifest: retainedStripped, program_identity: JSON.parse(next.graph.programIdentity()),
    status: ingressStatusJson(next),
    host_provenance: provenanceStatus(next)};
}

function handle(command) {
  switch (command.op) {
    case 'capabilities':
      return {
        execution_state_semantics: EXECUTION_STATE_SEMANTICS,
        supported_execution_state_semantics: [EXECUTION_STATE_SEMANTICS],
        agent: JSON.parse(wasm.agentCapabilities()),
        ingress: JSON.parse(wasm.semanticIngressManifestV2Capabilities()),
        multi_input: JSON.parse(wasm.multiInputGraphCapabilities()),
        program_bundle: JSON.parse(wasm.multiInputProgramBundleCapabilities()),
        // Complaint #07: role/layout vocabulary discoverable without extracting
        // strings from the binary.
        vocabulary: vocabularyDocument(),
        // Complaint #08: multi-input plan invariants disclosed up front.
        plan_constraints: planConstraintsDocument(),
        // Complaint #06: canonical layer-type identity across every surface.
        layer_types: layerTypeCatalog(),
        // Complaints #10/#11/#12: execution gate, fingerprint contract, and
        // checkpoint-integrity responsibility, stated once.
        runtime_semantics: {
          execution_gate: EXECUTION_GATE,
          authority_modes: ['caller_declared', 'ed25519_host_enforced', 'ed25519_host_durable'],
          fingerprint_requirement: {
            description: 'the input fingerprint invariant differs by ingress authority mode; '
              + 'there are no hidden requirements when migrating from caller_declared to signed ingress',
            by_authority_mode: {
              caller_declared: {fingerprint_required: false,
                note: 'bind accepts an empty fingerprint; no signature is checked'},
              ed25519_host_enforced: {fingerprint_required: true, path: 'fingerprint',
                constraint: 'non_empty',
                note: 'the canonical signed claim requires a nonempty fingerprint; '
                  + 'an empty fingerprint fails bind with an invalid_argument error naming '
                  + 'path=fingerprint, constraint=non_empty and the authority_mode'},
              ed25519_host_durable: {fingerprint_required: true, path: 'fingerprint',
                constraint: 'non_empty',
                note: 'same as ed25519_host_enforced; the fingerprint is also committed to the replay ledger'},
            },
          },
          checkpoint_integrity: {
            byte_level_integrity: 'host_responsibility',
            detail: 'the program bundle format validates structure per field; it carries no trailing '
              + 'whole-payload checksum, so a caller that stores or moves bundle bytes must verify '
              + 'integrity itself. Use op verifyCheckpointIntegrity with the digest recorded at export time.',
            helper: 'verifyCheckpointIntegrity',
            durable_mode: 'restore verifies sha256(bundle bytes) against the receipt '
              + 'state_checkpoint_bytes_sha256 before import; mismatched bytes are rejected',
            bundle_versioning: 'bundles remain readable across versions via the explicit bundle schema field',
          },
        },
        host_provenance: {mode: provenanceVerifier?.mode ?? 'caller_declared',
          signed_claim: SIGNED_INPUT_CLAIM_SCHEMA_V1,
          signed_claim_schemas: provenanceVerifier?.acceptedClaimSchemas ?? [], trust_root: 'host_startup_only',
          replay_scope: provenanceVerifier?.replayScope ?? 'none',
          host_subject: provenanceVerifier?.hostSubject ?? null,
          state_bound_input_claims_supported: provenanceVerifier?.stateBoundInputClaimsSupported ?? false,
          state_bound_input_claims_required: requireStateBoundInputs,
          execution_receipt: provenanceVerifier?.ledger ? 'burn-research.host-execution-receipt.v2' : null,
          state_checkpoint: 'burn-research.multi-input-program-bundle.v1',
          state_checkpoint_export_enabled: allowStateCheckpointExport,
          receipt_bound_checkpoint_restore_enabled: allowCheckpointRestore,
          checkpoint_restore_event: provenanceVerifier?.ledger ? 'burn-research.host-checkpoint-restore.v1' : null,
          state_handoff: provenanceVerifier?.ledger ? 'burn-research.host-state-handoff.v1' : null,
          wasm_origin_authentication: false},
      };
    // Complaint #08: session-free plan validator. Reports the port analysis
    // without creating a session; create uses the same analysis and throws
    // plan_minimum_ports / plan_unused_ports on violation.
    case 'validatePlan': {
      const analysis = analyzePlanPorts(command.numSlots, command.steps, command.ports, command.outputSlot);
      const valid = analysis.declared_ports.length >= analysis.minimum_ports
        && analysis.unused_ports.length === 0;
      return {...analysis, valid,
        code: valid ? 'ok'
          : (analysis.declared_ports.length < analysis.minimum_ports ? 'plan_minimum_ports' : 'plan_unused_ports')};
    }
    case 'create': return createSession(command);
    case 'restore': return restoreSession(command.receipt_id);
    case 'explain': {
      const s = requireSession();
      return presentPlanExplanation(JSON.parse(s.graph.explainPlan(s.registry)));
    }
    case 'map': {
      const s = requireSession();
      return {changed: s.ingress.addRuntimePort(command.id, command.slot, command.source),
        status: ingressStatusJson(s), host_provenance: provenanceStatus(s)};
    }
    case 'defer': {
      const s = requireSession();
      // Complaint #07: deferred-port roles use the same vocabulary.
      assertValidRole(command.role, 'defer', {logical_port_id: command.id});
      return {changed: s.ingress.addDeferredPort(command.id, command.role, command.required ?? false),
        status: ingressStatusJson(s), host_provenance: provenanceStatus(s)};
    }
    case 'setWeights': {
      const s = requireSession();
      const index = command.layer;
      const meta = Number.isInteger(index) ? s.layerMeta[index] : undefined;
      if (!meta) throw fail('invalid_argument', 'setWeights', 'unknown layer index',
        {layer: index, layer_count: s.layerMeta.length,
         remediation_hint: 'use a layer index from 0 to layer_count-1'});
      if (!WEIGHTED_LAYER_TYPES.has(meta.type)) throw fail('invalid_argument', 'setWeights',
        'layer type carries no mutable weights', {layer: index, layer_type: meta.type});
      const values = requireArray(command.values, 'values');
      const current = s.registry.getWeightsFlat(meta.id, meta.type);
      if (values.length !== current.length) throw fail('invalid_argument', 'setWeights',
        'weight count mismatch for layer',
        {layer: index, expected: current.length, actual: values.length});
      if (!values.every(v => typeof v === 'number' && Number.isFinite(v))) throw fail('invalid_argument',
        'setWeights', 'weights must be finite numbers', {layer: index});
      s.registry.setWeightsFlat(meta.id, meta.type, new Float32Array(values));
      s.registryRevision += 1;
      s.layerWeightRevisions[index] += 1;
      return {layer: index, execution_state: executionState(s),
        host_provenance: provenanceStatus(s)};
    }
    case 'bind': {
      const s = requireSession();
      const hasValues = command.values !== undefined;
      const hasBytes = command.values_f32_le_base64 !== undefined;
      if (hasValues === hasBytes) throw fail('invalid_argument', 'bind',
        'bind must provide exactly one of values or values_f32_le_base64',
        {remediation_hint: 'provide values (f64 array) or values_f32_le_base64, not both/neither'});
      const shape = requireArray(command.shape, 'shape');
      const values = hasBytes ? decodeF32Base64(command.values_f32_le_base64, shape) : requireArray(command.values, 'values');
      const binding = {...command, values};
      // Complaint #07: vocabulary errors name the actual value and valid choices.
      assertValidRole(command.role, 'bind', {slot: command.slot});
      assertValidLayout(command.layout, 'bind', {slot: command.slot});
      const claimSchema = command.proof?.claim?.schema;
      if (requireStateBoundInputs && claimSchema !== SIGNED_INPUT_CLAIM_SCHEMA_V2) {
        throw fail('policy_violation', 'preflight',
          'host policy requires every signed input to use state-bound claim v2',
          {policy: 'state_bound_claim_v2_required', remediation_hint: 'sign inputs with claim schema v2'});
      }
      const context = provenanceVerifier
        ? manifestContext(s, command.slot, claimSchema === SIGNED_INPUT_CLAIM_SCHEMA_V2) : null;
      if (context && command.source !== context.expected_source) throw fail('source_mismatch', 'bind',
        'signed input source differs from the current logical port',
        {slot: command.slot, actual_source: command.source, expected_source: context.expected_source});
      const hasHandoff = command.handoff_receipt_id !== undefined;
      if (hasHandoff) {
        if (!provenanceVerifier?.ledger) throw fail('policy_violation', 'bind',
          'state handoff requires durable signed ingress', {policy: 'durable_ingress_required'});
        if (command.role !== 'state') throw fail('invalid_handoff', 'bind',
          'receipt handoff is only valid for a state input', {role: command.role});
        if (!hasBytes) throw fail('invalid_handoff', 'bind',
          'state handoff requires the parent receipt output_f32_le_base64 bytes');
        const parent = provenanceVerifier.getReceipt(command.handoff_receipt_id);
        if (!encodedF32Matches(parent, shape, command.values_f32_le_base64)) {
          throw fail('handoff_payload_mismatch', 'bind',
            'state handoff payload does not match the parent receipt output',
            {handoff_receipt_id: command.handoff_receipt_id});
        }
      }
      // Complaint #11: the signed-ingress fingerprint contract, surfaced as a
      // preflight before the verifier. Signed modes require a nonempty
      // fingerprint; caller_declared mode accepts an empty one. The policy is
      // disclosed in capabilities.runtime_semantics.fingerprint_requirement so
      // migrating from caller_declared to signed ingress finds no hidden
      // requirement.
      if (provenanceVerifier && (typeof command.fingerprint !== 'string' || command.fingerprint.length === 0))
        throw fail('invalid_argument', 'bind',
          'signed ingress requires a nonempty input fingerprint',
          {path: 'fingerprint', constraint: 'non_empty', authority_mode: provenanceVerifier.mode,
           capability_ref: 'capabilities.runtime_semantics.fingerprint_requirement',
           remediation_hint: 'provide a nonempty fingerprint with the bind, as in the signed claim'});
      const ticket = provenanceVerifier?.verify(binding, context, command.proof);
      const tensor = new wasm.WasmTensor(new Float32Array(values), new Uint32Array(shape));
      try {
        const changed = s.bundle.bindInput(command.slot, tensor, command.role, command.layout, command.source, BigInt(command.revision), command.fingerprint ?? '');
        if (ticket && !changed) throw fail('stale_claim_rebind', 'bind',
          'signed claim cannot rebind an unchanged input', {slot: command.slot});
        if (ticket) {
          try {
            const handoff = provenanceVerifier.commit(ticket, hasHandoff ? {
              parentReceiptId: command.handoff_receipt_id,
              branchId: command.handoff_branch_id ?? 'main',
            } : undefined);
            s.proofs.set(command.slot, ticket.claim);
            if (handoff) s.handoffs.set(command.slot, handoff);
            else s.handoffs.delete(command.slot);
          } catch (error) {
            // A failed durable commit cannot leave a runnable tensor behind.
            s.bundle.clearInput(command.slot);
            s.proofs.delete(command.slot);
            s.handoffs.delete(command.slot);
            throw error;
          }
        }
        const slotStatus = JSON.parse(s.ingress.inputPortStatus(command.slot, s.bundle));
        stripExecutionAuthorized(slotStatus);
        const assessment = assessBinding(s, slotStatus);
        return {changed, status: slotStatus,
          binding: {accepted: true, resolvable: assessment.resolvable, executable: assessment.executable},
          blockers: assessment.blockers,
          host_provenance: provenanceStatus(s)};
      } finally {
        tensor.free();
      }
    }
    case 'clear': {
      const s = requireSession();
      const cleared = s.bundle.clearInput(command.slot);
      s.proofs.delete(command.slot);
      s.handoffs.delete(command.slot);
      return {cleared, status: ingressStatusJson(s), host_provenance: provenanceStatus(s)};
    }
    case 'port': {
      const s = requireSession();
      const portStatus = JSON.parse(s.ingress.inputPortStatus(command.slot, s.bundle));
      stripExecutionAuthorized(portStatus);
      return portStatus;
    }
    case 'consumer': {
      const s = requireSession();
      const spec = new wasm.InputPortConsumerSpec(command.id, JSON.stringify(requireArray(command.acceptedRoles, 'acceptedRoles')), command.allowExtensionRoles ?? false, command.requireFingerprint ?? false, BigInt(command.minimumRevision ?? 0));
      try {
        const compatibility = JSON.parse(s.ingress.consumerCompatibility(command.slot, s.bundle, spec));
        stripExecutionAuthorized(compatibility);
        return compatibility;
      } finally {
        spec.free();
      }
    }
    case 'validate': {
      const s = requireSession();
      const status = ingressStatusJson(s);
      const blockers = sessionBlockers(status);
      return {executable: status.ready === true, ready: status.ready === true,
        blockers, ports: status.ports, preflight: logicalPortsPreflight(status),
        host_provenance: provenanceStatus(s)};
    }
    case 'inspect': {
      const s = requireSession();
      const plan = JSON.parse(s.plan.toJSON());
      stripExecutionAuthorized(plan);
      return {
        manifest: manifestJson(s),
        plan,
        program_identity: JSON.parse(s.graph.programIdentity()),
        status: ingressStatusJson(s),
        host_provenance: provenanceStatus(s),
      };
    }
    case 'checkpoint': {
      if (!allowStateCheckpointExport) throw fail('policy_violation', 'checkpoint',
        'state checkpoint export is disabled by host startup policy',
        {policy: 'checkpoint_export_disabled', remediation_hint: 'restart the host with --allow-state-checkpoint-export and durable signed ingress'});
      const s = requireSession();
      const hostProvenance = requireProvenance(s);
      return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        const checkpoint = exportStateCheckpoint(s);
        return {schema: 'burn-research.multi-input-program-bundle.v1',
          program_identity: annotateProgramIdentity(JSON.parse(s.graph.programIdentity())), state_included: true,
          checkpoint_bytes_sha256: checkpoint.checkpoint_bytes_sha256,
          bundle_f32le_base64: Buffer.from(checkpoint.bytes).toString('base64'),
          host_provenance: hostProvenance};
      });
    }
    // Complaint #12: byte-level integrity is a host responsibility. The bundle
    // format validates structure per field and carries no trailing
    // whole-payload checksum, so this session-free helper lets a caller detect
    // corruption at ANY byte position (including trailing bytes the parser
    // never consumes) by comparing against the digest recorded at export time.
    // In durable mode, restore additionally verifies sha256(bytes) against the
    // receipt before import.
    case 'verifyCheckpointIntegrity': {
      const encoded = command.bundle_f32le_base64;
      if (typeof encoded !== 'string' || encoded.length === 0)
        throw fail('invalid_argument', 'verifyCheckpointIntegrity',
          'bundle_f32le_base64 must be a non-empty base64 string',
          {path: 'bundle_f32le_base64', remediation_hint: 'use the bundle_f32le_base64 value from op=checkpoint'});
      let bytes;
      try {
        bytes = Buffer.from(encoded, 'base64');
        if (bytes.length === 0) throw new Error('empty');
      } catch {
        throw fail('invalid_argument', 'verifyCheckpointIntegrity',
          'bundle_f32le_base64 is not valid base64', {path: 'bundle_f32le_base64'});
      }
      const expected = command.expected_digest;
      if (typeof expected !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(expected))
        throw fail('invalid_argument', 'verifyCheckpointIntegrity',
          'expected_digest must be a lowercase sha256 digest like sha256:<64 hex chars>',
          {path: 'expected_digest',
           remediation_hint: 'use the checkpoint_bytes_sha256 value recorded when the bundle was exported'});
      const actual = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
      return {integrity_ok: actual === expected, actual_digest: actual,
        expected_digest: expected, byte_length: bytes.length,
        verified_against: 'expected_digest',
        note: 'integrity_ok=false means at least one byte differs from the exported bundle'};
    }
    case 'trace':
    case 'run': {
      const s = requireSession();
      const hostProvenance = requireProvenance(s);
      const traced = command.op === 'trace';
      const traceOptions = traced ? {
        startStep: command.startStep ?? 0,
        maxSteps: command.maxSteps ?? 64,
        maxTensorBytes: command.maxTensorBytes ?? 1_048_576,
      } : null;
      if (traced && Object.entries(traceOptions).some(([key, value]) =>
        !Number.isSafeInteger(value) || value < 0 || value > 0xffffffff ||
        (key === 'maxSteps' && value === 0))) {
        throw fail('invalid_argument', 'trace',
        'trace bounds must be unsigned 32-bit integers; maxSteps must be positive',
        {startStep: traceOptions.startStep, maxSteps: traceOptions.maxSteps, maxTensorBytes: traceOptions.maxTensorBytes});
      }
      let executionStarted = false;
      const execute = () => {
        let output;
        let traceRun;
        try {
          assertIngressPreflightReady(s, 'preflight');
          executionStarted = true;
          let executionTrace;
          if (traced) {
            traceRun = s.ingress.runWithTrace(s.registry, s.graph, s.bundle,
              traceOptions.startStep, traceOptions.maxSteps, traceOptions.maxTensorBytes);
            executionTrace = JSON.parse(traceRun.report());
            if (executionTrace.execution_status !== 'completed') {
              const error = fail('trace_failed', 'trace',
                `trace failed at step ${executionTrace.fault_step_index}: ${executionTrace.error}`,
                {fault_step_index: executionTrace.fault_step_index, execution_started: true});
              error.execution_trace = presentExecutionTrace(executionTrace);
              throw error;
            }
            output = traceRun.output();
          } else {
            output = s.ingress.run(s.registry, s.graph, s.bundle);
          }
          const values = Array.from(output.to_array());
          if (traced && executionTrace.terminal_output?.finite_values === true
            && executionTrace.terminal_output.value_sha256 !== f32ValueDigest(values)) {
            throw fail('digest_mismatch', 'trace', 'trace terminal digest differs from exact host output bytes',
              {execution_started: true});
          }
          const checkpoint = provenanceVerifier?.ledger ? exportStateCheckpoint(s) : null;
          return {shape: Array.from(output.shape()), values,
            execution_state: executionState(s),
            ...(traced ? {execution_trace: presentExecutionTrace(executionTrace)} : {}),
            ...(checkpoint ? {state_checkpoint_bytes_sha256: checkpoint.checkpoint_bytes_sha256} : {}),
            ...(checkpoint ? {checkpoint_bytes: Buffer.from(checkpoint.bytes),
              checkpoint_manifest: manifestJson(s)} : {}),
            output_f32_le_base64: provenanceVerifier?.ledger ? f32ValueBytes(values).toString('base64') : undefined,
            ingress: ingressStatusJson(s), host_provenance: hostProvenance};
        } finally {
          output?.free();
          traceRun?.free();
        }
      };
      if (!provenanceVerifier?.ledger) return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        try {
          return execute();
        } catch (error) {
          if (executionStarted) {
            if (error instanceof HostError) error.executionStarted = true;
            else {
              const wrapped = fail('internal_error', command.op,
                String(error?.message ?? error), {execution_started: true});
              if (error?.execution_trace) wrapped.execution_trace = error.execution_trace;
              throw wrapped;
            }
          }
          throw error;
        }
      });
      try {
        const result = provenanceVerifier.executeWithReceipt(s.proofs.values(), {
          subject: provenanceVerifier.hostSubject,
          programIdentity: JSON.parse(s.graph.programIdentity()),
          manifestSha256: manifestDigest(manifestJson(s)),
          stateParentReceiptId: s.stateParentReceiptId,
          restoreEventId: s.restoreEvent?.restore_id,
        }, () => {
          assertCurrentStateBoundClaims(s);
          return execute();
        }, s.handoffs);
        s.stateParentReceiptId = result.execution_receipt.receipt_id;
        return result;
      } catch (error) {
        // A numerical call may mutate state before a failed disk commit.
        // Discard that state so no future receipt asserts an unproven parent.
        if (executionStarted) {
          if (error instanceof HostError) error.executionStarted = true;
          else {
            const wrapped = fail('internal_error', command.op,
              String(error?.message ?? error), {execution_started: true});
            if (error?.execution_trace) wrapped.execution_trace = error.execution_trace;
            session = undefined;
            releaseSession(s);
            throw wrapped;
          }
          session = undefined;
          releaseSession(s);
        }
        throw error;
      }
    }
    case 'receipt': {
      const receipt = provenanceVerifier?.getReceipt(command.receipt_id);
      if (!receipt) throw fail('ledger_required', 'receipt',
      'durable host ledger required to retrieve execution receipts',
      {remediation_hint: 'restart the host with a trust policy and ledgerPath for durable signed ingress'});
      const hasValues = command.values !== undefined;
      const hasBytes = command.output_f32_le_base64 !== undefined;
      if (hasValues && hasBytes) throw fail('invalid_argument', 'receipt',
      'compare output using either values or exact f32 bytes');
      if ((command.shape !== undefined) !== (hasValues || hasBytes)) throw fail('invalid_argument', 'receipt',
      'provide output shape together with values or f32 bytes');
      const shape = command.shape === undefined ? null : requireArray(command.shape, 'shape');
      // Complaint #06: the stored receipt is byte-identical; the canonical
      // layer-type identity is attached at presentation time only.
      const presented = receipt.program_identity
        ? {...receipt, program_identity: annotateProgramIdentity(receipt.program_identity)}
        : receipt;
      return {receipt: presented, output_matches: shape === null ? null : hasBytes
        ? encodedF32Matches(receipt, shape, command.output_f32_le_base64)
        : receiptMatchesOutput(receipt, shape, requireArray(command.values, 'values'))};
    }
    case 'verify': {
      const s = requireSession();
      const hostProvenance = requireProvenance(s);
      const candidate = new Float32Array(requireArray(command.candidate, 'candidate'));
      return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        let detail;
        try {
          detail = JSON.parse(s.ingress.verifyFlat(s.registry, s.graph, s.bundle,
            candidate, command.absTol ?? 1e-6, command.relTol ?? 1e-6));
        } catch (error) {
          const summary = shapeMismatchSummary(error, candidate.length);
          if (summary) return {verification_summary: summary, host_provenance: hostProvenance};
          throw error;
        }
        // Complaint #10: strip the misleading hardcoded execution_authorized
        // from the WASM verification detail; the gate is `ready`.
        stripExecutionAuthorized(detail);
        return {
          ...detail,
          verification_summary: summarizeVerification(detail?.reference?.verification),
          host_provenance: hostProvenance,
        };
      });
    }
    case 'close': {
      releaseSession(session);
      session = undefined;
      return {closed: true};
    }
    default: throw fail('unknown_operation', 'input', `unknown operation: ${command.op}`, {op: command.op});
  }
}

const input = readline.createInterface({input: process.stdin, crlfDelay: Infinity});
for await (const line of input) {
  if (!line.trim()) continue;
  let command;
  try {
    command = JSON.parse(line);
    const result = handle(command);
    process.stdout.write(`${JSON.stringify({request_id: command.request_id ?? null, ok: true, result})}\n`);
  } catch (error) {
    const envelope = errorEnvelope(error, command?.op);
    process.stdout.write(`${JSON.stringify({request_id: command?.request_id ?? null, ok: false,
      error: envelope.message, error_envelope: envelope,
      ...(error?.execution_trace ? {execution_trace: error.execution_trace} : {})})}\n`);
  }
  if (command?.op === 'close') break;
}
input.close();
releaseSession(session);
