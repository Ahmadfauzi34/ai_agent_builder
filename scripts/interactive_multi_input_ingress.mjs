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
const supportedConstructors = new Set(JSON.parse(wasm.agentCapabilities()).agent_facade.constructors);
let session;

function requireSession() {
  if (!session) throw new Error('create a graph session first');
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
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value;
}

function manifestContext(value, slot, stateBound = false) {
  const manifest = JSON.parse(value.ingress.toJSON());
  const port = manifest.ports.find(port => port.backing === 'graph_input_slot' && port.slot === slot);
  if (!port) throw new Error(`slot ${slot} has no logical port mapping`);
  return {
    plan_hex: manifest.plan_hex,
    manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest),
    logical_port_id: port.logical_port_id,
    expected_source: port.expected_source,
    ...(stateBound ? {active_state_checkpoint_bytes_sha256: exportStateCheckpoint(value).checkpoint_bytes_sha256} : {}),
  };
}

function provenanceStatus(value) {
  if (!provenanceVerifier) return {mode: 'caller_declared', ready: null, execution_authorized: false};
  const ingress = JSON.parse(value.ingress.status(value.registry, value.graph, value.bundle));
  const currentManifestSha = manifestDigest(JSON.parse(value.ingress.toJSON()));
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
  return {mode: provenanceVerifier.mode, ready: ingress.ready && ports.every(port => port.status === 'host_signature_verified'),
    execution_authorized: false, replay_scope: provenanceVerifier.replayScope,
    host_subject: provenanceVerifier.hostSubject,
    state_bound_input_claims_supported: stateBoundInputClaimsSupported,
    state_bound_input_claims_required: requireStateBoundInputs,
    ...(activeStateCheckpointBytesSha256 ? {active_state_checkpoint_bytes_sha256: activeStateCheckpointBytesSha256} : {}),
    ports};
}

function requireProvenance(value) {
  const status = provenanceStatus(value);
  if (provenanceVerifier && !status.ready) throw new Error('host provenance preflight failed; execution was not started');
  return status;
}

function withCurrentProvenance(value, execute) {
  return provenanceVerifier ? provenanceVerifier.withCurrent(value.proofs.values(), execute) : execute();
}

function assertCurrentStateBoundClaims(value) {
  const claims = [...value.proofs.values()].filter(claim => claim.schema === SIGNED_INPUT_CLAIM_SCHEMA_V2);
  if (requireStateBoundInputs && claims.length !== value.proofs.size) {
    throw new Error('host policy requires every signed input to use state-bound claim v2');
  }
  if (!claims.length) return;
  const active = exportStateCheckpoint(value).checkpoint_bytes_sha256;
  if (claims.some(claim => claim.active_state_checkpoint_bytes_sha256 !== active)) {
    throw new Error('signed claim checkpoint digest differs from the active program state');
  }
}

function createSession(command) {
  const next = {registry: new wasm.LayerRegistry(), layers: [], proofs: new Map(), handoffs: new Map()};
  try {
    for (const layer of requireArray(command.layers, 'layers')) {
      if (!supportedConstructors.has(layer.constructor)) throw new Error(`unknown typed layer constructor: ${layer.constructor}`);
      const spec = wasm.AgentLayerSpec[layer.constructor](...requireArray(layer.args, 'layer.args'));
      next.registry.initAgentLayer(spec);
      next.layers.push(spec);
    }
    next.builder = new wasm.AgentGraphBuilder(command.numSlots);
    for (const step of requireArray(command.steps, 'steps')) {
      const spec = next.layers[step.layer];
      if (!spec) throw new Error(`missing layer at index ${step.layer}`);
      const slots = requireArray(step.slots, 'step.slots');
      if (step.kind === 'unary' && slots.length === 2) next.builder.addUnary(spec, ...slots);
      else if (step.kind === 'binary' && slots.length === 3) next.builder.addBinary(spec, ...slots);
      else throw new Error('step.kind and step.slots must describe a unary or binary step');
    }
    next.builder.setOutput(command.outputSlot);
    next.plan = next.builder.multiInputPlanV1();
    for (const port of requireArray(command.ports, 'ports')) {
      const shape = requireArray(port.shape, 'port.shape');
      if (shape.length !== 4) throw new Error('port.shape must contain four dimensions');
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
  return {
    manifest: JSON.parse(next.ingress.toJSON()),
    program_identity: JSON.parse(next.graph.programIdentity()),
    status: JSON.parse(next.ingress.status(next.registry, next.graph, next.bundle)),
    host_provenance: provenanceStatus(next),
  };
}

function restoreSession(receiptId) {
  if (!allowCheckpointRestore) throw new Error('checkpoint restore is disabled by host startup policy');
  const {receipt, bytes, manifest} = provenanceVerifier.getCheckpoint(receiptId);
  const next = {registry: new wasm.LayerRegistry(), layers: [], proofs: new Map(), handoffs: new Map()};
  try {
    next.graph = wasm.importMultiInputProgramBundle(next.registry, bytes);
    if (JSON.stringify(JSON.parse(next.graph.programIdentity())) !== JSON.stringify(receipt.program_identity)) {
      throw new Error('restored graph structural identity differs from receipt');
    }
    next.plan = wasm.MultiInputGraphPlan.fromBytes(next.graph.inputPlanV1());
    next.bundle = new wasm.MultiInputInputBundle(next.plan);
    next.ingress = new wasm.SemanticIngressManifestV2(next.plan);
    for (const port of requireArray(manifest.ports, 'checkpoint manifest ports')) {
      if (port.backing === 'graph_input_slot') next.ingress.addRuntimePort(port.logical_port_id, port.slot, port.expected_source);
      else if (port.backing === 'deferred') next.ingress.addDeferredPort(port.logical_port_id, port.role, port.required);
      else throw new Error('unknown checkpoint manifest port backing');
    }
    if (JSON.stringify(JSON.parse(next.ingress.toJSON())) !== JSON.stringify(manifest)
      || manifestDigest(manifest) !== receipt.manifest_sha256) {
      throw new Error('restored manifest differs from receipt');
    }
    next.restoreEvent = provenanceVerifier.recordRestore(receiptId, JSON.parse(next.graph.programIdentity()),
      manifestDigest(manifest), receipt.state_checkpoint_bytes_sha256);
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
    manifest, program_identity: JSON.parse(next.graph.programIdentity()),
    status: JSON.parse(next.ingress.status(next.registry, next.graph, next.bundle)),
    host_provenance: provenanceStatus(next)};
}

function handle(command) {
  switch (command.op) {
    case 'capabilities':
      return {
        agent: JSON.parse(wasm.agentCapabilities()),
        ingress: JSON.parse(wasm.semanticIngressManifestV2Capabilities()),
        multi_input: JSON.parse(wasm.multiInputGraphCapabilities()),
        program_bundle: JSON.parse(wasm.multiInputProgramBundleCapabilities()),
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
    case 'create': return createSession(command);
    case 'restore': return restoreSession(command.receipt_id);
    case 'explain': {
      const s = requireSession();
      return JSON.parse(s.graph.explainPlan(s.registry));
    }
    case 'map': {
      const s = requireSession();
      return {changed: s.ingress.addRuntimePort(command.id, command.slot, command.source),
        status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle)), host_provenance: provenanceStatus(s)};
    }
    case 'defer': {
      const s = requireSession();
      return {changed: s.ingress.addDeferredPort(command.id, command.role, command.required ?? false),
        status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle)), host_provenance: provenanceStatus(s)};
    }
    case 'bind': {
      const s = requireSession();
      const hasValues = command.values !== undefined;
      const hasBytes = command.values_f32_le_base64 !== undefined;
      if (hasValues === hasBytes) throw new Error('bind must provide exactly one of values or values_f32_le_base64');
      const shape = requireArray(command.shape, 'shape');
      const values = hasBytes ? decodeF32Base64(command.values_f32_le_base64, shape) : requireArray(command.values, 'values');
      const binding = {...command, values};
      const claimSchema = command.proof?.claim?.schema;
      if (requireStateBoundInputs && claimSchema !== SIGNED_INPUT_CLAIM_SCHEMA_V2) {
        throw new Error('host policy requires every signed input to use state-bound claim v2');
      }
      const context = provenanceVerifier
        ? manifestContext(s, command.slot, claimSchema === SIGNED_INPUT_CLAIM_SCHEMA_V2) : null;
      if (context && command.source !== context.expected_source) throw new Error('signed input source differs from the current logical port');
      const hasHandoff = command.handoff_receipt_id !== undefined;
      if (hasHandoff) {
        if (!provenanceVerifier?.ledger) throw new Error('state handoff requires durable signed ingress');
        if (command.role !== 'state') throw new Error('receipt handoff is only valid for a state input');
        if (!hasBytes) throw new Error('state handoff requires the parent receipt output_f32_le_base64 bytes');
        const parent = provenanceVerifier.getReceipt(command.handoff_receipt_id);
        if (!encodedF32Matches(parent, shape, command.values_f32_le_base64)) {
          throw new Error('state handoff payload does not match the parent receipt output');
        }
      }
      const ticket = provenanceVerifier?.verify(binding, context, command.proof);
      const tensor = new wasm.WasmTensor(new Float32Array(values), new Uint32Array(shape));
      try {
        const changed = s.bundle.bindInput(command.slot, tensor, command.role, command.layout, command.source, BigInt(command.revision), command.fingerprint ?? '');
        if (ticket && !changed) throw new Error('signed claim cannot rebind an unchanged input');
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
        return {changed, status: JSON.parse(s.ingress.inputPortStatus(command.slot, s.bundle)),
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
      return {cleared, status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle)), host_provenance: provenanceStatus(s)};
    }
    case 'port': {
      const s = requireSession();
      return JSON.parse(s.ingress.inputPortStatus(command.slot, s.bundle));
    }
    case 'consumer': {
      const s = requireSession();
      const spec = new wasm.InputPortConsumerSpec(command.id, JSON.stringify(requireArray(command.acceptedRoles, 'acceptedRoles')), command.allowExtensionRoles ?? false, command.requireFingerprint ?? false, BigInt(command.minimumRevision ?? 0));
      try {
        return JSON.parse(s.ingress.consumerCompatibility(command.slot, s.bundle, spec));
      } finally {
        spec.free();
      }
    }
    case 'inspect': {
      const s = requireSession();
      return {
        manifest: JSON.parse(s.ingress.toJSON()),
        plan: JSON.parse(s.plan.toJSON()),
        program_identity: JSON.parse(s.graph.programIdentity()),
        status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle)),
        host_provenance: provenanceStatus(s),
      };
    }
    case 'checkpoint': {
      if (!allowStateCheckpointExport) throw new Error('state checkpoint export is disabled by host startup policy');
      const s = requireSession();
      const hostProvenance = requireProvenance(s);
      return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        const checkpoint = exportStateCheckpoint(s);
        return {schema: 'burn-research.multi-input-program-bundle.v1',
          program_identity: JSON.parse(s.graph.programIdentity()), state_included: true,
          checkpoint_bytes_sha256: checkpoint.checkpoint_bytes_sha256,
          bundle_f32le_base64: Buffer.from(checkpoint.bytes).toString('base64'),
          host_provenance: hostProvenance};
      });
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
        throw new Error('trace bounds must be unsigned 32-bit integers; maxSteps must be positive');
      }
      const execute = () => {
        let output;
        let traceRun;
        try {
          let executionTrace;
          if (traced) {
            traceRun = s.ingress.runWithTrace(s.registry, s.graph, s.bundle,
              traceOptions.startStep, traceOptions.maxSteps, traceOptions.maxTensorBytes);
            executionTrace = JSON.parse(traceRun.report());
            if (executionTrace.execution_status !== 'completed') {
              const error = new Error(`trace failed at step ${executionTrace.fault_step_index}: ${executionTrace.error}`);
              error.execution_trace = executionTrace;
              throw error;
            }
            output = traceRun.output();
          } else {
            output = s.ingress.run(s.registry, s.graph, s.bundle);
          }
          const values = Array.from(output.to_array());
          if (traced && executionTrace.terminal_output?.finite_values === true
            && executionTrace.terminal_output.value_sha256 !== f32ValueDigest(values)) {
            throw new Error('trace terminal digest differs from exact host output bytes');
          }
          const checkpoint = provenanceVerifier?.ledger ? exportStateCheckpoint(s) : null;
          return {shape: Array.from(output.shape()), values,
            ...(traced ? {execution_trace: executionTrace} : {}),
            ...(checkpoint ? {state_checkpoint_bytes_sha256: checkpoint.checkpoint_bytes_sha256} : {}),
            ...(checkpoint ? {checkpoint_bytes: Buffer.from(checkpoint.bytes),
              checkpoint_manifest: JSON.parse(s.ingress.toJSON())} : {}),
            output_f32_le_base64: provenanceVerifier?.ledger ? f32ValueBytes(values).toString('base64') : undefined,
            ingress: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle)), host_provenance: hostProvenance};
        } finally {
          output?.free();
          traceRun?.free();
        }
      };
      if (!provenanceVerifier?.ledger) return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        return execute();
      });
      let executionStarted = false;
      try {
        const result = provenanceVerifier.executeWithReceipt(s.proofs.values(), {
          subject: provenanceVerifier.hostSubject,
          programIdentity: JSON.parse(s.graph.programIdentity()),
          manifestSha256: manifestDigest(JSON.parse(s.ingress.toJSON())),
          stateParentReceiptId: s.stateParentReceiptId,
          restoreEventId: s.restoreEvent?.restore_id,
        }, () => {
          assertCurrentStateBoundClaims(s);
          executionStarted = true;
          return execute();
        }, s.handoffs);
        s.stateParentReceiptId = result.execution_receipt.receipt_id;
        return result;
      } catch (error) {
        // A numerical call may mutate state before a failed disk commit.
        // Discard that state so no future receipt asserts an unproven parent.
        if (executionStarted) {
          session = undefined;
          releaseSession(s);
        }
        throw error;
      }
    }
    case 'receipt': {
      const receipt = provenanceVerifier?.getReceipt(command.receipt_id);
      if (!receipt) throw new Error('durable host ledger required to retrieve execution receipts');
      const hasValues = command.values !== undefined;
      const hasBytes = command.output_f32_le_base64 !== undefined;
      if (hasValues && hasBytes) throw new Error('compare output using either values or exact f32 bytes');
      if ((command.shape !== undefined) !== (hasValues || hasBytes)) throw new Error('provide output shape together with values or f32 bytes');
      const shape = command.shape === undefined ? null : requireArray(command.shape, 'shape');
      return {receipt, output_matches: shape === null ? null : hasBytes
        ? encodedF32Matches(receipt, shape, command.output_f32_le_base64)
        : receiptMatchesOutput(receipt, shape, requireArray(command.values, 'values'))};
    }
    case 'verify': {
      const s = requireSession();
      const hostProvenance = requireProvenance(s);
      return withCurrentProvenance(s, () => {
        assertCurrentStateBoundClaims(s);
        return {
          ...JSON.parse(s.ingress.verifyFlat(s.registry, s.graph, s.bundle,
            new Float32Array(requireArray(command.candidate, 'candidate')), command.absTol ?? 1e-6, command.relTol ?? 1e-6)),
          host_provenance: hostProvenance,
        };
      });
    }
    case 'close': {
      releaseSession(session);
      session = undefined;
      return {closed: true};
    }
    default: throw new Error(`unknown operation: ${command.op}`);
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
    process.stdout.write(`${JSON.stringify({request_id: command?.request_id ?? null, ok: false,
      error: String(error?.message ?? error), ...(error?.execution_trace ? {execution_trace: error.execution_trace} : {})})}\n`);
  }
  if (command?.op === 'close') break;
}
input.close();
releaseSession(session);
