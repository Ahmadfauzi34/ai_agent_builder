import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath, pathToFileURL} from 'node:url';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const defaultPackageDir = fs.existsSync(path.join(scriptDir, 'node.mjs')) ? scriptDir : 'pkg';
const packageDir = path.resolve(process.argv[2] ?? defaultPackageDir);
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

function requireArray(value, name) {
  if (!Array.isArray(value)) throw new Error(`${name} must be an array`);
  return value;
}

function createSession(command) {
  const next = {registry: new wasm.LayerRegistry(), layers: []};
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
  };
}

function handle(command) {
  switch (command.op) {
    case 'capabilities':
      return {
        agent: JSON.parse(wasm.agentCapabilities()),
        ingress: JSON.parse(wasm.semanticIngressManifestV2Capabilities()),
        multi_input: JSON.parse(wasm.multiInputGraphCapabilities()),
      };
    case 'create': return createSession(command);
    case 'map': {
      const s = requireSession();
      return {changed: s.ingress.addRuntimePort(command.id, command.slot, command.source), status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle))};
    }
    case 'defer': {
      const s = requireSession();
      return {changed: s.ingress.addDeferredPort(command.id, command.role, command.required ?? false), status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle))};
    }
    case 'bind': {
      const s = requireSession();
      const tensor = new wasm.WasmTensor(new Float32Array(requireArray(command.values, 'values')), new Uint32Array(requireArray(command.shape, 'shape')));
      try {
        const changed = s.bundle.bindInput(command.slot, tensor, command.role, command.layout, command.source, BigInt(command.revision), command.fingerprint ?? '');
        return {changed, status: JSON.parse(s.ingress.inputPortStatus(command.slot, s.bundle))};
      } finally {
        tensor.free();
      }
    }
    case 'clear': {
      const s = requireSession();
      return {cleared: s.bundle.clearInput(command.slot), status: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle))};
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
      };
    }
    case 'run': {
      const s = requireSession();
      const output = s.ingress.run(s.registry, s.graph, s.bundle);
      try {
        return {shape: Array.from(output.shape()), values: Array.from(output.to_array()), ingress: JSON.parse(s.ingress.status(s.registry, s.graph, s.bundle))};
      } finally {
        output.free();
      }
    }
    case 'verify': {
      const s = requireSession();
      return JSON.parse(s.ingress.verifyFlat(s.registry, s.graph, s.bundle, new Float32Array(requireArray(command.candidate, 'candidate')), command.absTol ?? 1e-6, command.relTol ?? 1e-6));
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
    process.stdout.write(`${JSON.stringify({request_id: command?.request_id ?? null, ok: false, error: String(error?.message ?? error)})}\n`);
  }
  if (command?.op === 'close') break;
}
releaseSession(session);
