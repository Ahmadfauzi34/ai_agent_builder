// Shared WASM-direct helper for graph-vs-graph work (used by verify.mjs and checkpoint.mjs).
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { checkArgValues, analyzePlanPorts, isFiniteF32 } from './spec.mjs';
import { pkgDir } from './engine.mjs';
import { webcrypto } from 'node:crypto';

globalThis.crypto ??= webcrypto;

const HERE = path.dirname(fileURLToPath(import.meta.url));
// Engine resolution (single source of truth, shared with engine.mjs):
// BURN_WASM_PKG_DIR (canonical) or BURN_MATH_PKG_DIR (legacy alias) win;
// otherwise pkgDir() handles repo layout (pkg/) and built-pack layout
// (engine-audit/ under the pack root).
const PKG_DIR = process.env.BURN_WASM_PKG_DIR
  ? path.resolve(process.env.BURN_WASM_PKG_DIR)
  : process.env.BURN_MATH_PKG_DIR
    ? path.resolve(process.env.BURN_MATH_PKG_DIR)
    : pkgDir();

function codedError(code, phase, message) {
  const e = new Error(message);
  e.code = code;
  e.phase = phase;
  return e;
}

export function verifyEngineIntegritySync(pkgDir = PKG_DIR) {
  const jsPath = path.join(pkgDir, 'burn_research.js');
  const wasmPath = path.join(pkgDir, 'burn_research_bg.wasm');
  let pins;
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'MANIFEST.json'), 'utf8'));
    pins = manifest?.sha256 ?? {};
  } catch { pins = {}; }
  for (const [file, key] of [
    [wasmPath, 'burn_research_bg.wasm'],
    [jsPath, 'burn_research.js'],
  ]) {
    let bytes;
    try { bytes = fs.readFileSync(file); }
    catch (e) { throw codedError('invalid_argument', 'loadWasm', `cannot read engine file ${file}: ${e.message}`); }
    const actual = createHash('sha256').update(bytes).digest('hex');
    if (!pins[key] || pins[key] !== actual) {
      throw codedError('invalid_argument', 'loadWasm',
        `engine package integrity check failed: ${file} sha256 ${actual} does not match the MANIFEST pin ${pins[key] ?? '(no pin)'}`);
    }
  }
}

export async function loadWasm() {
  const jsPath = path.join(PKG_DIR, 'burn_research.js');
  const wasmPath = path.join(PKG_DIR, 'burn_research_bg.wasm');
  verifyEngineIntegritySync(PKG_DIR);
  const wasmBytes = await fs.promises.readFile(wasmPath);
  const rt = await import(pathToFileURL(jsPath).href);
  rt.initSync({ module: wasmBytes });
  return rt;
}

export const LAYER_TYPE = { linear: 1, norm: 2, conv: 3, embedding: 5 };

export const WEIGHTED_CONSTRUCTORS = {
  linear: 'linear',
  batchNorm: 'norm',
  groupNorm: 'norm',
  instanceNorm: 'norm',
  layerNorm: 'norm',
  rmsNorm: 'norm',
  conv1d: 'conv',
  conv2d: 'conv',
  convTranspose2d: 'conv',
  embedding: 'embedding',
};

export function buildGraph(rt, spec) {
  const argIssues = checkArgValues(spec);
  if (argIssues.length) {
    throw codedError('invalid_argument', 'buildGraph', `invalid spec arguments: ${argIssues.join('; ')}`);
  }
  const planCheck = analyzePlanPorts(spec);
  if (!planCheck.valid) {
    if (planCheck.code === 'plan_minimum_ports') {
      throw codedError('plan_minimum_ports', 'buildGraph',
        `plan requires at least ${planCheck.minimum_ports} external ports, got ${planCheck.declared_ports.length}`);
    }
    throw codedError('plan_unused_ports', 'buildGraph',
      `declared external input slots are not read before a graph write or selected as output: [${planCheck.unused_ports.join(', ')}]`);
  }
  const reg = new rt.LayerRegistry();
  const specs = [];
  const weighted = new Map();
  for (const [li, decl] of spec.layers.entries()) {
    if (typeof decl.constructor !== 'string' || !Object.hasOwn(rt.AgentLayerSpec, decl.constructor) || typeof rt.AgentLayerSpec[decl.constructor] !== 'function') {
      throw codedError('unknown_layer_constructor', 'buildGraph', `unknown layer constructor: ${decl.constructor}`);
    }
    const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
    if (typeof s !== 'object' || s === null) {
      throw codedError('unknown_layer_constructor', 'buildGraph', `unknown layer constructor: ${decl.constructor}`);
    }
    reg.initAgentLayer(s);
    specs.push(s);
    const wt = Object.hasOwn(WEIGHTED_CONSTRUCTORS, decl.constructor) ? WEIGHTED_CONSTRUCTORS[decl.constructor] : null;
    if (wt) weighted.set(li, { registryId: decl.args[0], type: LAYER_TYPE[wt] });
  }
  const b = new rt.AgentGraphBuilder(spec.numSlots);
  for (const step of spec.steps) {
    const s = specs[step.layer];
    if (!s) throw codedError('invalid_argument', 'buildGraph', `step references missing layer index ${step.layer}`);
    const slots = step.slots;
    if (step.kind === 'unary' && slots.length === 2) b.addUnary(s, slots[0], slots[1]);
    else if (step.kind === 'binary' && slots.length === 3) b.addBinary(s, slots[0], slots[1], slots[2]);
    else throw codedError('invalid_argument', 'buildGraph', `bad step (need unary:[a,b] or binary:[a,b,c]): ${JSON.stringify(step)}`);
  }
  b.setOutput(spec.outputSlot);
  const plan = new rt.MultiInputGraphPlan(b);
  for (const p of spec.ports) {
    plan.addInputPort(p.slot, p.role, ...p.shape, p.layout, false, 0n);
  }
  for (const [indexStr, values] of Object.entries(spec.weights ?? {})) {
    const w = weighted.get(Number(indexStr));
    if (!w) throw codedError('invalid_argument', 'buildGraph', `weights for layer index ${indexStr}: not a weighted layer`);
    reg.setWeightsFlat(w.registryId, w.type, Float32Array.from(values));
  }
  const g = reg.compileMultiInputGraph(plan);
  return { reg, plan, g };
}

export function makeBundle(rt, plan, inputsBySlot, portsBySlot) {
  if (!inputsBySlot || typeof inputsBySlot !== 'object' || Array.isArray(inputsBySlot)) {
    throw codedError('invalid_argument', 'makeBundle', 'case inputs must be an object mapping slot -> values array');
  }
  const bundle = new rt.MultiInputInputBundle(plan);
  for (const [slotStr, values] of Object.entries(inputsBySlot)) {
    if (!/^(?:0|[1-9]\d*)$/.test(slotStr)) {
      throw codedError('invalid_argument', 'makeBundle', `input key ${JSON.stringify(slotStr)} must be a canonical non-negative integer`);
    }
    const slot = Number(slotStr);
    const port = portsBySlot.get(slot);
    if (!port) throw codedError('invalid_argument', 'makeBundle', `input for slot ${slot} has no declared port`);
    if (!Array.isArray(values) || values.some((v) => !isFiniteF32(v))) {
      throw codedError('invalid_argument', 'makeBundle', `input for slot ${slot}: values must be an array of finite f32 numbers`);
    }
    const t = new rt.WasmTensor(Float32Array.from(values), Uint32Array.from(port.shape));
    const ok = bundle.bindInput(slot, t, port.role, port.layout, 'burn-math-skill', 1n, '');
    t.free();
    if (!ok) throw codedError('invalid_argument', 'makeBundle', `bindInput failed for slot ${slot}`);
  }
  return bundle;
}

// Scope arena for a whole graph (mirror of withTensorScope in engine.mjs,
// one level up: registry + plan + compiled graph + bundles).
//
// buildGraph's products live on the WASM heap; forgetting .free() leaks
// memory that no JS GC will reclaim. withGraphScope builds the graph,
// runs fn({reg, plan, g, bundle, track, release}), then .free()s every
// tracked object in LIFO order when fn settles — including when fn
// throws. Teardown never masks fn's error.
//
//   const values = await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
//     const b = bundle(spec.inputs, portBySlot(spec));
//     const out = g.run(reg, b);
//     const values = Array.from(out.to_array());
//     out.free();
//     return values;
//   }); // reg, plan, g, b are freed here, even on throw
//
// Known limit (pre-existing, shared with verify.mjs): if buildGraph
// itself throws mid-construction (e.g. an unknown constructor after
// some layers were initialized), the partially built registry has no
// handle to free. Making buildGraph exception-safe is a separate
// follow-up; this scope covers everything buildGraph returns.
export async function withGraphScope(rt, spec, fn) {
  const owned = [];
  const track = (obj) => {
    if (obj && typeof obj.free === 'function') owned.push(obj);
    return obj;
  };
  const release = (obj) => {
    const i = owned.lastIndexOf(obj);
    if (i >= 0) owned.splice(i, 1);
    return obj;
  };
  const { reg, plan, g } = buildGraph(rt, spec);
  track(reg);
  track(plan);
  track(g);
  try {
    return await fn({
      reg,
      plan,
      g,
      bundle: (inputs, ports) => track(makeBundle(rt, plan, inputs, ports)),
      track,
      release,
      get size() {
        return owned.length;
      },
    });
  } finally {
    for (let i = owned.length - 1; i >= 0; i--) {
      try {
        owned[i].free();
      } catch {
        // Already freed manually, or never fully constructed: never fail teardown.
      }
    }
  }
}
