// Engine loading + graph building for the wasm-opt helper.
//
// Loads the repo's own WASM package: <repo>/pkg/ as produced by
// `wasm-pack build --target web --out-dir pkg` (see .github/workflows/main.yml).
// Override with BURN_WASM_PKG_DIR to point at a different build.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';
import { webcrypto } from 'node:crypto';

// Node 18 ESM quirk (lane #2, 2026-10-28): in Node 18, globalThis.crypto
// is present in CJS contexts but UNDEFINED inside ESM modules — and the
// wasm glue calls globalThis.crypto.getRandomValues for thread_rng
// (ParamId generation on first layer creation). Without this line, every
// engine-path forward under Node 18 panics with RuntimeError: unreachable
// at the first buildGraph. wasm_direct.mjs already had this polyfill,
// which is why verify passed under Node 18 while prove-rules trapped.
// Node 24+ has crypto in ESM, so this is a no-op there.
globalThis.crypto ??= webcrypto;

const HERE = path.dirname(fileURLToPath(import.meta.url));
// Engine package dir resolution (in order):
//   1. BURN_WASM_PKG_DIR env override (explicit).
//   2. CI-bundle layout: this file lives at <pkg>/helper/lib/, so the
//      engine files are two levels up (<pkg>/burn_research_bg.wasm).
//   3. Built-pack layout: this file lives at <pack>/lib/, so the engine
//      is the sibling <pack>/engine-audit/ (the builder copies the full
//      bundle there). WITHOUT this candidate, a standalone pack user
//      running any shape/engine tool without the env var got a crash:
//      catalogPath() fell back to a nonexistent parent-of-pack docs dir
//      (proven by an isolated built-pack regression run, 2026-10-28).
//   4. Repo source layout: <repo>/helper/lib/, engine at
//      <repo>/pkg/ (as produced by `wasm-pack build --target web --out-dir pkg`).
function resolvePkgDir() {
  if (process.env.BURN_WASM_PKG_DIR) return path.resolve(process.env.BURN_WASM_PKG_DIR);
  const packed = path.join(HERE, '..', '..');
  if (fs.existsSync(path.join(packed, 'burn_research_bg.wasm'))) return packed;
  // Same engine.mjs serves all layouts so the repo stays the single
  // source of truth for the pack it builds.
  const packEngine = path.join(packed, 'engine-audit');
  if (fs.existsSync(path.join(packEngine, 'burn_research_bg.wasm'))) return packEngine;
  const siblingEngine = path.join(HERE, '..', 'engine-audit');
  if (fs.existsSync(path.join(siblingEngine, 'burn_research_bg.wasm'))) return siblingEngine;
  return path.join(HERE, '..', '..', 'pkg');
}
const PKG_DIR = resolvePkgDir();

export function pkgDir() {
  return PKG_DIR;
}

// Location of the host runner script (spawned by lib/runner.mjs).
// Packed layout: <pkg>/interactive_multi_input_ingress.mjs.
// Repo source layout: <repo>/scripts/interactive_multi_input_ingress.mjs.
export function runnerPath() {
  const packed = path.join(PKG_DIR, 'interactive_multi_input_ingress.mjs');
  if (fs.existsSync(packed)) return packed;
  return path.join(HERE, '..', '..', 'scripts', 'interactive_multi_input_ingress.mjs');
}

// Location of the agent layer catalog (read by lib/rewrites.mjs, lib/shapes.mjs).
// Packed layout: <pkg>/docs/agent-layer-catalog.v1.json.
// Repo source layout: <repo>/docs/contracts/agent-layer-catalog.v1.json.
export function catalogPath() {
  const packed = path.join(PKG_DIR, 'docs', 'agent-layer-catalog.v1.json');
  if (fs.existsSync(packed)) return packed;
  return path.join(HERE, '..', '..', 'docs', 'contracts', 'agent-layer-catalog.v1.json');
}

// Same fallback pattern as catalogPath(), for the layout contracts doc
// (used by validate's layout-compatibility warnings and bin/layout.mjs).
export function layoutContractsPath() {
  const packed = path.join(PKG_DIR, 'docs', 'agent-layout-contracts.v1.json');
  if (fs.existsSync(packed)) return packed;
  return path.join(HERE, '..', '..', 'docs', 'contracts', 'agent-layout-contracts.v1.json');
}

export async function loadWasm() {
  const jsPath = path.join(PKG_DIR, 'burn_research.js');
  const wasmPath = path.join(PKG_DIR, 'burn_research_bg.wasm');
  const [rt, bytes] = await Promise.all([
    import(pathToFileURL(jsPath).href),
    fs.promises.readFile(wasmPath),
  ]);
  rt.initSync({ module: bytes });
  return rt;
}

// Numeric layer-type codes (src/protocol.rs); these are the types whose
// weights setWeightsFlat supports.
export const LAYER_TYPE = { linear: 1, norm: 2, conv: 3, embedding: 5 };

// Spec constructor name -> LAYER_TYPE key, for the weighted layer types.
const WEIGHTED_CONSTRUCTORS = {
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

// Build {reg, plan, g} from a spec (layers/steps/ports/weights).
// Weight layout is column-major (audited).
export function buildGraph(rt, spec) {
  const reg = new rt.LayerRegistry();
  const specs = [];
  const weighted = new Map(); // layer index -> {registryId, type}
  for (const [index, decl] of spec.layers.entries()) {
    const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
    if (typeof s !== 'object' || s === null) {
      throw new Error(`unknown layer constructor: ${decl.constructor}`);
    }
    reg.initAgentLayer(s);
    specs.push(s);
    const wt = WEIGHTED_CONSTRUCTORS[decl.constructor];
    if (wt) weighted.set(index, { registryId: decl.args[0], type: LAYER_TYPE[wt] });
  }
  const b = new rt.AgentGraphBuilder(spec.numSlots);
  for (const step of spec.steps) {
    const s = specs[step.layer];
    if (!s) throw new Error(`step references missing layer index ${step.layer}`);
    const slots = step.slots;
    if (step.kind === 'unary' && slots.length === 2) b.addUnary(s, slots[0], slots[1]);
    else if (step.kind === 'binary' && slots.length === 3) b.addBinary(s, slots[0], slots[1], slots[2]);
    else throw new Error(`bad step (need unary:[a,b] or binary:[a,b,c]): ${JSON.stringify(step)}`);
  }
  b.setOutput(spec.outputSlot);
  const plan = new rt.MultiInputGraphPlan(b);
  for (const p of spec.ports) {
    plan.addInputPort(p.slot, p.role, ...p.shape, p.layout, false, 0n);
  }
  // Deterministic zero-init is the audited default; explicit weights override.
  for (const [indexStr, values] of Object.entries(spec.weights ?? {})) {
    const w = weighted.get(Number(indexStr));
    if (!w) throw new Error(`weights for layer index ${indexStr}: not a weighted layer`);
    reg.setWeightsFlat(w.registryId, w.type, Float32Array.from(values));
  }
  const g = reg.compileMultiInputGraph(plan);
  return { reg, plan, g };
}

export function makeBundle(rt, plan, inputsBySlot, portsBySlot) {
  const bundle = new rt.MultiInputInputBundle(plan);
  for (const [slotStr, values] of Object.entries(inputsBySlot)) {
    const slot = Number(slotStr);
    const port = portsBySlot.get(slot);
    if (!port) throw new Error(`input for slot ${slot} has no declared port`);
    const t = new rt.WasmTensor(Float32Array.from(values), Uint32Array.from(port.shape));
    const ok = bundle.bindInput(slot, t, port.role, port.layout, 'wasm-opt', 1n, '');
    t.free();
    if (!ok) throw new Error(`bindInput failed for slot ${slot}`);
  }
  return bundle;
}

// ---------------------------------------------------------------------------
// Agent-hardening helpers.
// ---------------------------------------------------------------------------

// mathCheckOperation with forgiving defaults.
//
// The raw wasm-bindgen export requires all 5 arguments; calling it with only
// (opId, lhsShape, rhsShape) throws
//   TypeError: Cannot read properties of undefined (reading 'length')
// from passArray32ToWasm0. This wrapper defaults u32_params/f32_params to
// empty arrays and accepts plain JS arrays for every shape/param argument.
// Returns the parsed preflight object (the engine returns a JSON string).
// Throws the engine's message for unknown operation ids.
export function checkOperation(rt, opId, lhsShape, rhsShape, u32Params, f32Params) {
  const u32 = u32Params === undefined || u32Params === null
    ? new Uint32Array(0)
    : Uint32Array.from(u32Params);
  const f32 = f32Params === undefined || f32Params === null
    ? new Float32Array(0)
    : Float32Array.from(f32Params);
  // Null/undefined shapes degrade to [] so the engine returns a structured
  // "rejected" verdict instead of a raw TypeError from Uint32Array.from
  // (same hardening PR #387 applied to the params above; shapes were missed).
  const raw = rt.mathCheckOperation(
    opId,
    Uint32Array.from(lhsShape ?? []),
    Uint32Array.from(rhsShape ?? []),
    u32,
    f32,
  );
  return JSON.parse(raw);
}

// Scope arena for WasmTensor.
//
// WasmTensor objects live on the WASM heap; forgetting .free() leaks memory
// that no JS GC will reclaim. withTensorScope runs fn(scope) and .free()s
// every tensor created via scope.tensor() (or registered via scope.track())
// when fn settles — including when fn throws. LIFO order.
//
//   const out = await withTensorScope(rt, async (scope) => {
//     const a = scope.tensor([1, 2, 3], [1, 3, 1, 1]);
//     const b = scope.tensor([4, 5, 6], [1, 3, 1, 1]);
//     ... compute, return plain values (not tensors) ...
//     return [4.5, 10.5];
//   }); // a and b are freed here, even on throw
//
// scope.release(t) untracks a tensor (e.g. to return it to the caller —
// returning a tracked tensor without release would hand back freed memory).
export async function withTensorScope(rt, fn) {
  const owned = [];
  const scope = {
    tensor(values, shape) {
      const t = new rt.WasmTensor(Float32Array.from(values), Uint32Array.from(shape));
      owned.push(t);
      return t;
    },
    track(t) {
      owned.push(t);
      return t;
    },
    release(t) {
      const i = owned.lastIndexOf(t);
      if (i >= 0) owned.splice(i, 1);
      return t;
    },
    get size() {
      return owned.length;
    },
  };
  try {
    return await fn(scope);
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
