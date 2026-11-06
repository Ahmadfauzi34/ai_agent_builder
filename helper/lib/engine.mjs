// Engine loading + graph building for the wasm-opt helper.
//
// Loads the repo's own WASM package: <repo>/pkg/ as produced by
// `wasm-pack build --target web --out-dir pkg` (see .github/workflows/main.yml).
// Override with BURN_WASM_PKG_DIR to point at a different build.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';
import { createHash, webcrypto } from 'node:crypto';

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

function verifyPkgIntegritySync(dir = PKG_DIR) {
  const manifestCandidates = [
    path.join(HERE, '..', 'MANIFEST.json'),
    path.join(HERE, '..', '..', 'MANIFEST.json'),
  ];
  let pins = null;
  for (const mf of manifestCandidates) {
    if (fs.existsSync(mf)) {
      try {
        const parsed = JSON.parse(fs.readFileSync(mf, 'utf8'));
        pins = parsed?.sha256 ?? null;
        if (pins) break;
      } catch { /* ignore */ }
    }
  }
  if (!pins) return;
  for (const [file, key] of [
    [path.join(dir, 'burn_research_bg.wasm'), 'burn_research_bg.wasm'],
    [path.join(dir, 'burn_research.js'), 'burn_research.js'],
  ]) {
    let bytes;
    try { bytes = fs.readFileSync(file); }
    catch (e) { throw new Error(`cannot read engine file ${file}: ${e.message}`); }
    const actual = createHash('sha256').update(bytes).digest('hex');
    if (pins[key] && pins[key] !== actual) {
      throw new Error(`engine package integrity check failed: ${file} sha256 ${actual} does not match MANIFEST pin ${pins[key]}`);
    }
  }
}

export async function loadWasm() {
  const jsPath = path.join(PKG_DIR, 'burn_research.js');
  const wasmPath = path.join(PKG_DIR, 'burn_research_bg.wasm');
  verifyPkgIntegritySync(PKG_DIR);
  const [rt, bytes] = await Promise.all([
    import(pathToFileURL(jsPath).href),
    fs.promises.readFile(wasmPath),
  ]);
  rt.initSync({ module: bytes });
  return rt;
}

// Numeric layer-type codes (src/protocol.rs); these are the types whose
// weights setWeightsFlat supports.
// Tahap 2a+2b: codedError — a plain Error carries no machine code, so
// the runner wrapped buildGraph failures in an `internal_error`
// envelope (the regression "buildGraph exception-safe" test failed on
// exactly that for unknown constructors). With .code/.phase, the
// envelope names the failure. Defined once, here: the core lives in
// this file, and wasm_direct.mjs imports it for the gated wrapper.
// (Port delta C6: engine.mjs must NOT import isFiniteF32/isU32 from
// spec.mjs — spec imports catalogPath from engine, so that import
// would create an engine <-> spec cycle. The two one-line predicates
// are inlined here; spec.mjs stays the canonical definition, and the
// folded regression test pins the f32-overflow boundary (1e39).)
export function codedError(code, phase, message) {
  const e = new Error(message);
  e.code = code;
  e.phase = phase;
  return e;
}

// Local copies of spec.mjs's predicates (see C6 above).
const isFiniteF32 = (v) => typeof v === 'number' && Number.isFinite(v) && Number.isFinite(Math.fround(v));
const isU32 = (d) => Number.isInteger(d) && d > 0;

export const LAYER_TYPE = { linear: 1, norm: 2, conv: 3, embedding: 5 };

// Spec constructor name -> LAYER_TYPE key, for the weighted layer types.
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

function freeQuiet(obj) {
  if (obj && typeof obj.free === 'function') {
    try { obj.free(); } catch { /* already freed: teardown never fails */ }
  }
}

// Tahap 2b: buildGraph IS the ungated core — the single shared build
// loop. No host-policy gates here (no checkArgValues/analyzePlanPorts/
// checkSpec): lib/recurse, lib/quant_budget and lib/harness import
// this buildGraph, and a 1-port solve-gd spec or a harmless duplicate
// registry id are legitimate engine graphs. The gated buildGraph
// (gates + this core) lives in wasm_direct.mjs for interactive callers.
// Build {reg, plan, g} from a spec (layers/steps/ports/weights).
// Weight layout is column-major (audited).
// Exception-safe and leak-free: temporary AgentLayerSpec and AgentGraphBuilder
// handles are freed before returning, and reg/plan are freed if construction throws.
export function buildGraph(rt, spec) {
  const reg = new rt.LayerRegistry();
  const specs = [];
  const weighted = new Map(); // layer index -> {registryId, type}
  let b = null;
  let plan = null;
  try {
    for (const [index, decl] of spec.layers.entries()) {
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
      if (wt) weighted.set(index, { registryId: decl.args[0], type: LAYER_TYPE[wt] });
    }
    b = new rt.AgentGraphBuilder(spec.numSlots);
    for (const step of spec.steps) {
      const s = specs[step.layer];
      if (!s) throw codedError('invalid_argument', 'buildGraph', `step references missing layer index ${step.layer}`);
      const slots = step.slots;
      if (step.kind === 'unary' && slots.length === 2) b.addUnary(s, slots[0], slots[1]);
      else if (step.kind === 'binary' && slots.length === 3) b.addBinary(s, slots[0], slots[1], slots[2]);
      else throw codedError('invalid_argument', 'buildGraph', `bad step (need unary:[a,b] or binary:[a,b,c]): ${JSON.stringify(step)}`);
    }
    b.setOutput(spec.outputSlot);
    plan = new rt.MultiInputGraphPlan(b);
    for (const p of spec.ports) {
      plan.addInputPort(p.slot, p.role, ...p.shape, p.layout, false, 0n);
    }
    // Deterministic zero-init is the audited default; explicit weights override.
    for (const [indexStr, values] of Object.entries(spec.weights ?? {})) {
      const w = weighted.get(Number(indexStr));
      if (!w) throw codedError('invalid_argument', 'buildGraph', `weights for layer index ${indexStr}: not a weighted layer`);
      reg.setWeightsFlat(w.registryId, w.type, Float32Array.from(values));
    }
    const g = reg.compileMultiInputGraph(plan);
    return { reg, plan, g };
  } catch (e) {
    freeQuiet(plan);
    freeQuiet(reg);
    // PR Tahap 1: normalize a raw string throw (wasm-bindgen
    // Result<T, String>) to an Error before rethrowing, so downstream
    // catches see e.message instead of undefined.
    throw e instanceof Error ? e : new Error(String(e));
  } finally {
    freeQuiet(b);
    for (let i = specs.length - 1; i >= 0; i--) freeQuiet(specs[i]);
  }
}

// Tahap 2b: makeBundle IS the lenient core — it accepts BOTH input
// forms (an object mapping slot -> values AND an array whose index is
// the slot; the lib fit path passes case.inputs in array form). The
// strict object-only check lives in the wasm_direct.mjs wrapper, for
// interactive callers only. Length + finite-f32 checks with slot
// attribution run here, BEFORE the WASM call.
export function makeBundle(rt, plan, inputsBySlot, portsBySlot) {
  const bundle = new rt.MultiInputInputBundle(plan);
  try {
    for (const [slotStr, values] of Object.entries(inputsBySlot)) {
      if (!/^(?:0|[1-9]\d*)$/.test(slotStr)) {
        throw codedError('invalid_argument', 'makeBundle', `input key ${JSON.stringify(slotStr)} must be a canonical non-negative integer`);
      }
      const slot = Number(slotStr);
      const port = portsBySlot.get(slot);
      if (!port) throw codedError('invalid_argument', 'makeBundle', `input for slot ${slot} has no declared port`);
      // Tahap 2b: input-length check with slot attribution, BEFORE the
      // WASM call — the bare WasmTensor::new error names the shape but
      // not the slot. Same shape guard as checkInputLengths: only when
      // the port shape is a valid 4-D positive-u32 array.
      if (Array.isArray(port.shape) && port.shape.length === 4 && port.shape.every((d) => isU32(d) && d > 0)) {
        const numel = port.shape[0] * port.shape[1] * port.shape[2] * port.shape[3];
        if (!Array.isArray(values) || values.length !== numel) {
          throw codedError('invalid_argument', 'makeBundle',
            `inputs slot ${slot}: expected ${numel} values for shape ${JSON.stringify(port.shape)}, got ${Array.isArray(values) ? values.length : 'non-array'}`);
        }
      }
      if (!Array.isArray(values) || values.some((v) => !isFiniteF32(v))) {
        throw codedError('invalid_argument', 'makeBundle', `input for slot ${slot}: values must be an array of finite f32 numbers`);
      }
      const t = new rt.WasmTensor(Float32Array.from(values), Uint32Array.from(port.shape));
      const ok = bundle.bindInput(slot, t, port.role, port.layout, 'burn-math-skill', 1n, '');
      t.free();
      if (!ok) throw codedError('invalid_argument', 'makeBundle', `bindInput failed for slot ${slot}`);
    }
    return bundle;
  } catch (e) {
    freeQuiet(bundle);
    // PR Tahap 1: normalize a raw string throw (wasm-bindgen
    // Result<T, String>) to an Error before rethrowing, so downstream
    // catches see e.message instead of undefined.
    throw e instanceof Error ? e : new Error(String(e));
  }
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
