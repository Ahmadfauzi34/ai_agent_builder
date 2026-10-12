// Engine loading + graph building for the wasm-opt helper.
//
// Loads the repo's own WASM package: <repo>/pkg/ as produced by
// `wasm-pack build --target web --out-dir pkg` (see .github/workflows/main.yml).
// Override with BURN_WASM_PKG_DIR to point at a different build.
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
// Engine package dir resolution (in order):
//   1. BURN_WASM_PKG_DIR env override (explicit).
//   2. Packed layout: this file lives at <pkg>/wasm-opt/lib/, so the
//      engine files are two levels up (<pkg>/burn_research_bg.wasm).
//   3. Repo source layout: <repo>/scripts/wasm-opt/lib/, engine at
//      <repo>/pkg/ (as produced by `wasm-pack build --target web --out-dir pkg`).
function resolvePkgDir() {
  if (process.env.BURN_WASM_PKG_DIR) return path.resolve(process.env.BURN_WASM_PKG_DIR);
  const packed = path.join(HERE, '..', '..');
  if (fs.existsSync(path.join(packed, 'burn_research_bg.wasm'))) return packed;
  return path.join(HERE, '..', '..', '..', 'pkg');
}
const PKG_DIR = resolvePkgDir();

export function pkgDir() {
  return PKG_DIR;
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

// Numeric layer-type codes; linear == 1.
export const LAYER_TYPE = { linear: 1 };

// Build {reg, plan, g} from a spec (layers/steps/ports/weights).
// Weight layout is column-major (audited).
export function buildGraph(rt, spec) {
  const reg = new rt.LayerRegistry();
  const specs = [];
  const weighted = []; // {registryId, type, values}
  for (const decl of spec.layers) {
    const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
    if (typeof s !== 'object' || s === null) {
      throw new Error(`unknown layer constructor: ${decl.constructor}`);
    }
    reg.initAgentLayer(s);
    specs.push(s);
    if (decl.constructor === 'linear') {
      weighted.push({ registryId: decl.args[0], type: LAYER_TYPE.linear });
    }
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
    const w = weighted[Number(indexStr)];
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
