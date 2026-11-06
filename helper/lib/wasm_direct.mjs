// Shared WASM-direct helper for graph-vs-graph work (used by verify.mjs and checkpoint.mjs).
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { checkArgValues, analyzePlanPorts } from './spec.mjs';
import { checkSpec } from './shapecore.mjs';
import { buildGraph as buildGraphCore, makeBundle as makeBundleCore, codedError, pkgDir } from './engine.mjs';
import { installMemoryOracle } from './opt.mjs';
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
  installMemoryOracle();
  rt.initSync({ module: wasmBytes });
  return rt;
}

export { LAYER_TYPE, WEIGHTED_CONSTRUCTORS } from './engine.mjs';

// The gated buildGraph for interactive callers (Tahap 2b): the three
// gates below are the same checks run.mjs and the 2a per-bin gates
// apply, now shared by every interactive caller of wasm_direct.
// They do NOT apply to the internal/lib path: lib/recurse,
// lib/quant_budget and lib/harness import buildGraph from engine.mjs,
// which is the ungated core (a 1-port solve-gd spec and a harmless
// duplicate registry id are legitimate engine graphs).
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
  const checked = checkSpec(spec);
  if (checked.mismatches.length || !checked.dataflow.ok) {
    throw codedError('static_check_failed', 'buildGraph',
      checked.mismatches[0]?.message ?? checked.dataflow.issues[0] ?? 'static check failed');
  }
  return buildGraphCore(rt, spec);
}

// The strict makeBundle for interactive callers: case inputs must be
// an object mapping slot -> values array (canonical behavior). The
// lib path (engine.mjs) is the lenient core instead (it also accepts
// array-form case inputs, index = slot).
export function makeBundle(rt, plan, inputsBySlot, portsBySlot) {
  if (!inputsBySlot || typeof inputsBySlot !== 'object' || Array.isArray(inputsBySlot)) {
    throw codedError('invalid_argument', 'makeBundle', 'case inputs must be an object mapping slot -> values array');
  }
  return makeBundleCore(rt, plan, inputsBySlot, portsBySlot);
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
// buildGraph itself is exception-safe (2026-10-30): a mid-construction
// throw frees the partial registry, specs, builder, and plan inside
// buildGraph, so this scope only ever covers what buildGraph returns.
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
