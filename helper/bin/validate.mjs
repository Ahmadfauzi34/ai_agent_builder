#!/usr/bin/env node
// burn-math validate: static + engine plan check. No session is created.
// Usage: node validate.mjs <spec.json>
// Exit 0 + valid:true when layers 1 and 2 pass; exit 2 otherwise, with
// the failing layer's structured detail printed as JSON.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { printJsonAndExit, failCli } from '../lib/runner.mjs';
import { loadSpec, checkArgValues, analyzePlanPorts, expectedWeightCount } from '../lib/spec.mjs';
import { checkSpec } from '../lib/shapecore.mjs';
import { verifyEngineIntegritySync } from '../lib/wasm_direct.mjs';
import { catalogPath, layoutContractsPath } from '../lib/engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(HERE, '..');
const args = process.argv.slice(2);
const [specFile] = args;
if (!specFile || args.includes('--help') || args.includes('-h')) {
  console.error('usage: node validate.mjs <spec.json>');
  process.exit(2);
}

let spec;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e instanceof Error ? e.message : String(e)); }

try { verifyEngineIntegritySync(); }
catch (e) { failCli(e instanceof Error ? e.message : String(e)); }

// --- Layer 1a: constructor existence. ---
// Engine contracts resolve through engine.mjs: built-pack docs first,
// then the checked-in docs/contracts/ copies (the repo's single source).
let catalog, layoutDoc;
try {
  catalog = JSON.parse(fs.readFileSync(catalogPath(), 'utf8'));
  layoutDoc = JSON.parse(fs.readFileSync(layoutContractsPath(), 'utf8'));
} catch (e) {
  failCli(`cannot read engine contracts: ${e instanceof Error ? e.message : String(e)}`);
}
const known = new Set(Object.keys(catalog.constructors ?? {}));
const unknownConstructors = [...new Set(spec.layers.map((l) => (l && typeof l === 'object' ? l.constructor : null)).filter((c) => typeof c !== 'string' || !known.has(c)))];
if (unknownConstructors.length) {
  printJsonAndExit({
    ok: false, valid: false, stage: 'static', code: 'unknown_layer_constructor',
    unknownConstructors,
    remediation_hint: 'run: node helper/bin/find-op.mjs --list and use only constructors from that list',
  }, 2);
}

// --- Layer 1c: strict arg/value typing. ---
const argIssues = checkArgValues(spec, catalog);
if (argIssues.length) {
  printJsonAndExit({
    ok: false, valid: false, stage: 'static', code: 'invalid_argument',
    issues: argIssues,
    remediation_hint: 'layer args must match the catalog parameters: required-arg counts, u32 = integer in 0..4294967295, bool = real boolean, f32/f64 = finite number, null only for Option params, registry ids unique; weights/inputs are finite f32 arrays with canonical integer keys; port shapes are 4-D positive integers',
  }, 2);
}

// --- Layer 1b: static shape/dataflow check. ---
const checked = checkSpec(spec);
if (checked.mismatches.length || !checked.dataflow.ok) {
  printJsonAndExit({
    ok: false, valid: false, stage: 'static', code: 'static_check_failed',
    mismatches: checked.mismatches, dataflow: checked.dataflow,
    warnings: checked.warnings,
    remediation_hint: 'fix the listed steps/slots/shapes, then re-run validate',
  }, 2);
}

// Static warnings: port role/layout vocabularies, layout compatibility edges,
// int* constructors in MultiInputGraphPlan, and weights/inputs cardinality.
const CANONICAL_ROLES = new Set(['observation', 'state', 'feature', 'candidate', 'parameter', 'reward', 'context']);
const EXTERNAL_LAYOUTS = new Set([
  'unknown', 'any_rank4', 'feature_axis1_singleton', 'channel_first',
  'channel_first_singleton_width', 'feature_last',
  'token_ids_axis1_singleton', 'sequence_feature_axis2_singleton_width',
]);
const ENGINE_REJECTED_EXTERNAL_LAYOUTS = new Set(['dynamic', 'preserve_input']);
const INT_CONSTRUCTORS = new Set(['intAdd', 'intSub', 'intMul', 'intMatmul', 'intToFloat', 'intFromFloat']);

function isRoleValid(role) {
  if (typeof role !== 'string' || role.length === 0 || Buffer.byteLength(role) > 64) return false;
  if (CANONICAL_ROLES.has(role)) return true;
  return /^x-[a-z0-9\-_.]+$/.test(role);
}

const staticWarnings = [...checked.warnings];
const portMap = new Map();
const slotLayouts = new Map();
for (const p of spec.ports) {
  if (!p || typeof p !== 'object') continue;
  portMap.set(p.slot, p);
  slotLayouts.set(p.slot, p.layout);
  if (!isRoleValid(p.role)) {
    staticWarnings.push(`port slot ${p.slot}: unknown or invalid role '${p.role}' (invalid_enum at create)`);
  }
  if (ENGINE_REJECTED_EXTERNAL_LAYOUTS.has(p.layout)) {
    staticWarnings.push(`port slot ${p.slot}: layout '${p.layout}' is rejected by the engine on external ports (invalid_enum at create)`);
  } else if (!EXTERNAL_LAYOUTS.has(p.layout)) {
    staticWarnings.push(`port slot ${p.slot}: unknown external layout '${p.layout}' (invalid_enum at create)`);
  }
}

// Check int* constructors and layout propagation compatibility
const layoutCtors = layoutDoc?.constructors ?? {};
const layoutCompat = layoutDoc?.compatibility ?? {};
spec.steps.forEach((st, si) => {
  const decl = spec.layers[st.layer];
  if (!decl) return;
  const ctor = decl.constructor;
  if (INT_CONSTRUCTORS.has(ctor)) {
    staticWarnings.push(`step ${si} (${ctor}): int_binary/int_cast layers operate on WasmIntTensor (i64) via forwardIntBinary/forwardIntCast and cannot execute inside MultiInputGraphPlan f32 graphs`);
  }
  const lc = Object.hasOwn(layoutCtors, ctor) ? layoutCtors[ctor] : null;
  if (!lc) return;
  const inSlots = st.kind === 'binary' ? st.slots.slice(0, 2) : st.slots.slice(0, 1);
  const outSlot = st.slots[st.slots.length - 1];
  for (const s of inSlots) {
    const srcLayout = slotLayouts.get(s);
    if (srcLayout && Object.hasOwn(layoutCompat, srcLayout) && lc.input !== 'any_rank4' && lc.input !== 'dynamic') {
      const comp = layoutCompat[srcLayout];
      const accepted = [...(comp.accepts_consumers ?? []), ...(comp.conditional_consumers ?? [])];
      if (!accepted.includes(lc.input)) {
        staticWarnings.push(`step ${si} (${ctor}): slot ${s} layout '${srcLayout}' is incompatible with required input layout '${lc.input}'`);
      }
    }
  }
  if (lc.output === 'preserve_input') {
    slotLayouts.set(outSlot, slotLayouts.get(inSlots[0]) ?? 'unknown');
  } else {
    slotLayouts.set(outSlot, lc.output);
  }
});

for (const [layerIdxStr, values] of Object.entries(spec.weights ?? {})) {
  const li = Number(layerIdxStr);
  const decl = spec.layers[li];
  if (!decl) {
    staticWarnings.push(`weights key '${layerIdxStr}': layer index ${li} does not exist (0..${spec.layers.length - 1})`);
  } else {
    // v2.5 (F4): exact weight-count cross-check for all 10 weighted
    // constructors (was: linear only). Warnings, not errors: a spec
    // without weights is still valid (the engine supplies defaults).
    const expected = expectedWeightCount(decl);
    if (expected === null) {
      staticWarnings.push(`weights key '${layerIdxStr}': layer ${li} (${decl.constructor}) carries no mutable weights`);
    } else if (Array.isArray(values) && values.length !== expected) {
      staticWarnings.push(`weights layer ${li} (${decl.constructor}): expected ${expected} floats, got ${values.length}`);
    }
  }
}

for (const [slotStr, values] of Object.entries(spec.inputs ?? {})) {
  const slot = Number(slotStr);
  const port = portMap.get(slot);
  if (!port) {
    staticWarnings.push(`inputs slot ${slotStr}: no declared port for slot ${slotStr}`);
  } else if (Array.isArray(port.shape) && port.shape.length === 4 && Array.isArray(values)) {
    const expectedLen = port.shape[0] * port.shape[1] * port.shape[2] * port.shape[3];
    if (values.length !== expectedLen) {
      staticWarnings.push(`inputs slot ${slot}: expected ${expectedLen} values for shape ${JSON.stringify(port.shape)}, got ${values.length}`);
    }
  }
}

// --- Layer 2: engine port check (in-process, session-free). ---
const result = analyzePlanPorts(spec);
if (!result.valid) {
  printJsonAndExit({
    ok: false, valid: false, stage: 'plan', code: result.code ?? 'plan_check_failed',
    minimum_ports: result.minimum_ports, declared_ports: result.declared_ports,
    consumed_ports: result.consumed_ports, unused_ports: result.unused_ports,
    remediation_hint: 'declare at least 2 external ports and make every declared port consumed by a step',
  }, 2);
}
printJsonAndExit({ ok: true, ...result, ...(staticWarnings.length ? { static_warnings: staticWarnings } : {}) }, 0);
