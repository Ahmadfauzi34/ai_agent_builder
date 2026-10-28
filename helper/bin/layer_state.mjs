#!/usr/bin/env node
// burn-math layer_state: per-layer state inspection and transplant.
//
// The bundle checkpoint moves a whole program; this tool works one layer
// at a time, for debugging: which layer's state changed, and what happens
// if a saved state is loaded into another build of the same program.
//
//   node layer_state.mjs inspect <spec.json>
//     -> per layer: {layer_index, constructor, registry_id, type_code,
//        state_bytes, state_sha256, has_state, weights?, weight_layout?}
//        (state bytes are deterministic — ParamIds are re-keyed on export —
//        so digests are comparable across builds and sessions.)
//   node layer_state.mjs get <spec.json> <layer_index>
//     -> {ok, state_base64, state_sha256, state_bytes}
//   node layer_state.mjs set <spec.json> <layer_index> <state.b64>
//     -> passes the bytes through the decode guard first (refused when the
//        tier rejects them), loads them into a fresh build of the spec,
//        then runs the spec's inputs so the effect is observable:
//        {ok, applied, guard, outputs}
//
// Type codes are read from the engine's own program identity fingerprints
// (type=XX;id=N), never from a hand-maintained map. Parameter-free layers
// (binary/pool/shift/featureNorm families) report has_state:false with 0
// state bytes — the engine returns an empty state for them, not an error.
import fs from 'node:fs';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';

const [sub, specFile, arg3, arg4] = process.argv.slice(2);
if (!sub || !specFile || !['inspect', 'get', 'set'].includes(sub)) {
  console.error('usage: node layer_state.mjs inspect <spec.json> | get <spec.json> <layer_index> | set <spec.json> <layer_index> <state.b64>');
  process.exit(2);
}

let spec;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }

const { createHash } = await import('node:crypto');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

try {
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);

  // Authoritative per-layer type codes from program identity fingerprints.
  const identity = JSON.parse(g.programIdentity());
  const typeById = new Map();
  for (const fp of identity.layer_init_fingerprints ?? []) {
    const m = /^type=([0-9a-f]+);id=(\d+);/.exec(fp);
    if (m) typeById.set(Number(m[2]), parseInt(m[1], 16));
  }
  const layerInfo = (index) => {
    const decl = spec.layers[index];
    if (!decl) throw new Error(`no layer at index ${index}`);
    const id = decl.args[0];
    const type = typeById.get(id);
    if (type === undefined) throw new Error(`no identity fingerprint for layer id ${id}`);
    return { decl, id, type };
  };

  if (sub === 'inspect') {
    const layers = spec.layers.map((decl, index) => {
      const { id, type } = layerInfo(index);
      const state = reg.getLayerState(id, type);
      const entry = {
        layer_index: index,
        constructor: decl.constructor,
        registry_id: id,
        type_code: type,
        state_bytes: state.length,
        state_sha256: sha256(state),
        has_state: state.length > 0,
      };
      try {
        const w = reg.getWeightsFlat(id, type);
        if (w.length > 0) {
          entry.weights = Array.from(w);
          entry.weight_layout = JSON.parse(reg.weightLayout(id, type));
        }
      } catch { /* parameter-free or no flat weights */ }
      return entry;
    });
    printJsonAndExit({ ok: true, layers }, 0);
  }

  const index = Number(arg3);
  if (!Number.isInteger(index)) failCli('layer_index must be an integer');
  const { id, type } = layerInfo(index);

  if (sub === 'get') {
    const state = reg.getLayerState(id, type);
    printJsonAndExit({
      ok: true,
      layer_index: index,
      state_bytes: state.length,
      state_sha256: sha256(state),
      state_base64: Buffer.from(state).toString('base64'),
    }, 0);
  }

  // set
  if (!arg4) failCli('usage: node layer_state.mjs set <spec.json> <layer_index> <state.b64>');
  let b64;
  try { b64 = fs.readFileSync(arg4, 'utf8').trim(); }
  catch (e) { failCli(`cannot read state file: ${e.message}`); }
  const bytes = Buffer.from(b64, 'base64');
  const guard = JSON.parse(rt.decodeGuardCheck(bytes));
  if (!guard.accepted) {
    printJsonAndExit({ ok: false, applied: false, guard }, 2);
  }
  reg.loadLayerState(id, type, bytes);
  const bundle = makeBundle(rt, plan, spec.inputs, portBySlot(spec));
  const out = g.run(reg, bundle);
  const values = Array.from(out.to_array());
  out.free();
  printJson({
    ok: true,
    applied: true,
    layer_index: index,
    state_bytes: bytes.length,
    state_sha256: sha256(bytes),
    guard,
    outputs: { values },
  });
} catch (e) {
  failCli(e.message, e.envelope ?? null);
}
