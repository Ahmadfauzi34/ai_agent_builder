// Spec format for the wasm-opt helper CLIs (see ../README.md).
// Loads a JSON spec file and normalizes the optional sections.
import fs from 'node:fs';

export function loadSpec(file) {
  let raw;
  try { raw = fs.readFileSync(file, 'utf8'); }
  catch (e) { throw new Error(`cannot read spec file ${file}: ${e.message}`); }
  let spec;
  try { spec = JSON.parse(raw); }
  catch (e) { throw new Error(`spec file ${file} is not valid JSON: ${e.message}`); }
  for (const key of ['numSlots', 'layers', 'steps', 'outputSlot', 'ports']) {
    if (spec[key] === undefined) throw new Error(`spec missing required key: ${key}`);
  }
  if (!Array.isArray(spec.layers) || !Array.isArray(spec.steps) || !Array.isArray(spec.ports)) {
    throw new Error('spec.layers / spec.steps / spec.ports must be arrays');
  }
  spec.weights = spec.weights ?? {};
  spec.inputs = spec.inputs ?? {};
  return spec;
}

// Port lookup by slot, for binding inputs with the declared shape/role/layout.
export function portBySlot(spec) {
  const m = new Map();
  for (const p of spec.ports) m.set(p.slot, p);
  return m;
}
