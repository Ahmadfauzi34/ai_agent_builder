#!/usr/bin/env node
// surface.mjs — return ONE wasm surface entry, complete.
// Usage:
//   node helper/bin/surface.mjs <name>   -> function, class, or method entry as JSON
//   node helper/bin/surface.mjs --list   -> one line per entry:
//                                    "function <name>" or "class <name>"
// Exit 0 on found, 2 on not found / bad usage.
// This is the only supported way for an agent to read the wasm surface.
// Do NOT open the engine docs/wasm-surface.v1.json,
// wasm-surface.bindings.actual.json, or burn_research.d.ts directly;
// this tool reads them
// programmatically and prints one entry. Signatures come from the .d.ts
// (the generated type file); method lists come from the bindings snapshot.
// v2.1: method queries now pair each signature with its class. The old
// version printed a flat, truncated list of signatures with no class
// attached — and parameter order DOES differ between classes (e.g.
// LayerRegistry.setWeightsFlat(layer_id, layer_type, data) vs
// GraphMutationTransaction.setWeightsFlat(layer_type, layer_id, values)).
// JSON (including error JSON) goes to stdout; plain usage text goes
// to stderr, so an agent reading stdout always gets parseable JSON.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { printJsonAndExit, exitAfterFlush } from '../lib/runner.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
import { pkgDir } from '../lib/engine.mjs';
const ENGINE = pkgDir();
const [arg] = process.argv.slice(2);
if (!arg || arg === '--help' || arg === '-h') { console.error('usage: node helper/bin/surface.mjs <name> | --list'); process.exit(2); }
let surface, bindings, dts;
try {
  surface = JSON.parse(fs.readFileSync(path.join(ENGINE, 'docs', 'wasm-surface.v1.json'), 'utf8'));
  bindings = JSON.parse(fs.readFileSync(path.join(ENGINE, 'wasm-surface.bindings.actual.json'), 'utf8'));
  dts = fs.readFileSync(path.join(ENGINE, 'burn_research.d.ts'), 'utf8');
} catch (e) { printJsonAndExit({ ok: false, error: `cannot read engine surface from ${ENGINE} (need a full engine bundle: build pkg/ with the surface docs, or set BURN_WASM_PKG_DIR to a CI bundle): ${e instanceof Error ? e.message : String(e)}` }, 1); }
const fnNames = surface.functions ?? [];
const classes = surface.classes ?? {};
if (arg === '--list') {
  for (const n of [...fnNames].sort()) console.log(`function ${n}`);
  for (const n of Object.keys(classes).sort()) console.log(`class ${n}`);
  exitAfterFlush(0);
}
// Parse .d.ts class blocks: {ClassName: [signature lines in that class]}.
// A class block starts at "export class X {" and ends at a line that is
// exactly "}". Method lines inside are "name(...): ...;" or "constructor();".
function classBlocks(text) {
  const blocks = {};
  let current = null;
  for (const line of text.split('\n')) {
    const m = line.match(/^export class (\w+) \{/);
    if (m) { current = m[1]; blocks[current] = []; continue; }
    if (current && line === '}') { current = null; continue; }
    if (current) blocks[current].push(line.trim());
  }
  return blocks;
}
const blocks = classBlocks(dts);
// Function: signature line from the .d.ts, if present.
if (fnNames.includes(arg)) {
  const line = dts.split('\n').find((l) => l.trim().startsWith(`export function ${arg}(`));
  printJsonAndExit({ kind: 'function', name: arg, signature: line ? line.trim() : null }, 0);
}
// Class: method list from the bindings snapshot (falls back to the surface doc).
if (Object.prototype.hasOwnProperty.call(classes, arg)) {
  printJsonAndExit({ kind: 'class', name: arg, methods: bindings.classes?.[arg] ?? classes[arg] }, 0);
}
// Method name: each owning class paired with ITS OWN signature (or null
// if the .d.ts class block does not carry that method line).
// v2.4 (Audit_7 E3): .d.ts declares factory methods as `static linear(`
// and properties as `flags: number;` — the old match required the line
// to START with the bare name, so every static method (including all 48
// AgentLayerSpec constructors) returned null. Strip leading modifiers
// before matching, and also match property declarations (`name:`).
const owners = Object.keys(classes).filter((c) => (bindings.classes?.[c] ?? classes[c] ?? []).includes(arg));
if (owners.length) {
  const perClass = {};
  const stripModifiers = (l) => l.replace(/^(?:static|private|readonly|abstract)\s+/g, '');
  for (const c of owners) {
    const lines = blocks[c] ?? [];
    const hit = lines.find((l) => {
      const s = stripModifiers(l);
      return s.startsWith(`${arg}(`) || s === `${arg}();` || s.startsWith(`${arg}();`) || s.startsWith(`${arg}:`);
    });
    perClass[c] = hit ?? null;
  }
  printJsonAndExit({ kind: 'method', name: arg, per_class: perClass }, 0);
}
printJsonAndExit({ ok: false, error: `unknown surface entry: ${arg}`, hint: 'run: node helper/bin/surface.mjs --list' }, 2);
