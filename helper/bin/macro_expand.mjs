#!/usr/bin/env node
// burn-math #1: macro expansion + proof.
//
// Usage:
//   node macro_expand.mjs expand <macro.json>            -> flat spec JSON
//   node macro_expand.mjs prove <macro.json> <reference.json>
//       Expands the macro, runs BOTH specs on the engine with the macro's
//       inputs, and requires bit-identical outputs.
//       Prints {ok, equivalent, outputsMacro, outputsReference}.
//       Exit 0 iff equivalent.
//
// macro.json: {macro, params, weights?, ports, inputs, idBase?, slotBase?,
//   outputSlot?} — see lib/macros.mjs. The reference spec must share the
//   macro's ports/inputs (same input values) for the proof to mean anything.
import fs from 'node:fs';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { expandMacro, macroNames } from '../lib/macros.mjs';
import { printJson, failCli } from '../lib/runner.mjs';

const [cmd, aFile, bFile] = process.argv.slice(2);
if (!cmd || !aFile || !['expand', 'prove'].includes(cmd)) {
  console.error('usage: node macro_expand.mjs expand <macro.json> | prove <macro.json> <reference.json>');
  console.error(`known macros: ${macroNames().join(', ')}`);
  process.exit(2);
}
let inv;
try { inv = JSON.parse(fs.readFileSync(aFile, 'utf8')); }
catch (e) { failCli(`cannot read macro file: ${e.message}`); }

let spec;
try { spec = expandMacro(inv); }
catch (e) { failCli(`expansion failed: ${e.message}`); }

if (cmd === 'expand') {
  printJson({ ok: true, spec });
  process.exit(0);
}

// prove
let ref;
try { ref = loadSpec(bFile); } catch (e) { failCli(`reference: ${e.message}`); }

const rt = await loadWasm();
function runOne(s) {
  const { reg, plan, g } = buildGraph(rt, s);
  const ports = portBySlot(s);
  const bundle = makeBundle(rt, plan, s.inputs, ports);
  const out = g.run(reg, bundle);
  const values = Array.from(out.to_array());
  out.free();
  return values;
}
let vM, vR;
try { vM = runOne(spec); } catch (e) { failCli(`macro spec failed on engine: ${e.message}`); }
try { vR = runOne(ref); } catch (e) { failCli(`reference spec failed on engine: ${e.message}`); }
const equivalent = vM.length === vR.length && vM.every((x, i) => x === vR[i]);
printJson({
  ok: equivalent, equivalent,
  outputsMacro: vM, outputsReference: vR,
  note: equivalent
    ? 'bit-identical outputs on the engine: the expansion faithfully implements the macro'
    : 'DIVERGENCE: expansion does not match the reference',
});
process.exit(equivalent ? 0 : 2);
