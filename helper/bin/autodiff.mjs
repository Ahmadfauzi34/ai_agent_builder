#!/usr/bin/env node
// burn-math #12: autodiff as a MAP.
//
//   node autodiff.mjs map
//     Print the rule table (inspectable map): every rule, what it saves,
//     its proof status, and the classified gaps.
//
//   node autodiff.mjs grad <spec.json> [dout.json]
//     Reverse-mode gradient of sum(dOut * outputs): runs the forward spec
//     on the engine, captures the saved forward values each rule needs,
//     walks the graph backward applying the VJP table.
//     dout.json: {"<outputSlot>": [...]} (default: ones).
//     Prints {inputGrads: {slot: [...]}, paramGrads: {layerIdx: {w,b}}}.
//     Exit 2 if any op has no rule (honest error, never silent).
//
//   node autodiff.mjs prove-rules
//     Prove every rule twice: (1) hand-computed exact case, (2) central
//     finite differences of the engine's forward pass. Exit 2 on failure.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, failCli } from '../lib/runner.mjs';
import { RULES, GAPS, ruleFor, ruleNames, inferShapes, makeMap, numel, backwardWithSaved } from '../lib/autodiff.mjs';
import fs from 'node:fs';

const [cmd, aFile, bFile] = process.argv.slice(2);

if (cmd === 'map') {
  printJson({ ok: true, ...makeMap() });
  process.exit(0);
}

const rt = await loadWasm();

function runForward(spec, outputSlot) {
  const s = { ...spec, outputSlot };
  const { reg, plan, g } = buildGraph(rt, s);
  const out = g.run(reg, makeBundle(rt, plan, s.inputs, portBySlot(s)));
  const v = Array.from(out.to_array());
  out.free();
  return v;
}

// Capture forward values: every step-input slot, plus output slots that any
// rule lists in `saved`. (Capturing all inputs is simple and robust; probe
// runs are cheap next to a finite-difference sweep.)
function captureSaved(spec, shapes) {
  const saved = new Map(); // slot -> {values, shape}
  const need = new Set();
  for (const step of spec.steps) {
    const rule = ruleFor(spec.layers[step.layer].constructor);
    const inSlots = step.slots.slice(0, step.kind === 'binary' ? 2 : 1);
    const outSlot = step.slots[step.kind === 'binary' ? 2 : 1];
    inSlots.forEach((s) => need.add(s));
    for (const w of rule.saved) if (w === 'output') need.add(outSlot);
  }
  for (const s of need) saved.set(s, { values: runForward(spec, s), shape: shapes.get(s) });
  return saved;
}

function splitWeights(values, args) {
  // linear(id,in_dim,out_dim,bias): [W column-major (in*out), b (out)?]
  const [, inD, outD, bias] = args;
  const nw = inD * outD;
  return { w: values.slice(0, nw), b: bias ? values.slice(nw, nw + outD) : null };
}

export function backward(spec, doutSeed) {
  const shapes = inferShapes(spec);
  const saved = captureSaved(spec, shapes);
  const weightOf = (layerIdx, args) => {
    const vals = spec.weights?.[String(layerIdx)];
    if (!vals) failCli(`linear layer ${layerIdx} has no weights`);
    const [, inD, outD, bias] = args;
    const nw = inD * outD;
    return { w: vals.slice(0, nw), b: bias ? vals.slice(nw, nw + outD) : null };
  };
  return backwardWithSaved(spec, shapes, saved, doutSeed, weightOf);
}

if (cmd === 'grad') {
  if (!aFile) failCli('usage: node autodiff.mjs grad <spec.json> [dout.json]');
  const spec = loadSpec(aFile);
  const dout = bFile ? JSON.parse(fs.readFileSync(bFile, 'utf8')) : null;
  try {
    const r = backward(spec, dout);
    printJson({ ok: true, ...r });
  } catch (e) {
    failCli(e.message);
  }
  process.exit(0);
}

if (cmd === 'prove-rules') {
  const { proveAll } = await import('./autodiff_prove.mjs');
  const r = await proveAll(runForward);
  printJson(r);
  process.exit(r.ok ? 0 : 2);
}

if (cmd === 'verify-fd') {
  // node autodiff.mjs verify-fd <spec.json> [eps=1e-3] [tol=1e-3] [dout.json]
  // Independent check: backward() input+param grads vs central finite
  // differences of the engine's forward pass. This is the command any
  // agent (or human) runs to audit the map on an arbitrary graph.
  if (!aFile) failCli('usage: node autodiff.mjs verify-fd <spec.json> [eps] [tol] [dout.json]');
  const eps = Number(process.argv[4] ?? 1e-3), tol = Number(process.argv[5] ?? 1e-3);
  const spec = loadSpec(aFile);
  const dout = process.argv[6] ? JSON.parse(fs.readFileSync(process.argv[6], 'utf8')) : null;
  const b = backward(spec, dout);
  const seed = dout?.[spec.outputSlot] ?? null;
  const L = (vals) => seed ? vals.reduce((a, v, i) => a + seed[i] * v, 0) : vals.reduce((a, v) => a + v, 0);
  const checks = [];
  let maxErr = 0, ok = true;
  const fwd = (sp) => runForward({ ...sp, outputSlot: spec.outputSlot }, spec.outputSlot);
  for (const p of spec.ports) {
    const slot = String(p.slot), base = [...spec.inputs[p.slot]];
    const ag = b.inputGrads[slot];
    for (let i = 0; i < base.length; i++) {
      const xp = { ...spec.inputs, [p.slot]: base.map((v, j) => (j === i ? v + eps : v)) };
      const xm = { ...spec.inputs, [slot]: base.map((v, j) => (j === i ? v - eps : v)) };
      const nd = (L(fwd({ ...spec, inputs: xp })) - L(fwd({ ...spec, inputs: xm }))) / (2 * eps);
      const err = Math.abs(nd - ag[i]);
      maxErr = Math.max(maxErr, err);
      if (err > tol) { ok = false; checks.push(`in slot ${slot}[${i}]: analytic=${ag[i]} fd=${nd}`); }
    }
  }
  for (const [li, pg] of Object.entries(b.paramGrads)) {
    for (const [k, arr] of Object.entries(pg)) {
      if (!arr) continue;
      const base = [...spec.weights[li]];
      const off = k === 'b' ? base.length - arr.length : 0;
      for (let i = 0; i < arr.length; i++) {
        const wp = base.map((v, j) => (j === off + i ? v + eps : v));
        const wm = base.map((v, j) => (j === off + i ? v - eps : v));
        const nd = (L(fwd({ ...spec, weights: { ...spec.weights, [li]: wp } })) - L(fwd({ ...spec, weights: { ...spec.weights, [li]: wm } }))) / (2 * eps);
        const err = Math.abs(nd - arr[i]);
        maxErr = Math.max(maxErr, err);
        if (err > tol) { ok = false; checks.push(`param ${li}.${k}[${i}]: analytic=${arr[i]} fd=${nd}`); }
      }
    }
  }
  printJson({ ok, maxErr, mismatches: checks.slice(0, 5), note: ok ? 'backward() matches finite differences' : 'MISMATCH: map disagrees with oracle' });
  process.exit(ok ? 0 : 2);
}

failCli('usage: node autodiff.mjs map | grad <spec.json> [dout.json] | prove-rules | verify-fd <spec.json> [eps] [tol] [dout.json]');
