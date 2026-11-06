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
//   node autodiff.mjs verify-fd <spec.json> [eps] [tol] [dout.json]
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot, analyzePlanPorts } from '../lib/spec.mjs';
import { printJsonAndExit, failCli } from '../lib/runner.mjs';
import { inferShapes, makeMap, backwardWithSaved, captureSavedForSpec } from '../lib/autodiff.mjs';
import fs from 'node:fs';

const rawArgs = process.argv.slice(2);
const [cmd, aFile, bFile] = rawArgs;
if (!cmd || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node autodiff.mjs map | grad <spec.json> [dout.json] | prove-rules | verify-fd <spec.json> [eps] [tol] [dout.json]');
  process.exit(2);
}

if (cmd === 'map') {
  printJsonAndExit({ ok: true, ...makeMap() }, 0);
}

const rt = await loadWasm();

function ensurePortsConsumed(spec) {
  const planCheck = analyzePlanPorts(spec);
  if (planCheck.valid || planCheck.code !== 'plan_unused_ports') return spec;
  const s = JSON.parse(JSON.stringify(spec));
  let nextRegId = Math.max(0, ...s.layers.map((l) => l.args[0] ?? 0)) + 1;
  let nextSlot = s.numSlots;
  for (const unusedSlot of planCheck.unused_ports) {
    const drainIdx = s.layers.length;
    s.layers.push({ constructor: 'relu', args: [nextRegId++] });
    s.steps.push({ kind: 'unary', layer: drainIdx, slots: [unusedSlot, nextSlot++] });
  }
  s.numSlots = nextSlot;
  return s;
}

function runForward(spec, outputSlot) {
  const s = ensurePortsConsumed({ ...spec, outputSlot });
  const { reg, plan, g } = buildGraph(rt, s);
  const bundle = makeBundle(rt, plan, s.inputs, portBySlot(s));
  try {
    const out = g.run(reg, bundle);
    const v = Array.from(out.to_array());
    out.free();
    return v;
  } finally {
    if (bundle?.free) bundle.free();
    if (g?.free) g.free();
    if (plan?.free) plan.free();
    if (reg?.free) reg.free();
  }
}

export function backward(spec, doutSeed) {
  const shapes = inferShapes(spec);
  const saved = captureSavedForSpec(spec, shapes, runForward);
  const weightOf = (layerIdx, args, ctor = 'linear') => {
    const vals = spec.weights?.[String(layerIdx)];
    if (ctor === 'rmsNorm') {
      const F = args[1];
      return { w: vals ? vals.slice(0, F) : new Array(F).fill(1), b: null };
    }
    if (ctor === 'layerNorm') {
      const F = args[1];
      return {
        w: vals ? vals.slice(0, F) : new Array(F).fill(1),
        b: vals ? vals.slice(F, 2 * F) : new Array(F).fill(0),
      };
    }
    const [, inD, outD, bias] = args;
    if (!vals) failCli(`linear layer ${layerIdx} has no weights`);
    const nw = inD * outD;
    return { w: vals.slice(0, nw), b: bias ? vals.slice(nw, nw + outD) : null };
  };
  return backwardWithSaved(spec, shapes, saved, doutSeed, weightOf);
}

if (cmd === 'grad') {
  if (!aFile) failCli('usage: node autodiff.mjs grad <spec.json> [dout.json]');
  let spec;
  try { spec = loadSpec(aFile); } catch (e) { failCli(e); }
  const dout = bFile ? JSON.parse(fs.readFileSync(bFile, 'utf8')) : null;
  try {
    const r = backward(spec, dout);
    printJsonAndExit({ ok: true, ...r }, 0);
  } catch (e) {
    failCli(e);
  }
}

if (cmd === 'prove-rules') {
  const { proveAll } = await import('./autodiff_prove.mjs');
  const r = await proveAll(runForward);
  printJsonAndExit(r, r.ok ? 0 : 2);
}

if (cmd === 'verify-fd') {
  // node autodiff.mjs verify-fd <spec.json> [eps=1e-3] [tol=1e-3] [dout.json]
  // Independent check: backward() input+param grads vs central finite
  // differences of the engine's forward pass. This is the command any
  // agent (or human) runs to audit the map on an arbitrary graph.
  if (!aFile) failCli('usage: node autodiff.mjs verify-fd <spec.json> [eps] [tol] [dout.json]');
  const eps = Number(process.argv[4] ?? 1e-3), tol = Number(process.argv[5] ?? 1e-3);
  let spec;
  try { spec = loadSpec(aFile); } catch (e) { failCli(e); }
  try {
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
        const decl = spec.layers[Number(li)];
        const [, inD, outD, bias] = decl?.args ?? [];
        const base = [...(spec.weights?.[li] ?? new Array((inD * outD) + (bias ? outD : 0)).fill(0))];
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
    printJsonAndExit({ ok, maxErr, mismatches: checks.slice(0, 5), note: ok ? 'backward() matches finite differences' : 'MISMATCH: map disagrees with oracle' }, ok ? 0 : 2);
  } catch (e) {
    failCli(e instanceof Error ? e.message : String(e));
  }
}

failCli('usage: node autodiff.mjs map | grad <spec.json> [dout.json] | prove-rules | verify-fd <spec.json> [eps] [tol] [dout.json]');
