// Engine-backed low-level optimization helpers.
//
// The burn WASM engine PROVIDES the helper: measurement comes from the engine
// itself, not from outside guesses. This module patches WebAssembly.Instance
// (the Fase A memory-oracle technique) before the engine instantiates, then
// profiles spec execution with per-phase wall time and linear-memory page
// deltas. WASM linear memory only grows, so page counts are an exact,
// noise-free allocator signal — unlike host RSS.
//
// Used by bin/opt_profile.mjs and bin/opt_compare.mjs.
import { performance } from 'node:perf_hooks';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadWasm, buildGraph, makeBundle, pkgDir } from './engine.mjs';
import { portBySlot } from './spec.mjs';

let capturedMemory = null;
let patched = false;

// Install the memory oracle. MUST be called before loadWasm(), because the
// generated JS instantiates the module during initSync.
export function installMemoryOracle() {
  if (patched) return;
  const OrigInstance = WebAssembly.Instance;
  WebAssembly.Instance = function (module, imports) {
    const inst = new OrigInstance(module, imports);
    if (inst.exports && inst.exports.memory instanceof WebAssembly.Memory) {
      capturedMemory = inst.exports.memory;
    }
    return inst;
  };
  patched = true;
}

// Current WASM linear-memory size in 64KiB pages.
export function pages() {
  if (!capturedMemory) {
    throw new Error('memory oracle: no WASM memory captured — call installMemoryOracle() before loadWasm()');
  }
  return capturedMemory.buffer.byteLength / 65536;
}

export function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Load the engine with the oracle installed. Returns { rt, loadMs, loadPages, wasmBytes }.
export async function loadEngineWithOracle() {
  installMemoryOracle();
  const t0 = performance.now();
  const rt = await loadWasm();
  const loadMs = performance.now() - t0;
  return { rt, loadMs, loadPages: pages(), wasmBytes: wasmFileBytes() };
}

// Size of the engine's .wasm binary in bytes (bundle-size metric).
function wasmFileBytes() {
  try {
    return fs.statSync(path.join(pkgDir(), 'burn_research_bg.wasm')).size;
  } catch {
    return null;
  }
}

// Callable tiered-limit guard (complaint #14).
//
// Asks the engine's own guard decision function whether a state-record
// payload of the given bytes would be accepted and under which bincode byte
// limit tier — without decoding anything. Only the input length is inspected,
// so probing is always safe. Returns the parsed verdict:
// {input_bytes, tier_limit|null, accepted, reason|null}.
export function guardCheck(rt, bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : Uint8Array.from(bytes);
  return JSON.parse(rt.decodeGuardCheck(u8));
}

// Per-step map via the engine's own trace facility.
// Returns [{index, layer, kind, in_slots, out_slot, out_shape, out_bytes, wall_ms}]
// so an agent can see WHICH step holds the bytes AND which step costs wall time
// (memory hotspot map + time hotspot map).
// Each step is traced individually (runWithTrace with max_steps=1) and timed;
// wall_ms is the median of `repeats` runs. Note: this is traced wall time —
// it includes trace-capture overhead, so use it for relative comparison
// between steps, not as absolute kernel cost.
// Pure observation: does not affect the profiled run.
export function traceSteps(rt, spec, reg, g, bundle,
    { maxTensorBytes = 1024 * 1024, repeats = 3 } = {}) {
  const steps = [];
  for (let i = 0; i < spec.steps.length; i++) {
    const dts = [];
    let stepObs = null;
    for (let r = 0; r < repeats; r++) {
      const t0 = performance.now();
      const traced = g.runWithTrace(reg, bundle, i, 1, maxTensorBytes);
      const dt = performance.now() - t0;
      try {
        const rep = JSON.parse(traced.report());
        stepObs = (rep.steps ?? [])[0] ?? stepObs;
      } finally {
        if (traced.free) traced.free();
      }
      dts.push(dt);
    }
    if (!stepObs) throw new Error(`trace: no observation for step ${i}`);
    const stepDef = spec.steps[i] ?? {};
    const layerDef = spec.layers[stepDef.layer] ?? {};
    steps.push({
      index: i,
      layer: layerDef.constructor ?? `type:${stepObs.layer_type}`,
      kind: stepDef.kind ?? (stepObs.arity === 2 ? 'binary' : 'unary'),
      in_slots: stepObs.input_slots ?? [],
      out_slot: stepObs.output_slot ?? null,
      out_shape: stepObs.observation?.shape ?? null,
      out_bytes: stepObs.observation?.byte_length ?? null,
      wall_ms: +median(dts).toFixed(3),
    });
  }
  return steps;
}

// Profile one spec: build/bind/run phases with wall time and page deltas.
// Run phase = 1 cold iteration + (iterations-1) steady iterations; outputs
// are freed each iteration so steady-state page growth indicates a real leak.
// With { trace: true }, also captures the per-step memory map via the
// engine's own runWithTrace facility (which step holds the bytes).
export function profileSpec(rt, spec, { iterations = 11, trace = false } = {}) {
  if (iterations < 2) throw new Error('iterations must be >= 2 (1 cold + steady)');

  // --- build ---
  let t0 = performance.now();
  const p0 = pages();
  const { reg, plan, g } = buildGraph(rt, spec);
  const buildMs = performance.now() - t0;
  const buildPages = pages() - p0;
  const ports = portBySlot(spec);

  // --- bind ---
  t0 = performance.now();
  const q0 = pages();
  const bundle = makeBundle(rt, plan, spec.inputs, ports);
  const bindMs = performance.now() - t0;
  const bindPages = pages() - q0;

  // --- run ---
  const runMs = [];
  const runPages = [];
  let outputShape = null;
  let outputValues = null;
  for (let i = 0; i < iterations; i++) {
    const r0 = pages();
    t0 = performance.now();
    const out = g.run(reg, bundle);
    const dt = performance.now() - t0;
    if (i === 0) {
      const arr = out.to_array();
      outputShape = Array.from(out.shape ?? []);
      outputValues = Array.from(arr).slice(0, 8);
    }
    out.free();
    runMs.push(dt);
    runPages.push(pages() - r0);
  }
  if (bundle.free) bundle.free();
  if (g.free) g.free();

  const coldMs = runMs[0];
  const steadyMs = runMs.slice(1);
  const steadyPages = runPages.slice(1);
  // Leak-free iff every steady iteration leaves the page count unchanged.
  const leakFree = steadyPages.every((d) => d === 0);

  // Optional per-step map (memory + wall time). Needs a fresh bundle because
  // the profile run above freed its bundle; rebuild cheaply (not timed).
  let steps = null;
  let hotspots = null;
  if (trace) {
    const { reg: reg2, plan: plan2, g: g2 } = buildGraph(rt, spec);
    const bundle2 = makeBundle(rt, plan2, spec.inputs, ports);
    try {
      steps = traceSteps(rt, spec, reg2, g2, bundle2);
      const byTime = [...steps].sort((a, b) => b.wall_ms - a.wall_ms)[0];
      const byBytes = [...steps].sort((a, b) => (b.out_bytes ?? 0) - (a.out_bytes ?? 0))[0];
      hotspots = {
        slowest_step: byTime ? { index: byTime.index, layer: byTime.layer, wall_ms: byTime.wall_ms } : null,
        biggest_step: byBytes ? { index: byBytes.index, layer: byBytes.layer, out_bytes: byBytes.out_bytes } : null,
      };
    } finally {
      if (bundle2.free) bundle2.free();
      if (g2.free) g2.free();
    }
  }

  return {
    phases: {
      build: { ms: +buildMs.toFixed(3), pages_delta: buildPages },
      bind: { ms: +bindMs.toFixed(3), pages_delta: bindPages },
      run_cold: { ms: +coldMs.toFixed(3), pages_delta: runPages[0] },
    },
    steady: {
      iterations: steadyMs.length,
      run_ms_median: +median(steadyMs).toFixed(3),
      run_ms_min: +Math.min(...steadyMs).toFixed(3),
      run_ms_max: +Math.max(...steadyMs).toFixed(3),
      pages_delta_total: steadyPages.reduce((a, b) => a + b, 0),
      leak_free: leakFree,
    },
    steps,
    hotspots,
    output_shape: outputShape,
    output_preview: outputValues,
  };
}
