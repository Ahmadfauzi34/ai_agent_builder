#!/usr/bin/env node
// wasm-opt opt_profile: engine-provided low-level optimization profiler.
//
// The burn WASM engine provides the helper — it measures itself. Patches
// WebAssembly.Instance before the engine instantiates (Fase A memory-oracle
// technique), then reports per-phase wall time and linear-memory page deltas
// for build/bind/run, plus a steady-state leak check.
//
// Usage: node opt_profile.mjs <spec.json> [iterations] [--trace]
// Prints a JSON envelope on stdout; exit 0 on success, 2 on error.
import { loadEngineWithOracle, profileSpec } from '../lib/opt.mjs';
import { loadSpec } from '../lib/spec.mjs';
import { printJson } from '../lib/json.mjs';

const argv = process.argv.slice(2);
const trace = argv.includes('--trace');
const positional = argv.filter((a) => a !== '--trace');
const [specFile, iterArg] = positional;
if (!specFile) {
  console.error('usage: node opt_profile.mjs <spec.json> [iterations] [--trace]');
  process.exit(2);
}
const iterations = iterArg !== undefined ? Number(iterArg) : 11;

try {
  const spec = loadSpec(specFile);
  const { rt, loadMs, loadPages, wasmBytes } = await loadEngineWithOracle();
  const profile = profileSpec(rt, spec, { iterations, trace });
  printJson({
    ok: true,
    spec: specFile,
    engine: { load_ms: +loadMs.toFixed(3), pages_after_load: loadPages, wasm_bytes: wasmBytes },
    pages_baseline: loadPages,
    iterations,
    ...profile,
  });
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
