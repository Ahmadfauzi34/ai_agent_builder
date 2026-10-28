#!/usr/bin/env node
// burn-math opt_profile: engine-provided low-level optimization profiler.
//
// The burn WASM engine provides the helper — it measures itself. Patches
// WebAssembly.Instance before the engine instantiates (Fase A oracle), then
// reports per-phase wall time and linear-memory page deltas for
// build/bind/run, plus a steady-state leak check.
//
// Usage: node opt_profile.mjs <spec.json> [iterations] [--trace]
//        node opt_profile.mjs <a.json> <b.json> ... [--iterations N] [--trace]
//   Batch mode (#6): the engine loads ONCE, then profiles each spec in one
//   process. Single spec: legacy shape unchanged. Multiple specs: prints
//   [{spec, iterations, trace, engine, ...profile}, ...].
//   Disambiguation: with 2 positionals where the 2nd is numeric and not
//   *.json, the legacy (spec, iterations) reading wins; otherwise every
//   positional is a spec (use --iterations for batch iteration count).
//   Each spec is profiled independently (fresh LayerRegistry per profile;
//   graph/bundle freed after; the leak check is self-relative per spec),
//   so profile N cannot observe profile N-1.
// Prints a JSON envelope on stdout; exit 0 on success, 2 on error.
import { loadEngineWithOracle, profileSpec, pages } from '../lib/opt.mjs';
import { loadSpec } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';

// NOTE: this file used to define its own drain-safe printJsonAndExit
// (single async write + drain wait). Port v2.5 retires it in favor of
// the shared looped-sync version from lib/runner.mjs (same call sites,
// same semantics, one implementation).
const argv = process.argv.slice(2);
let iterationsFlag;
let trace = false;
const positional = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--trace') trace = true;
  else if (a === '--iterations') iterationsFlag = Number(argv[++i]);
  else positional.push(a);
}

const isNum = (s) => s !== '' && !Number.isNaN(Number(s));
let specFiles, iterations;
if (iterationsFlag !== undefined) {
  specFiles = positional;
  iterations = iterationsFlag;
} else if (
  positional.length === 2 &&
  !positional[1].endsWith('.json') &&
  isNum(positional[1])
) {
  specFiles = [positional[0]]; // legacy: spec + iterations
  iterations = Number(positional[1]);
} else {
  specFiles = positional;
  iterations = 11;
}
if (!specFiles.length) {
  console.error('usage: node opt_profile.mjs <spec.json> [iterations] [--trace]');
  process.exit(2);
}

// Profile one spec with an already-loaded engine. Never throws.
function profileOne(rt, specFile, engine) {
  let spec;
  try {
    spec = loadSpec(specFile);
  } catch (e) {
    return { spec: specFile, ok: false, error: e.message };
  }
  try {
    const profile = profileSpec(rt, spec, { iterations, trace });
    return { spec: specFile, ok: true, iterations, trace, engine, ...profile };
  } catch (e) {
    return { spec: specFile, ok: false, error: e.message };
  }
}

try {
  const { rt, loadMs, loadPages, wasmBytes } = await loadEngineWithOracle();
  const engine = {
    load_ms: +loadMs.toFixed(3),
    pages_after_load: loadPages,
    wasm_bytes: wasmBytes,
  };

  if (specFiles.length === 1) {
    // Legacy single-spec path: historical shape and exit codes unchanged.
    const r = profileOne(rt, specFiles[0], engine);
    if (!r.ok) {
      printJsonAndExit({ ok: false, error: r.error }, 2);
    }
    const { spec, ok, trace: _trace, engine: _engine, ...profile } = r;
    void ok; void _trace; void _engine;
    printJsonAndExit({
      ok: true,
      spec,
      engine,
      pages_baseline: loadPages,
      iterations: r.iterations,
      ...profile,
    }, 0);
  }

  const results = specFiles.map((f) => profileOne(rt, f, engine));
  const allOk = results.every((r) => r.ok);
  if (!allOk) printJsonAndExit(results, 2);
  printJson(results);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
