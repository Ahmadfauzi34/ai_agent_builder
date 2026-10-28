#!/usr/bin/env node
// burn-math soak: long-run determinism proof (WASM-direct).
// Usage: node soak.mjs <spec.json> [N=100]
// Runs the spec N times with a FRESH input bundle each iteration (covers the
// bind path too), hashes the raw f32 output bytes per run, and reports whether
// all N runs are bit-identical. Prints {ok, runs, identical, sha256, ...}.
// Exit 0 iff all runs identical, 2 otherwise.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit } from '../lib/runner.mjs';
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';

const [specFile, nArg] = process.argv.slice(2);
if (!specFile) {
  console.error('usage: node soak.mjs <spec.json> [N=100]');
  process.exit(2);
}
const N = nArg !== undefined ? Number(nArg) : 100;
if (!Number.isInteger(N) || N < 2) {
  printJsonAndExit({ ok: false, error: 'N must be an integer >= 2' }, 2);
}

try {
  const spec = loadSpec(specFile);
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);

  let refHash = null;
  let firstMismatch = -1;
  let outLen = 0;
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    const bundle = makeBundle(rt, plan, spec.inputs, ports);
    const out = g.run(reg, bundle);
    const arr = out.to_array(); // Float32Array view/copy
    outLen = arr.length;
    const h = createHash('sha256')
      .update(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength))
      .digest('hex');
    out.free();
    if (bundle.free) bundle.free();
    if (refHash === null) refHash = h;
    else if (h !== refHash && firstMismatch === -1) firstMismatch = i;
  }
  const totalMs = performance.now() - t0;
  if (g.free) g.free();

  const identical = firstMismatch === -1;
  printJsonAndExit({
    ok: true,
    runs: N,
    identical,
    sha256: `sha256:${refHash}`,
    output_f32_count: outLen,
    first_mismatch_index: identical ? null : firstMismatch,
    total_ms: +totalMs.toFixed(1),
    ms_per_run: +(totalMs / N).toFixed(3),
  }, identical ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
