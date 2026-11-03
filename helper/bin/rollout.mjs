#!/usr/bin/env node
// burn-math rollout: Temporal & Execution-Time Recursion CLI.
//
// Usage:
//   node bin/rollout.mjs scan <spec.json> [--steps N] [--state-port <slot>] [--inputs-seq <seq.json>]
//   node bin/rollout.mjs fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot>]
//   node bin/rollout.mjs unroll <spec.json> [out.json] [--steps N] [--state-port <slot>] [--compact-slots]
//   node bin/rollout.mjs bisect <base.json> <cand.json>
import fs from 'node:fs';
import path from 'node:path';
import { loadWasm } from '../lib/wasm_direct.mjs';
import { loadSpec, hydrateSpecArtifactsInPlace, isRecursiveSpecV2 } from '../lib/spec.mjs';
import { compileRecursiveSpec } from '../lib/composer.mjs';
import { rolloutScan, rolloutFixpoint, bisectDivergence, unrollRolloutSpec } from '../lib/rollout.mjs';
import { printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node rollout.mjs scan <spec.json> [--steps N] [--state-port <slot>] | fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot>] | unroll <spec.json> [out.json] [--steps N] [--state-port <slot>] [--compact-slots] | bisect <base.json> <cand.json>');
  process.exit(2);
}

const cmd = rawArgs[0];
function flagVal(flag, defVal) {
  const idx = rawArgs.indexOf(flag);
  if (idx >= 0 && idx + 1 < rawArgs.length) return rawArgs[idx + 1];
  return defVal;
}

const valuedFlags = new Set(['--steps', '--state-port', '--inputs-seq', '--tol', '--max-steps', '--div-threshold', '--out']);
const positional = rawArgs.slice(1).filter((a, i, arr) => {
  if (a.startsWith('--')) return false;
  if (i > 0 && valuedFlags.has(arr[i - 1])) return false;
  return true;
});
assertKnownFlags(rawArgs.slice(1), { valued: [...valuedFlags], boolean: ['--compact-slots'], prog: `rollout ${cmd}` });

function loadV1OrV2(file) {
  const baseDir = path.dirname(path.resolve(file));
  const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
  hydrateSpecArtifactsInPlace(raw, baseDir);
  if (isRecursiveSpecV2(raw)) {
    const compiled = compileRecursiveSpec(raw, { baseDir, autoRepair: true });
    hydrateSpecArtifactsInPlace(compiled, baseDir);
    return compiled;
  }
  return loadSpec(file);
}

try {
  if (cmd === 'unroll') {
    const [specFile, outPos] = positional;
    if (!specFile) {
      console.error('usage: node rollout.mjs unroll <spec.json> [out.json] [--steps N] [--state-port <slot>] [--compact-slots]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    const steps = Number(flagVal('--steps', (outPos && !outPos.endsWith('.json')) ? outPos : 3));
    const outFile = flagVal('--out', (outPos && outPos.endsWith('.json')) ? outPos : null);
    const statePort = flagVal('--state-port', null);
    const compactSlots = rawArgs.includes('--compact-slots');
    const unrolled = unrollRolloutSpec(spec, { steps, statePort, compactSlots });
    if (outFile) {
      fs.writeFileSync(outFile, JSON.stringify(unrolled, null, 2) + '\n');
    }
    printJsonAndExit({
      ok: true,
      mode: 'unroll',
      steps,
      state_port_slot: unrolled.meta.unrolled_rollout.state_port_slot,
      layers_count: unrolled.layers.length,
      steps_count: unrolled.steps.length,
      num_slots: unrolled.numSlots,
      output_slot: unrolled.outputSlot,
      ...(outFile ? { out_file: outFile } : {}),
      spec: unrolled,
    }, 0);
  }

  const rt = await loadWasm();

  if (cmd === 'scan') {
    const specFile = positional[0];
    if (!specFile) {
      console.error('usage: node rollout.mjs scan <spec.json> [--steps N] [--state-port <slot>]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    const steps = Number(flagVal('--steps', positional[1] ?? 10));
    const statePort = flagVal('--state-port', null);
    const seqFile = flagVal('--inputs-seq', null);
    const inputSequence = seqFile ? JSON.parse(fs.readFileSync(seqFile, 'utf8')) : null;

    const res = await rolloutScan(rt, spec, { steps, statePort, inputSequence });
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  if (cmd === 'fixpoint') {
    const specFile = positional[0];
    if (!specFile) {
      console.error('usage: node rollout.mjs fixpoint <spec.json> [--tol 1e-6] [--max-steps 100] [--state-port <slot>]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    const tol = Number(flagVal('--tol', 1e-6));
    const maxSteps = Number(flagVal('--max-steps', 100));
    const statePort = flagVal('--state-port', null);
    const divergenceThreshold = Number(flagVal('--div-threshold', 1e6));

    const res = await rolloutFixpoint(rt, spec, { statePort, maxSteps, tol, divergenceThreshold });
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  if (cmd === 'bisect') {
    const [baseFile, candFile] = positional;
    if (!baseFile || !candFile) {
      console.error('usage: node rollout.mjs bisect <base.json> <cand.json>');
      process.exit(2);
    }
    const baseSpec = loadV1OrV2(baseFile);
    const candSpec = loadV1OrV2(candFile);
    const res = await bisectDivergence(rt, baseSpec, candSpec);
    printJsonAndExit(res, 0);
  }

  printJsonAndExit({ ok: false, error: `unknown rollout command '${cmd}' (use: scan | fixpoint | unroll | bisect)` }, 2);
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  printJsonAndExit({ ok: false, code: e?.code ?? 'rollout_error', error: msg }, 2);
}
