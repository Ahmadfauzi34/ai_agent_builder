#!/usr/bin/env node
// burn-math recurse: Recursive Tree Search (Rewrites + Quantization + CheckpointBranchSet),
// Multi-Layer ES Fit, and Closed-Loop Reflective Backtracking GD Solver CLI.
//
// Usage:
//   node bin/recurse.mjs search <spec.json> [out.json] [--max-depth 3] [--quant-tol 1e-3] [--cases cases.json]
//   node bin/recurse.mjs fit <spec.json> <targets.json> [gens=40] [--sensitivity-guided]
//   node bin/recurse.mjs solve-gd <spec.json> [targets.json] [steps=20] [--steps 20] [--lr 0.1] [--layer 0|all] [--multi-layer] [--precondition] [--out out.json]
//   targets.json, when provided, must hold {"cases":[{"inputs":[...],"expected":[...]}]} (bare arrays are rejected, see B5).
//   Without targets, 1-output specs optimize the scalar output itself and multi-output specs minimize toward zero;
//   the mode taken is declared as targets_mode in the result ("explicit" | "scalar_objective" | "zero_default").
import fs from 'node:fs';
import path from 'node:path';
import { loadWasm } from '../lib/wasm_direct.mjs';
import { loadSpec, hydrateSpecArtifactsInPlace, isRecursiveSpecV2 } from '../lib/spec.mjs';
import { compileRecursiveSpec } from '../lib/composer.mjs';
import { searchGraphTree, optimizeMultiLayerES, reflectiveGdSolve } from '../lib/recurse.mjs';
import { printJsonAndExit, assertKnownFlags } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node recurse.mjs search <spec.json> [out.json] [--max-depth 3] [--quant-tol 1e-3] | fit <spec.json> <targets.json> [gens=40] [--sensitivity-guided] | solve-gd <spec.json> [targets.json] [--steps 20] [--lr 0.1] [--layer 0|all] [--multi-layer] [--precondition] [--out out.json]');
  process.exit(2);
}

const cmd = rawArgs[0];
function flagVal(flag, defVal) {
  const idx = rawArgs.indexOf(flag);
  if (idx >= 0 && idx + 1 < rawArgs.length) return rawArgs[idx + 1];
  return defVal;
}

const valuedFlags = new Set(['--max-depth', '--quant-tol', '--cases', '--abs-tol', '--rel-tol', '--freeze-threshold', '--lr', '--layer', '--steps', '--out', '--targets']);
const positional = rawArgs.slice(1).filter((a, i, arr) => {
  if (a.startsWith('--')) return false;
  if (i > 0 && valuedFlags.has(arr[i - 1])) return false;
  return true;
});
assertKnownFlags(rawArgs.slice(1), { valued: [...valuedFlags], boolean: ['--sensitivity-guided', '--multi-layer', '--precondition'], prog: `recurse ${cmd}` });

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
  const rt = await loadWasm();

  if (cmd === 'search') {
    const [specFile, outFile] = positional;
    if (!specFile) {
      console.error('usage: node recurse.mjs search <spec.json> [out.json] [--max-depth 3] [--quant-tol 1e-3]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    const maxDepth = Number(flagVal('--max-depth', 3));
    const quantTolArg = flagVal('--quant-tol', null);
    const quantTol = quantTolArg !== null ? Number(quantTolArg) : null;
    const casesFile = flagVal('--cases', null);
    const cases = casesFile ? JSON.parse(fs.readFileSync(casesFile, 'utf8')) : null;
    const absTol = Number(flagVal('--abs-tol', 1e-5));
    const relTol = Number(flagVal('--rel-tol', 1e-5));

    const res = await searchGraphTree(rt, spec, { maxDepth, quantTol, cases, absTol, relTol });
    if (outFile && res.best_spec) {
      fs.writeFileSync(outFile, JSON.stringify(res.best_spec, null, 2) + '\n');
    }
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  if (cmd === 'fit') {
    const [specFile, targetsFile, gensArg] = positional;
    if (!specFile || !targetsFile) {
      console.error('usage: node recurse.mjs fit <spec.json> <targets.json> [gens=40] [--sensitivity-guided]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    const targets = JSON.parse(fs.readFileSync(targetsFile, 'utf8'));
    const generations = gensArg !== undefined ? Number(gensArg) : 40;
    const sensitivityGuided = rawArgs.includes('--sensitivity-guided');
    const freezeThreshold = Number(flagVal('--freeze-threshold', 1e-6));

    const res = optimizeMultiLayerES(rt, spec, targets, { generations, sensitivityGuided, freezeThreshold });
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  if (cmd === 'solve-gd') {
    const [specFile, pos2, pos3] = positional;
    if (!specFile) {
      console.error('usage: node recurse.mjs solve-gd <spec.json> [targets.json] [steps=20] [--steps 20] [--lr 0.1] [--layer 0|all] [--multi-layer] [--precondition] [--out out.json]');
      process.exit(2);
    }
    const spec = loadV1OrV2(specFile);
    let targets = null;
    let steps = Number(flagVal('--steps', 20));
    const targetsFlag = flagVal('--targets', null);
    if (targetsFlag) {
      targets = JSON.parse(fs.readFileSync(targetsFlag, 'utf8'));
      if (pos2 !== undefined) steps = Number(pos2);
    } else if (pos2 !== undefined) {
      if (fs.existsSync(pos2)) {
        targets = JSON.parse(fs.readFileSync(pos2, 'utf8'));
        if (pos3 !== undefined) steps = Number(pos3);
      } else {
        steps = Number(pos2);
      }
    }
    const lr = Number(flagVal('--lr', 0.1));
    const layerArg = flagVal('--layer', null);
    const layerIndex = layerArg === 'all' ? 'all' : (layerArg !== null ? Number(layerArg) : null);
    const multiLayer = rawArgs.includes('--multi-layer') || layerArg === 'all';
    const precondition = rawArgs.includes('--precondition');
    const outFile = flagVal('--out', null);

    const res = reflectiveGdSolve(rt, spec, targets, { layerIndex, multiLayer, steps, lr, precondition });
    if (outFile && res.optimized_spec) {
      fs.writeFileSync(outFile, JSON.stringify(res.optimized_spec, null, 2) + '\n');
      res.out_file = outFile;
    }
    printJsonAndExit(res, res.ok ? 0 : 2);
  }

  printJsonAndExit({ ok: false, error: `unknown recurse command '${cmd}' (use: search | fit | solve-gd)` }, 2);
} catch (e) {
  const msg = e instanceof Error ? e.message : String(e);
  printJsonAndExit({ ok: false, error: msg }, 2);
}
