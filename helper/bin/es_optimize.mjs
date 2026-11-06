#!/usr/bin/env node
// burn-math es_optimize: evolution-strategies optimization of a spec's
// weights, judged entirely by the engine itself.
//
// The engine's EsOptimizer proposes candidate weight vectors (ask); each
// candidate is scored by RUNNING the graph through the engine on target
// cases (fitness = -sum MSE vs expected outputs); scores go back via tell.
//
// Supports both single-layer and multi-layer specs (via canonical layer
// parameter discovery in lib/recurse.mjs), with perturbation centered on
// initial weights and a monotonicity ratchet guard so optimization never
// degrades starting fitness.
//
// Usage: node es_optimize.mjs <spec.json> <targets.json> [gens=40] [pop] [sigma] [lr] [seed=1] [strategy=0] [--multi-layer]
//   targets.json: {"cases":[{"inputs":{slot:[...]}, "expected":[...]}, ...]}
// Scope: weighted layers must be of a type this helper's buildGraph wires
// weights for (linear, the norm family, convs, embedding). With exactly
// one weighted layer the single-layer path runs; with more (or
// --multi-layer) the multi-layer path runs.
// Prints {ok, dim, generations, seed, strategy, start_fitness,
// final_best_fitness, best_weights, trajectory, optimized_spec}.
// Exit 0 on completion; 2 on usage/spec errors.
import fs from 'node:fs';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { optimizeMultiLayerES } from '../lib/recurse.mjs';
import { printJson, failCli, assertKnownFlags , errText } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
const multiLayerFlag = rawArgs.includes('--multi-layer');
const posArgs = rawArgs.filter((a) => !a.startsWith('--'));
const [specFile, targetsFile, gensArg, popArg, sigmaArg, lrArg, seedArg, strategyArg] = posArgs;

if (!specFile || !targetsFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node es_optimize.mjs <spec.json> <targets.json> [gens] [pop] [sigma] [lr] [seed] [strategy] [--multi-layer]');
  process.exit(2);
}
assertKnownFlags(rawArgs, { boolean: ['--multi-layer', '--sensitivity-guided'], prog: 'es_optimize' });
const gens = gensArg !== undefined ? Number(gensArg) : 40;
const seed = seedArg !== undefined ? Number(seedArg) : 1;
const strategy = strategyArg !== undefined ? Number(strategyArg) : 0;
const pop = popArg !== undefined ? Number(popArg) : null;
const sigma = sigmaArg !== undefined ? Number(sigmaArg) : null;
const lr = lrArg !== undefined ? Number(lrArg) : null;

let spec, targets;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e); }
try { targets = JSON.parse(fs.readFileSync(targetsFile, 'utf8')); }
catch (e) { failCli(`cannot read targets: ${errText(e)}`); }
if (!Array.isArray(targets.cases) || targets.cases.length === 0) {
  failCli('targets.json must hold a non-empty "cases" array');
}

const weightedIdx = Object.keys(spec.weights ?? {});
if (weightedIdx.length === 0) {
  failCli('es_optimize needs at least one weighted layer, spec has 0');
}

// Multi-layer path (when spec has >1 weighted layers or --multi-layer is passed)
if (weightedIdx.length > 1 || multiLayerFlag) {
  try {
    const rt = await loadWasm();
    const res = optimizeMultiLayerES(rt, spec, targets, {
      generations: gens,
      pop,
      sigma,
      lr,
      seed,
      strategy,
      sensitivityGuided: rawArgs.includes('--sensitivity-guided'),
      // Consistency fix: recurse fit passes 1e-6, so its
      // --sensitivity-guided filter is live; es_optimize used the
      // library default 0 (filter dead, report-only). Mirror 1e-6.
      freezeThreshold: 1e-6,
    });
    printJson(res);
    process.exit(0);
  } catch (e) {
    failCli(e, e?.envelope ?? null);
  }
}

const layerIndex = Number(weightedIdx[0]);
const decl = spec.layers[layerIndex];
const WIRED_WEIGHTED = new Set(['linear', 'layerNorm', 'batchNorm', 'groupNorm', 'instanceNorm', 'rmsNorm', 'conv1d', 'conv2d', 'convTranspose2d', 'embedding']);
if (!decl || !WIRED_WEIGHTED.has(decl.constructor)) {
  failCli(`es_optimize supports weighted layers of a wired type (linear/norm/conv/embedding), got ${decl?.constructor}`);
}
const startWeights = [...spec.weights[weightedIdx[0]]];
const dim = startWeights.length;

try {
  const rt = await loadWasm();
  const ports = portBySlot(spec);

  const evalWeights = (weights) => {
    const s2 = JSON.parse(JSON.stringify(spec));
    s2.weights = { [weightedIdx[0]]: Array.from(weights) };
    const { reg, plan, g } = buildGraph(rt, s2);
    let mse = 0;
    try {
      for (const c of targets.cases) {
        const bundle = makeBundle(rt, plan, c.inputs, ports);
        const out = g.run(reg, bundle);
        const vals = Array.from(out.to_array());
        out.free();
        if (bundle.free) bundle.free();
        if (vals.length !== c.expected.length) {
          throw new Error(`case output length ${vals.length} != expected ${c.expected.length}`);
        }
        for (let i = 0; i < vals.length; i++) mse += (vals[i] - c.expected[i]) ** 2;
      }
    } finally {
      if (g?.free) g.free();
      if (plan?.free) plan.free();
      if (reg?.free) reg.free();
    }
    return -mse;
  };

  // Perturb around startWeights so existing good weights are preserved as the search center.
  const fitnessOfDelta = (deltaVec) => evalWeights(startWeights.map((w, i) => w + deltaVec[i]));

  const startFitness = evalWeights(startWeights);
  const es = rt.EsOptimizer.strict(dim, strategy, seed, pop, sigma, lr);
  const trajectory = [];
  for (let gen = 0; gen < gens; gen++) {
    const cand = es.ask();
    const n = es.batchSize();
    const fits = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      fits[i] = fitnessOfDelta(cand.slice(i * dim, (i + 1) * dim));
    }
    const rep = JSON.parse(es.tell(fits));
    trajectory.push({ gen, best: rep.best, mean: rep.mean });
  }
  const bestDelta = Array.from(es.best());
  const candBestWeights = startWeights.map((w, i) => w + bestDelta[i]);
  const esBestFitness = evalWeights(candBestWeights);
  es.free();

  // Monotonicity ratchet: never overwrite startWeights with a worse candidate.
  const improved = esBestFitness >= startFitness;
  const bestWeights = improved ? candBestWeights : startWeights;
  const finalFitness = improved ? esBestFitness : startFitness;

  const optimizedSpec = JSON.parse(JSON.stringify(spec));
  optimizedSpec.weights = { [weightedIdx[0]]: bestWeights };
  printJson({
    ok: true,
    dim,
    generations: gens,
    seed,
    strategy,
    start_fitness: startFitness,
    final_best_fitness: finalFitness,
    best_weights: bestWeights,
    trajectory,
    optimized_spec: optimizedSpec,
  });
} catch (e) {
  failCli(e, e?.envelope ?? null);
}
