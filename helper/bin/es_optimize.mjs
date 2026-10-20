#!/usr/bin/env node
// burn-math es_optimize: evolution-strategies optimization of a spec's
// weights, judged entirely by the engine itself.
//
// The engine's EsOptimizer proposes candidate weight vectors (ask); each
// candidate is scored by RUNNING the graph through the engine on target
// cases (fitness = -sum MSE vs expected outputs); scores go back via tell.
// No host-side math decides anything — the engine runs every evaluation.
//
// Usage: node es_optimize.mjs <spec.json> <targets.json> [gens=40] [pop] [sigma] [lr] [seed=1] [strategy=0]
//   targets.json: {"cases":[{"inputs":{slot:[...]}, "expected":[...]}, ...]}
// Scope: the spec must carry exactly ONE weighted layer, of a type this
// helper's buildGraph wires weights for (linear, the norm family, convs,
// embedding). The optimizer starts from a zero mean; the spec's own weights
// define the dimension and are reported as the starting point's fitness
// for comparison.
// Prints {ok, dim, generations, seed, strategy, start_fitness,
// final_best_fitness, best_weights, optimized_spec}.
// Exit 0 on completion; 2 on usage/spec errors.
import fs from 'node:fs';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, failCli } from '../lib/runner.mjs';

const [specFile, targetsFile, gensArg, popArg, sigmaArg, lrArg, seedArg, strategyArg] =
  process.argv.slice(2);
if (!specFile || !targetsFile) {
  console.error('usage: node es_optimize.mjs <spec.json> <targets.json> [gens] [pop] [sigma] [lr] [seed] [strategy]');
  process.exit(2);
}
const gens = gensArg !== undefined ? Number(gensArg) : 40;
const seed = seedArg !== undefined ? Number(seedArg) : 1;
const strategy = strategyArg !== undefined ? Number(strategyArg) : 0;
const pop = popArg !== undefined ? Number(popArg) : null;
const sigma = sigmaArg !== undefined ? Number(sigmaArg) : null;
const lr = lrArg !== undefined ? Number(lrArg) : null;

let spec, targets;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }
try { targets = JSON.parse(fs.readFileSync(targetsFile, 'utf8')); }
catch (e) { failCli(`cannot read targets: ${e.message}`); }
if (!Array.isArray(targets.cases) || targets.cases.length === 0) {
  failCli('targets.json must hold a non-empty "cases" array');
}

const weightedIdx = Object.keys(spec.weights ?? {});
if (weightedIdx.length !== 1) {
  failCli(`es_optimize needs exactly one weighted layer, spec has ${weightedIdx.length}`);
}
const layerIndex = Number(weightedIdx[0]);
const decl = spec.layers[layerIndex];
const WIRED_WEIGHTED = new Set(['linear', 'layerNorm', 'batchNorm', 'groupNorm', 'instanceNorm', 'rmsNorm', 'conv1d', 'conv2d', 'convTranspose2d', 'embedding']);
if (!decl || !WIRED_WEIGHTED.has(decl.constructor)) {
  failCli(`es_optimize supports one weighted layer of a wired type (linear/norm/conv/embedding), got ${decl?.constructor}`);
}
const startWeights = spec.weights[weightedIdx[0]];
const dim = startWeights.length;

try {
  const rt = await loadWasm();
  const ports = portBySlot(spec);

  const fitnessOf = (weights) => {
    const s2 = JSON.parse(JSON.stringify(spec));
    s2.weights = { [weightedIdx[0]]: Array.from(weights) };
    const { reg, plan, g } = buildGraph(rt, s2);
    let mse = 0;
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
    if (g.free) g.free();
    return -mse;
  };

  const startFitness = fitnessOf(startWeights);
  const es = rt.EsOptimizer.strict(dim, strategy, seed, pop, sigma, lr);
  const trajectory = [];
  for (let gen = 0; gen < gens; gen++) {
    const cand = es.ask();
    const n = es.batchSize();
    const fits = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      fits[i] = fitnessOf(cand.slice(i * dim, (i + 1) * dim));
    }
    const rep = JSON.parse(es.tell(fits));
    trajectory.push({ gen, best: rep.best, mean: rep.mean });
  }
  const bestWeights = Array.from(es.best());
  const finalFitness = fitnessOf(bestWeights);
  es.free();

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
  failCli(e.message, e.envelope ?? null);
}
