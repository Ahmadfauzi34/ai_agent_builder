import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const outPath = process.argv[3] ? path.resolve(process.argv[3]) : null;
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function approxEqual(a, b, tol = 1e-5) {
  return Math.abs(a - b) <= tol;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function makeInput(values) {
  return new m.WasmTensor(
    new Float32Array(values),
    new Uint32Array([1, values.length, 1, 1]),
  );
}

function runScalar(graph, registry, input) {
  const output = graph.run(registry, input);
  const values = output.to_array();
  output.free();
  assert(values.length === 1, 'scalar regression graph must return one value');
  assert(Number.isFinite(values[0]), 'graph produced non-finite scalar output');
  return values[0];
}

function buildRegressionGraph() {
  const registry = new m.LayerRegistry();
  const linear = m.AgentLayerSpec.linear(4101, 2, 1, true);
  registry.initAgentLayer(linear);

  const builder = new m.AgentGraphBuilder(2);
  builder.addUnary(linear, 0, 1);
  builder.setOutput(1);
  const graph = builder.compile(registry);
  const layout = JSON.parse(m.graphParameterLayout(graph, registry));

  assert(layout.owners.length === 1, 'regression graph should have exactly one trainable owner');
  assert(layout.total_len === 3, '2->1 Linear+bias should expose exactly three graph parameters');
  return { registry, graph, layout };
}

function buildRegressionDataset() {
  const points = [
    [-1, -1], [-1, 0], [-1, 1], [0, -1],
    [0, 0], [0, 1], [1, -1], [1, 0],
    [1, 1], [0.5, -0.5], [-0.5, 0.5], [2, -1],
    [-2, 1], [1.5, 0.25], [-1.5, -0.25], [0.25, 1.5],
  ];
  return points.map(([x0, x1]) => ({
    input: makeInput([x0, x1]),
    target: 1.5 * x0 - 0.75 * x1 + 0.25,
  }));
}

function regressionLoss(graph, registry, dataset, count = dataset.length) {
  let squared = 0;
  for (let i = 0; i < count; i += 1) {
    const prediction = runScalar(graph, registry, dataset[i].input);
    const error = prediction - dataset[i].target;
    squared += error * error;
  }
  return squared / count;
}

function applyCandidate(graph, registry, candidate) {
  m.setGraphParametersFlat(graph, registry, new Float32Array(candidate));
}

function exactFloatArray(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

function semanticTrainingProof(dataset) {
  const trained = buildRegressionGraph();
  const programIdentity = trained.graph.programIdentity();
  const bindingIdentity = m.graphParameterIdentity(trained.graph, trained.registry);
  const zero = new Float32Array(trained.layout.total_len);
  applyCandidate(trained.graph, trained.registry, zero);
  const initialLoss = regressionLoss(trained.graph, trained.registry, dataset);

  const generations = 60;
  const population = 32;
  const optimizer = m.EsOptimizer.strict(
    trained.layout.total_len,
    0,
    777,
    population,
    0.2,
    0.05,
  );

  for (let generation = 0; generation < generations; generation += 1) {
    const flat = Array.from(optimizer.ask());
    const batchSize = optimizer.batchSize();
    assert(batchSize === population, 'unexpected ES population size');
    assert(flat.length === batchSize * trained.layout.total_len, 'ask batch dimension mismatch');

    const fitness = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i += 1) {
      const start = i * trained.layout.total_len;
      const candidate = flat.slice(start, start + trained.layout.total_len);
      assert(candidate.every(Number.isFinite), 'ES emitted non-finite candidate');
      applyCandidate(trained.graph, trained.registry, candidate);
      const loss = regressionLoss(trained.graph, trained.registry, dataset);
      assert(Number.isFinite(loss), 'objective produced non-finite loss');
      fitness[i] = -loss;
    }
    optimizer.tell(fitness);
  }

  assert(optimizer.generation() === generations, 'ES generation counter mismatch');
  const best = Array.from(optimizer.best());
  assert(best.length === trained.layout.total_len, 'best candidate dimension mismatch');
  assert(best.every(Number.isFinite), 'best candidate contains non-finite values');
  applyCandidate(trained.graph, trained.registry, best);
  const finalLoss = regressionLoss(trained.graph, trained.registry, dataset);

  assert(finalLoss < initialLoss * 0.02, `training did not materially reduce loss: ${initialLoss} -> ${finalLoss}`);
  assert(finalLoss < 0.05, `final regression loss is unexpectedly high: ${finalLoss}`);
  assert(trained.graph.programIdentity() === programIdentity, 'mutable training changed program identity');
  assert(m.graphParameterIdentity(trained.graph, trained.registry) === bindingIdentity, 'mutable training changed binding identity');
  assert(
    exactFloatArray(Array.from(m.getGraphParametersFlat(trained.graph, trained.registry)), best),
    'best candidate did not replay exactly on trained graph',
  );

  const replay = buildRegressionGraph();
  assert(replay.graph.programIdentity() === programIdentity, 'fresh equivalent graph has different program identity');
  assert(m.graphParameterIdentity(replay.graph, replay.registry) === bindingIdentity, 'fresh equivalent graph has different binding identity');
  applyCandidate(replay.graph, replay.registry, best);
  const replayLoss = regressionLoss(replay.graph, replay.registry, dataset);
  assert(approxEqual(replayLoss, finalLoss, 1e-6), `fresh replay loss mismatch: ${finalLoss} vs ${replayLoss}`);

  const trainedOutputs = dataset.map((row) => runScalar(trained.graph, trained.registry, row.input));
  const replayOutputs = dataset.map((row) => runScalar(replay.graph, replay.registry, row.input));
  for (let i = 0; i < trainedOutputs.length; i += 1) {
    assert(approxEqual(trainedOutputs[i], replayOutputs[i], 1e-6), `fresh replay output mismatch at sample ${i}`);
  }

  const report = JSON.parse(optimizer.report());
  assert(report.gen === generations, 'ES report generation mismatch');
  assert(report.evals === population, 'ES report eval count mismatch');

  return {
    generations,
    population,
    parameterCount: trained.layout.total_len,
    initialLoss,
    finalLoss,
    improvementFactor: initialLoss / finalLoss,
    best,
    replayLoss,
    programIdentityStable: true,
    bindingIdentityStable: true,
    freshReplayMatched: true,
  };
}

function buildManyOwnerGraph() {
  const width = 8;
  const ownerCount = 8;
  const registry = new m.LayerRegistry();
  const specs = [];
  for (let i = 0; i < ownerCount; i += 1) {
    const spec = m.AgentLayerSpec.linear(5000 + i, width, width, true);
    registry.initAgentLayer(spec);
    specs.push(spec);
  }

  const builder = new m.AgentGraphBuilder(ownerCount + 1);
  for (let i = 0; i < ownerCount; i += 1) {
    builder.addUnary(specs[i], i, i + 1);
  }
  builder.setOutput(ownerCount);
  const graph = builder.compile(registry);
  const layout = JSON.parse(m.graphParameterLayout(graph, registry));
  assert(layout.owners.length === ownerCount, 'many-owner graph owner count mismatch');
  return { registry, graph, layout, width, ownerCount };
}

function buildVectorDataset(width, count = 32) {
  const rows = [];
  for (let row = 0; row < count; row += 1) {
    const values = new Float32Array(width);
    for (let col = 0; col < width; col += 1) {
      values[col] = ((((row + 1) * (col + 3) * 7) % 23) - 11) / 11;
    }
    rows.push(makeInput(values));
  }
  return rows;
}

function vectorZeroLoss(graph, registry, inputs, count) {
  let squared = 0;
  for (let i = 0; i < count; i += 1) {
    const output = graph.run(registry, inputs[i]);
    const values = output.to_array();
    output.free();
    for (const value of values) {
      assert(Number.isFinite(value), 'many-owner objective produced non-finite output');
      squared += value * value;
    }
  }
  return squared / count;
}

function measuredGeneration({ optimizer, layout, graph, registry, evalCount, objective }) {
  let start = performance.now();
  const flat = Array.from(optimizer.ask());
  const askMs = performance.now() - start;
  const batchSize = optimizer.batchSize();
  assert(flat.length === batchSize * layout.total_len, 'measured ask dimension mismatch');

  let applyMs = 0;
  let forwardObjectiveMs = 0;
  const fitness = new Float32Array(batchSize);

  for (let i = 0; i < batchSize; i += 1) {
    const offset = i * layout.total_len;
    const candidate = flat.slice(offset, offset + layout.total_len);

    start = performance.now();
    applyCandidate(graph, registry, candidate);
    applyMs += performance.now() - start;

    start = performance.now();
    const loss = objective(evalCount);
    forwardObjectiveMs += performance.now() - start;
    assert(Number.isFinite(loss), 'measured objective returned non-finite loss');
    fitness[i] = -loss;
  }

  start = performance.now();
  optimizer.tell(fitness);
  const tellMs = performance.now() - start;
  const candidateWorkMs = applyMs + forwardObjectiveMs;
  const generationMs = askMs + candidateWorkMs + tellMs;

  return {
    askMs,
    applyMs,
    forwardObjectiveMs,
    tellMs,
    generationMs,
    applyShareOfCandidateWork: candidateWorkMs > 0 ? applyMs / candidateWorkMs : 0,
    applyShareOfGeneration: generationMs > 0 ? applyMs / generationMs : 0,
  };
}

function summarizeGenerations(rows) {
  const keys = [
    'askMs',
    'applyMs',
    'forwardObjectiveMs',
    'tellMs',
    'generationMs',
    'applyShareOfCandidateWork',
    'applyShareOfGeneration',
  ];
  const result = {};
  for (const key of keys) result[key] = median(rows.map((row) => row[key]));
  return result;
}

function profileRegressionAmortization(dataset) {
  const results = {};
  for (const evalCount of [1, 8, 16]) {
    const workload = buildRegressionGraph();
    const optimizer = m.EsOptimizer.strict(workload.layout.total_len, 0, 1200 + evalCount, 32, 0.2, 0.05);
    const objective = (count) => regressionLoss(workload.graph, workload.registry, dataset, count);
    measuredGeneration({ optimizer, ...workload, evalCount, objective });
    const rows = [];
    for (let repeat = 0; repeat < 5; repeat += 1) {
      rows.push(measuredGeneration({ optimizer, ...workload, evalCount, objective }));
    }
    results[String(evalCount)] = summarizeGenerations(rows);
  }
  return results;
}

function profileManyOwnerAmortization() {
  const results = {};
  const workload = buildManyOwnerGraph();
  const inputs = buildVectorDataset(workload.width, 32);
  try {
    for (const evalCount of [1, 8, 32]) {
      const optimizer = m.EsOptimizer.strict(workload.layout.total_len, 0, 2200 + evalCount, 16, 0.02, 0.03);
      const objective = (count) => vectorZeroLoss(workload.graph, workload.registry, inputs, count);
      measuredGeneration({ optimizer, ...workload, evalCount, objective });
      const rows = [];
      for (let repeat = 0; repeat < 5; repeat += 1) {
        rows.push(measuredGeneration({ optimizer, ...workload, evalCount, objective }));
      }
      results[String(evalCount)] = summarizeGenerations(rows);
    }
  } finally {
    for (const input of inputs) input.free();
  }
  return {
    graphSteps: workload.ownerCount,
    uniqueTrainableOwners: workload.layout.owners.length,
    parameterCount: workload.layout.total_len,
    evaluationsPerCandidate: results,
  };
}

const regressionDataset = buildRegressionDataset();
let result;
try {
  const semanticTraining = semanticTrainingProof(regressionDataset);
  const regressionAmortization = profileRegressionAmortization(regressionDataset);
  const manyOwnerAmortization = profileManyOwnerAmortization();

  result = {
    schema: 'burn-research.es-graph-objective-loop.v1',
    issue: 173,
    verdict: 'PASS',
    boundary: {
      productionControllerAdded: false,
      reusableBindingAdded: false,
      publicWasmSurfaceChanged: false,
      optimizerDimensionSource: 'graphParameterLayout.total_len',
      candidateOrderingSource: 'canonical graph parameter binding',
      lifecycle: 'ask -> apply -> graph run(s) -> objective -> tell',
    },
    semanticTraining,
    amortization: {
      timingIsEvidenceNotThreshold: true,
      regression: regressionAmortization,
      manyOwner: manyOwnerAmortization,
    },
  };
} finally {
  for (const row of regressionDataset) row.input.free();
}

const text = `${JSON.stringify(result, null, 2)}\n`;
if (outPath) fs.writeFileSync(outPath, text);
console.log(text);
