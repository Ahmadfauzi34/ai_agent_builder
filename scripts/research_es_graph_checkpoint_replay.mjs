import crypto from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function approxEqual(a, b, tol = 1e-6) {
  return Math.abs(a - b) <= tol;
}

function exactArray(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

function exactBytes(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

function sha256(bytes) {
  return crypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex');
}

function expectThrow(fn, message) {
  let threw = false;
  try {
    fn();
  } catch {
    threw = true;
  }
  assert(threw, message);
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
  assert(values.length === 1, 'checkpoint regression graph must return one scalar');
  assert(Number.isFinite(values[0]), 'checkpoint regression graph produced non-finite output');
  return values[0];
}

function buildRegressionGraph(layerId = 6101) {
  const registry = new m.LayerRegistry();
  const linear = m.AgentLayerSpec.linear(layerId, 2, 1, true);
  registry.initAgentLayer(linear);

  const builder = new m.AgentGraphBuilder(2);
  builder.addUnary(linear, 0, 1);
  builder.setOutput(1);
  const graph = builder.compile(registry);
  const layout = JSON.parse(m.graphParameterLayout(graph, registry));

  assert(layout.owners.length === 1, 'checkpoint regression graph should have one trainable owner');
  assert(layout.total_len === 3, '2->1 Linear+bias should expose exactly three graph parameters');
  return { registry, graph, layout };
}

function buildDataset() {
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

function outputs(graph, registry, dataset) {
  return dataset.map((row) => runScalar(graph, registry, row.input));
}

function regressionLoss(graph, registry, dataset) {
  let squared = 0;
  for (const row of dataset) {
    const prediction = runScalar(graph, registry, row.input);
    const error = prediction - row.target;
    squared += error * error;
  }
  return squared / dataset.length;
}

function applyCandidate(graph, registry, candidate) {
  m.setGraphParametersFlat(graph, registry, new Float32Array(candidate));
}

function trainBest(dataset) {
  const trained = buildRegressionGraph();
  const zero = new Float32Array(trained.layout.total_len);
  applyCandidate(trained.graph, trained.registry, zero);

  const programIdentity = trained.graph.programIdentity();
  const bindingIdentity = m.graphParameterIdentity(trained.graph, trained.registry);
  const layoutJson = m.graphParameterLayout(trained.graph, trained.registry);
  const initialLoss = regressionLoss(trained.graph, trained.registry, dataset);

  const generations = 60;
  const population = 32;
  const optimizer = m.EsOptimizer.strict(
    trained.layout.total_len,
    0,
    1777,
    population,
    0.2,
    0.05,
  );

  for (let generation = 0; generation < generations; generation += 1) {
    const flat = Array.from(optimizer.ask());
    const batchSize = optimizer.batchSize();
    assert(batchSize === population, 'checkpoint ES population size mismatch');
    assert(flat.length === batchSize * trained.layout.total_len, 'checkpoint ES ask dimension mismatch');

    const fitness = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i += 1) {
      const start = i * trained.layout.total_len;
      const candidate = flat.slice(start, start + trained.layout.total_len);
      assert(candidate.every(Number.isFinite), 'checkpoint ES emitted non-finite candidate');
      applyCandidate(trained.graph, trained.registry, candidate);
      const loss = regressionLoss(trained.graph, trained.registry, dataset);
      assert(Number.isFinite(loss), 'checkpoint objective produced non-finite loss');
      fitness[i] = -loss;
    }
    optimizer.tell(fitness);
  }

  const best = Array.from(optimizer.best());
  assert(best.length === trained.layout.total_len, 'checkpoint best dimension mismatch');
  assert(best.every(Number.isFinite), 'checkpoint best candidate contains non-finite values');
  applyCandidate(trained.graph, trained.registry, best);

  const finalLoss = regressionLoss(trained.graph, trained.registry, dataset);
  assert(finalLoss < initialLoss * 0.02, `checkpoint training did not materially improve loss: ${initialLoss} -> ${finalLoss}`);
  assert(finalLoss < 0.05, `checkpoint training final loss is unexpectedly high: ${finalLoss}`);
  assert(trained.graph.programIdentity() === programIdentity, 'training changed structural program identity');
  assert(m.graphParameterIdentity(trained.graph, trained.registry) === bindingIdentity, 'training changed graph-parameter binding identity');
  assert(m.graphParameterLayout(trained.graph, trained.registry) === layoutJson, 'training changed graph-parameter layout');

  const readBack = Array.from(m.getGraphParametersFlat(trained.graph, trained.registry));
  assert(exactArray(readBack, best), 'trained best candidate did not read back exactly');

  return {
    ...trained,
    generations,
    population,
    programIdentity,
    bindingIdentity,
    layoutJson,
    initialLoss,
    finalLoss,
    best,
    trainedOutputs: outputs(trained.graph, trained.registry, dataset),
  };
}

function proveStatefulBundleReplay(trained, dataset) {
  const bundleA = m.exportProgramBundle(trained.graph, trained.registry, true);
  const bundleB = m.exportProgramBundle(trained.graph, trained.registry, true);
  assert(bundleA.length > 0, 'stateful ProgramBundle must not be empty');

  const repeatedExportExact = exactBytes(bundleA, bundleB);
  const digestA = sha256(bundleA);
  const digestB = sha256(bundleB);

  const importedRegistry = new m.LayerRegistry();
  const importedGraph = m.importProgramBundle(importedRegistry, bundleA);

  const importedProgramIdentity = importedGraph.programIdentity();
  const importedBindingIdentity = m.graphParameterIdentity(importedGraph, importedRegistry);
  const importedLayoutJson = m.graphParameterLayout(importedGraph, importedRegistry);
  const importedFlat = Array.from(m.getGraphParametersFlat(importedGraph, importedRegistry));
  const importedLoss = regressionLoss(importedGraph, importedRegistry, dataset);
  const importedOutputs = outputs(importedGraph, importedRegistry, dataset);

  assert(importedProgramIdentity === trained.programIdentity, 'stateful bundle replay changed program identity');
  assert(importedBindingIdentity === trained.bindingIdentity, 'stateful bundle replay changed graph-parameter binding identity');
  assert(importedLayoutJson === trained.layoutJson, 'stateful bundle replay changed graph-parameter layout');
  assert(exactArray(importedFlat, trained.best), 'stateful bundle replay did not restore the exact best flat candidate');
  assert(approxEqual(importedLoss, trained.finalLoss), `stateful bundle replay loss mismatch: ${trained.finalLoss} vs ${importedLoss}`);
  assert(importedOutputs.length === trained.trainedOutputs.length, 'stateful bundle replay output cardinality mismatch');
  for (let i = 0; i < importedOutputs.length; i += 1) {
    assert(
      approxEqual(importedOutputs[i], trained.trainedOutputs[i]),
      `stateful bundle replay output mismatch at sample ${i}: ${trained.trainedOutputs[i]} vs ${importedOutputs[i]}`,
    );
  }

  const importedBundle = m.exportProgramBundle(importedGraph, importedRegistry, true);
  const importedReexportExact = exactBytes(bundleA, importedBundle);

  return {
    bundleBytes: bundleA.length,
    sha256: digestA,
    repeatedExportSha256: digestB,
    repeatedExportExact,
    importedReexportSha256: sha256(importedBundle),
    importedReexportExact,
    importedLoss,
    importedProgramIdentityMatched: true,
    importedBindingIdentityMatched: true,
    importedLayoutMatched: true,
    importedBestMatchedExactly: true,
    importedOutputsMatched: true,
  };
}

function proveMalformedImportAtomicity(learnedBundle) {
  assert(learnedBundle.length > 1, 'learned ProgramBundle is too short for truncation proof');

  const sentinel = buildRegressionGraph(9901);
  applyCandidate(sentinel.graph, sentinel.registry, [0.5, -0.25, 0.125]);
  const sentinelIdentity = sentinel.graph.programIdentity();
  const sentinelBindingIdentity = m.graphParameterIdentity(sentinel.graph, sentinel.registry);
  const sentinelFlatBefore = Array.from(m.getGraphParametersFlat(sentinel.graph, sentinel.registry));
  const sentinelBundleBefore = m.exportProgramBundle(sentinel.graph, sentinel.registry, true);

  const truncated = learnedBundle.slice(0, learnedBundle.length - 1);
  expectThrow(
    () => m.importProgramBundle(sentinel.registry, truncated),
    'truncated learned ProgramBundle import must fail closed',
  );

  assert(sentinel.graph.programIdentity() === sentinelIdentity, 'failed bundle import changed sentinel program identity');
  assert(
    m.graphParameterIdentity(sentinel.graph, sentinel.registry) === sentinelBindingIdentity,
    'failed bundle import changed sentinel binding identity',
  );
  assert(
    exactArray(Array.from(m.getGraphParametersFlat(sentinel.graph, sentinel.registry)), sentinelFlatBefore),
    'failed bundle import changed sentinel trainable parameters',
  );
  const sentinelBundleAfter = m.exportProgramBundle(sentinel.graph, sentinel.registry, true);
  assert(
    exactBytes(sentinelBundleBefore, sentinelBundleAfter),
    'failed bundle import changed exact sentinel stateful bundle bytes',
  );

  return {
    truncatedBytes: truncated.length,
    targetProgramIdentityStable: true,
    targetBindingIdentityStable: true,
    targetFlatParametersStable: true,
    targetStatefulBundleBytesStable: true,
  };
}

const dataset = buildDataset();
let result;
try {
  const trained = trainBest(dataset);
  const learnedBundle = m.exportProgramBundle(trained.graph, trained.registry, true);
  const replay = proveStatefulBundleReplay(trained, dataset);
  const malformedImportAtomicity = proveMalformedImportAtomicity(learnedBundle);

  result = {
    schema: 'burn-research.es-graph-checkpoint-replay.v1',
    issue: 175,
    verdict: 'PASS',
    decision: 'KEEP_EXISTING_STATEFUL_PROGRAM_BUNDLE',
    boundary: {
      productionControllerAdded: false,
      newCheckpointSchemaAdded: false,
      newStateIdentityAdded: false,
      publicWasmSurfaceChanged: false,
      optimizerDimensionSource: 'graphParameterLayout.total_len',
      candidateOrderingSource: 'canonical graph parameter binding',
      checkpointArtifact: 'ProgramBundle(include_state=true)',
      bundleSha256Role: 'evidence_only_not_protocol_identity',
    },
    training: {
      generations: trained.generations,
      population: trained.population,
      parameterCount: trained.layout.total_len,
      initialLoss: trained.initialLoss,
      finalLoss: trained.finalLoss,
      improvementFactor: trained.initialLoss / trained.finalLoss,
      best: trained.best,
      programIdentityStable: true,
      bindingIdentityStable: true,
      layoutStable: true,
    },
    replay,
    malformedImportAtomicity,
  };
} finally {
  for (const row of dataset) row.input.free();
}

console.log(`${JSON.stringify(result, null, 2)}\n`);
