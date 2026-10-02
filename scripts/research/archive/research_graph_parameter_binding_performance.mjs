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

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const index = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[index];
}

function summarize(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    min: sorted[0],
    p25: percentile(sorted, 0.25),
    median: percentile(sorted, 0.50),
    p75: percentile(sorted, 0.75),
    p95: percentile(sorted, 0.95),
    max: sorted[sorted.length - 1],
  };
}

function measurePerOp(fn, { iterations, samples = 7, warmup = 8 }) {
  for (let i = 0; i < warmup; i += 1) fn();
  const perOp = [];
  for (let sample = 0; sample < samples; sample += 1) {
    const start = performance.now();
    for (let i = 0; i < iterations; i += 1) fn();
    perOp.push((performance.now() - start) / iterations);
  }
  return { unit: 'ms/op', iterations, samples, ...summarize(perOp) };
}

function measureBatch(fn, { samples = 5, warmup = 2 } = {}) {
  for (let i = 0; i < warmup; i += 1) fn();
  const totals = [];
  for (let sample = 0; sample < samples; sample += 1) {
    const start = performance.now();
    fn();
    totals.push(performance.now() - start);
  }
  return { unit: 'ms/batch', samples, ...summarize(totals) };
}

function buildChain({ name, width, uniqueOwners, repeatedSteps = null, fill = 0.001 }) {
  const registry = new m.LayerRegistry();
  const steps = repeatedSteps ?? uniqueOwners;
  const specs = [];
  if (repeatedSteps !== null) {
    const spec = m.AgentLayerSpec.linear(1000, width, width, true);
    registry.initAgentLayer(spec);
    specs.push(spec);
  } else {
    for (let i = 0; i < uniqueOwners; i += 1) {
      const spec = m.AgentLayerSpec.linear(1000 + i, width, width, true);
      registry.initAgentLayer(spec);
      specs.push(spec);
    }
  }

  const builder = new m.AgentGraphBuilder(steps + 1);
  for (let i = 0; i < steps; i += 1) {
    const spec = repeatedSteps !== null ? specs[0] : specs[i];
    builder.addUnary(spec, i, i + 1);
  }
  builder.setOutput(steps);
  const graph = builder.compile(registry);
  const layout = JSON.parse(m.graphParameterLayout(graph, registry));
  const candidate = new Float32Array(layout.total_len);
  candidate.fill(fill);
  m.setGraphParametersFlat(graph, registry, candidate);

  const inputValues = new Float32Array(width);
  inputValues.fill(0.5);
  const input = new m.WasmTensor(inputValues, new Uint32Array([1, width, 1, 1]));

  assert(layout.owners.length === uniqueOwners, `${name}: unexpected owner count`);
  assert(layout.total_len === candidate.length, `${name}: layout/candidate length mismatch`);
  assert(candidate.every(Number.isFinite), `${name}: candidate must be finite`);

  const smoke = graph.run(registry, input);
  const smokeValues = Array.from(smoke.to_array());
  smoke.free();
  assert(smokeValues.every(Number.isFinite), `${name}: forward smoke test produced non-finite values`);

  return { name, width, uniqueOwners, steps, registry, graph, layout, candidate, input };
}

function forwardOnce(w) {
  const out = w.graph.run(w.registry, w.input);
  const values = out.to_array();
  for (let i = 0; i < values.length; i += 1) {
    if (!Number.isFinite(values[i])) {
      out.free();
      throw new Error(`${w.name}: forward produced non-finite output during measurement`);
    }
  }
  out.free();
}

function layoutOnce(w) {
  const raw = m.graphParameterLayout(w.graph, w.registry);
  if (raw.length === 0) throw new Error(`${w.name}: empty layout`);
}

function readOnce(w) {
  const values = m.getGraphParametersFlat(w.graph, w.registry);
  if (values.length !== w.layout.total_len) throw new Error(`${w.name}: read length drift`);
}

function applyOnce(w) {
  m.setGraphParametersFlat(w.graph, w.registry, w.candidate);
}

function combinedOnce(w) {
  applyOnce(w);
  forwardOnce(w);
}

const workloadConfigs = [
  { name: 'small_2x2_two_owner', width: 2, uniqueOwners: 2, iterations: 160 },
  { name: 'many_owner_16x16', width: 16, uniqueOwners: 16, iterations: 50 },
  { name: 'few_large_128x128', width: 128, uniqueOwners: 2, iterations: 20 },
  { name: 'repeated_owner_16x16_x16', width: 16, uniqueOwners: 1, repeatedSteps: 16, iterations: 80 },
];

const workloads = [];
for (const config of workloadConfigs) {
  const w = buildChain(config);
  const layoutBuildProxy = measurePerOp(() => layoutOnce(w), { iterations: config.iterations });
  const read = measurePerOp(() => readOnce(w), { iterations: config.iterations });
  const apply = measurePerOp(() => applyOnce(w), { iterations: config.iterations });
  const forward = measurePerOp(() => forwardOnce(w), { iterations: config.iterations });
  const combined = measurePerOp(() => combinedOnce(w), { iterations: Math.max(8, Math.floor(config.iterations / 2)) });

  const populations = {};
  for (const population of [8, 32, 128]) {
    const batch = measureBatch(() => {
      for (let i = 0; i < population; i += 1) combinedOnce(w);
    });
    populations[String(population)] = {
      ...batch,
      perCandidateMedian: batch.median / population,
    };
  }

  const baselineTotal = apply.median + forward.median;
  workloads.push({
    name: w.name,
    width: w.width,
    graphSteps: w.steps,
    uniqueTrainableOwners: w.layout.owners.length,
    parameterCount: w.layout.total_len,
    timings: { layoutBuildProxy, read, apply, forward, combined },
    ratios: {
      applyShareOfApplyPlusForward: baselineTotal > 0 ? apply.median / baselineTotal : 0,
      layoutBuildProxyToApply: apply.median > 0 ? layoutBuildProxy.median / apply.median : 0,
      layoutBuildProxyToForward: forward.median > 0 ? layoutBuildProxy.median / forward.median : 0,
      readToForward: forward.median > 0 ? read.median / forward.median : 0,
    },
    populations,
  });
}

const result = {
  schema: 'burn-research.graph-parameter-binding-performance.v1',
  issue: 168,
  verdict: 'MEASURED',
  semantics: {
    productionCachingAdded: false,
    publicSurfaceChanged: false,
    finiteOnlyBaselineExpected: true,
    layoutBuildProxyNote: 'graphParameterLayout includes GraphParameterBinding::build plus JSON serialization; it is a public-path proxy, not an internal profiler.',
    applyNote: 'setGraphParametersFlat includes adapter build, apply validation/rebuild, finite scan, per-owner checks, and setters.',
  },
  environment: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
  },
  workloads,
};

const text = `${JSON.stringify(result, null, 2)}\n`;
if (outPath) fs.writeFileSync(outPath, text);
console.log(text);
