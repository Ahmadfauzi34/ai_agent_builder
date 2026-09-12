import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapter = await import(pathToFileURL(path.join(pkgDir, 'node.mjs')).href);
const m = await adapter.loadBurnRuntime();

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function dot(a, b) {
  assert(a.length === b.length, 'dot length mismatch');
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function normalize(norm, values) {
  const input = new m.WasmTensor(new Float32Array(values), new Uint32Array([1, values.length, 1, 1]));
  const output = norm.forward(input);
  const result = Array.from(output.to_array());
  output.free();
  input.free();
  return result;
}

// Query wants direction [1, 0]. A larger but less-aligned distractor wins raw
// dot-product only because of magnitude. L2 feature normalization should make
// direction, rather than scale, determine the ranking.
const query = [1.0, 0.0];
const relevant = [1.0, 0.1];
const distractor = [2.0, 2.0];

const rawRelevant = dot(query, relevant);
const rawDistractor = dot(query, distractor);
assert(rawDistractor > rawRelevant, 'fixture must expose magnitude-biased raw ranking');

const norm = m.WasmFeatureNorm.newFeatureNorm();
const normalizedQuery = normalize(norm, query);
const normalizedRelevant = normalize(norm, relevant);
const normalizedDistractor = normalize(norm, distractor);
const normalizedRelevantScore = dot(normalizedQuery, normalizedRelevant);
const normalizedDistractorScore = dot(normalizedQuery, normalizedDistractor);

assert(
  normalizedRelevantScore > normalizedDistractorScore,
  `FeatureNorm did not repair ranking: relevant=${normalizedRelevantScore}, distractor=${normalizedDistractorScore}`,
);
assert(norm.num_params() === 0, 'FeatureNorm must remain parameter-free');

const zero = normalize(norm, [0.0, 0.0]);
assert(zero.every((value) => Number.isFinite(value) && value === 0), `zero vector was not stable: ${zero}`);

let badShapeRejected = false;
const bad = new m.WasmTensor(new Float32Array([1, 2, 3, 4]), new Uint32Array([1, 2, 2, 1]));
try {
  norm.forward(bad);
} catch {
  badShapeRejected = true;
}
bad.free();
assert(badShapeRejected, 'FeatureNorm must reject non-[B,F,1,1] layout');

let badEpsilonRejected = false;
try {
  m.WasmFeatureNorm.newFeatureNorm(0);
} catch {
  badEpsilonRejected = true;
}
assert(badEpsilonRejected, 'FeatureNorm must reject non-positive epsilon');

console.log(JSON.stringify({
  verdict: 'PASS',
  task: 'embedding-memory retrieval ranking under magnitude bias',
  raw: {
    relevant: rawRelevant,
    distractor: rawDistractor,
    winner: 'distractor',
  },
  normalized: {
    query: normalizedQuery,
    relevant: normalizedRelevant,
    distractor: normalizedDistractor,
    relevantScore: normalizedRelevantScore,
    distractorScore: normalizedDistractorScore,
    winner: 'relevant',
  },
  zeroVectorSafe: true,
  badShapeRejected,
  badEpsilonRejected,
  parameterCount: norm.num_params(),
}, null, 2));

norm.free();
