#!/usr/bin/env node
// Build the distributable burn-math agent pack FROM this repo.
// The repo is the single source of truth; the pack is its output, never
// a side copy edited in place.
//
// Usage:
//   node scripts/build_agent_pack.mjs --out <dir> [--engine <dir>] [--version <str>]
//     --engine: full engine bundle dir (wasm + js + d.ts + docs/ + runner
//       script). Default: $BURN_WASM_PKG_DIR, else <repo>/pkg.
//     --version: pack version string. Default: repo-<shortsha>-<yyyymmdd>.
//   Prints a JSON summary on stdout (exit 0) or an error on stderr (exit 1):
//     {ok, out, pack_version, catalog_entries, surface_entries,
//      layout_entries, wasm_sha256, js_sha256}
//
// What it does:
//   1. Validates the engine bundle (all files the pack's tools read).
//   2. Copies helper/{bin,lib,AGENT.md,INDEX.md,spec-schema.md,reference,
//      examples} to <out>/ and the engine bundle to <out>/engine-audit/.
//   3. Generates <out>/contracts-index/*.jsonl from the engine docs.
//   4. Counts catalog/surface/layout entries via the packed tools.
//   5. Smoke-runs the packed run tool on examples/linear-relu.json.
//   6. Writes <out>/MANIFEST.json and refreshes helper/MANIFEST.json
//      (same sha256 pins; helper/MANIFEST.json is what lib/wasm_direct.mjs
//      integrity-checks, so it must be committed whenever the engine
//      changes — CI runs this builder and fails if it is stale).
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const HELPER = path.join(ROOT, 'helper');

function fail(msg) {
  console.error(`build_agent_pack: ${msg}`);
  process.exit(1);
}
function sha256File(f) {
  return createHash('sha256').update(fs.readFileSync(f)).digest('hex');
}
function runNode(args, env = {}) {
  return spawnSync('node', args, {
    encoding: 'utf8',
    timeout: 300000,
    maxBuffer: 256 * 1024 * 1024,
    env: { ...process.env, ...env },
  });
}
function countLines(out) {
  return out.trim().split('\n').filter((l) => l.trim().length > 0).length;
}

const argv = process.argv.slice(2);
let outDir = null, engineDir = null, version = null;
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--out') outDir = argv[++i];
  else if (argv[i] === '--engine') engineDir = argv[++i];
  else if (argv[i] === '--version') version = argv[++i];
  else fail(`unknown argument: ${argv[i]}`);
}
if (!outDir) fail('usage: node scripts/build_agent_pack.mjs --out <dir> [--engine <dir>] [--version <str>]');
outDir = path.resolve(outDir);
engineDir = engineDir
  ? path.resolve(engineDir)
  : process.env.BURN_WASM_PKG_DIR
    ? path.resolve(process.env.BURN_WASM_PKG_DIR)
    : path.join(ROOT, 'pkg');

// 1. Validate the engine bundle.
const REQUIRED = [
  'burn_research_bg.wasm',
  'burn_research.js',
  'burn_research.d.ts',
  'interactive_multi_input_ingress.mjs',
  path.join('docs', 'agent-layer-catalog.v1.json'),
  path.join('docs', 'agent-layout-contracts.v1.json'),
  path.join('docs', 'wasm-surface.v1.json'),
  'wasm-surface.bindings.actual.json',
];
const missing = REQUIRED.filter((f) => !fs.existsSync(path.join(engineDir, f)));
if (missing.length) {
  fail(`engine bundle ${engineDir} is missing: ${missing.join(', ')} ` +
    `(need a full bundle: build pkg/ with the surface docs, or point --engine / BURN_WASM_PKG_DIR at a CI bundle)`);
}

// Version: repo-<shortsha>-<yyyymmdd> unless given.
if (!version) {
  const git = spawnSync('git', ['-C', ROOT, 'rev-parse', '--short', 'HEAD'], { encoding: 'utf8' });
  const sha = git.status === 0 ? git.stdout.trim() : 'unknown';
  const d = new Date();
  const ymd = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}`;
  version = `repo-${sha}-${ymd}`;
}

// 2. Copy helper/ + engine bundle.
fs.rmSync(outDir, { recursive: true, force: true });
fs.mkdirSync(outDir, { recursive: true });
for (const sub of ['bin', 'lib', 'reference', 'examples']) {
  fs.cpSync(path.join(HELPER, sub), path.join(outDir, sub), { recursive: true });
}
for (const f of ['AGENT.md', 'INDEX.md', 'spec-schema.md']) {
  fs.copyFileSync(path.join(HELPER, f), path.join(outDir, f));
}
fs.cpSync(engineDir, path.join(outDir, 'engine-audit'), { recursive: true });

// 3. contracts-index/*.jsonl from the engine docs.
const catalog = JSON.parse(fs.readFileSync(path.join(engineDir, 'docs', 'agent-layer-catalog.v1.json'), 'utf8'));
const layoutDoc = JSON.parse(fs.readFileSync(path.join(engineDir, 'docs', 'agent-layout-contracts.v1.json'), 'utf8'));
const surfaceDoc = JSON.parse(fs.readFileSync(path.join(engineDir, 'docs', 'wasm-surface.v1.json'), 'utf8'));
const idxDir = path.join(outDir, 'contracts-index');
fs.mkdirSync(idxDir, { recursive: true });
const catalogLines = Object.entries(catalog.constructors ?? {}).map(([name, c]) => JSON.stringify({
  name, family: c.family ?? null, arity: c.arity ?? null,
  signature: c.signature ?? null, layout_contract_key: c.layout_contract_key ?? name,
}));
fs.writeFileSync(path.join(idxDir, 'catalog-index.jsonl'), catalogLines.join('\n') + '\n');
const layoutLines = Object.entries(layoutDoc.constructors ?? {}).map(([name, c]) => JSON.stringify({
  name, input: c.input ?? null, output: c.output ?? null,
}));
fs.writeFileSync(path.join(idxDir, 'layout-index.jsonl'), layoutLines.join('\n') + '\n');
const surfaceLines = [
  ...(surfaceDoc.functions ?? []).map((name) => JSON.stringify({ kind: 'function', name })),
  ...Object.keys(surfaceDoc.classes ?? {}).map((name) => JSON.stringify({ kind: 'class', name })),
];
fs.writeFileSync(path.join(idxDir, 'surface-index.jsonl'), surfaceLines.join('\n') + '\n');

// 4. Counts via the helper tools (same files the pack ships).
const env = { BURN_WASM_PKG_DIR: engineDir };
const count = (tool) => {
  const r = runNode([path.join(HELPER, 'bin', tool), '--list'], env);
  if (r.status !== 0) fail(`${tool} --list failed: ${r.stdout.slice(0, 200)} ${r.stderr.slice(0, 200)}`);
  return countLines(r.stdout);
};
const catalogEntries = count('find-op.mjs');
const surfaceEntries = count('surface.mjs');
const layoutEntries = count('layout.mjs');

// 5. Smoke-run the PACKED tools (not the helper originals).
const smoke = runNode(
  [path.join(outDir, 'bin', 'run.mjs'), path.join(outDir, 'examples', 'linear-relu.json')],
  { BURN_WASM_PKG_DIR: path.join(outDir, 'engine-audit') },
);
let smokeJson = null;
try { smokeJson = JSON.parse(smoke.stdout); } catch { /* reported below */ }
if (smoke.status !== 0 || !smokeJson || JSON.stringify(smokeJson.outputs?.values) !== JSON.stringify([4.5, 10.5])) {
  fail(`packed smoke run failed (exit ${smoke.status}): ${smoke.stdout.slice(0, 300)} ${smoke.stderr.slice(0, 300)}`);
}

// 6. MANIFESTs (same pins in both).
const wasmSha = sha256File(path.join(engineDir, 'burn_research_bg.wasm'));
const jsSha = sha256File(path.join(engineDir, 'burn_research.js'));
const pins = { 'burn_research_bg.wasm': wasmSha, 'burn_research.js': jsSha };
const packManifest = {
  pack_version: version,
  built_from: { repo: 'ai_agent_builder', script: 'scripts/build_agent_pack.mjs' },
  engine: { wasm_sha256: wasmSha, js_sha256: jsSha },
  sha256: pins,
  counts: { catalog_entries: catalogEntries, surface_entries: surfaceEntries, layout_entries: layoutEntries },
};
fs.writeFileSync(path.join(outDir, 'MANIFEST.json'), JSON.stringify(packManifest, null, 2) + '\n');
const helperManifest = {
  sha256: pins,
  engine: { wasm_sha256: wasmSha, js_sha256: jsSha },
  generated_by: 'scripts/build_agent_pack.mjs',
  note: 'Generated — do not hand-edit. Re-run the builder after any engine change.',
};
fs.writeFileSync(path.join(HELPER, 'MANIFEST.json'), JSON.stringify(helperManifest, null, 2) + '\n');

console.log(JSON.stringify({
  ok: true,
  out: outDir,
  pack_version: version,
  catalog_entries: catalogEntries,
  surface_entries: surfaceEntries,
  layout_entries: layoutEntries,
  wasm_sha256: wasmSha,
  js_sha256: jsSha,
}, null, 2));
