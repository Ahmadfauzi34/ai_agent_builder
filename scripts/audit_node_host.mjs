import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapterPath = path.join(pkgDir, 'node.mjs');
const typesPath = path.join(pkgDir, 'node.d.mts');
const supportPath = path.join(pkgDir, 'host-support.v1.json');
const packageJsonPath = path.join(pkgDir, 'package.json');
const surfaceActualPath = path.join(pkgDir, 'wasm-surface.actual.json');
const backgroundTypesPath = path.join(pkgDir, 'burn_research_bg.wasm.d.ts');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(adapterPath), 'packaged node.mjs is missing');
assert(fs.existsSync(typesPath), 'packaged node.d.mts is missing');
assert(fs.existsSync(supportPath), 'packaged host-support.v1.json is missing');
assert(fs.existsSync(packageJsonPath), 'packaged package.json is missing');
assert(fs.existsSync(surfaceActualPath), 'packaged wasm-surface.actual.json is missing');
assert(fs.existsSync(backgroundTypesPath), 'packaged burn_research_bg.wasm.d.ts is missing');

const manifest = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
assert(Array.isArray(manifest.files), 'package.json files allowlist is missing');
const requiredPackageFiles = [
  'burn_research_bg.wasm',
  'burn_research.js',
  'burn_research.d.ts',
  'burn_research_bg.wasm.d.ts',
  'wasm-surface.actual.json',
  'node.mjs',
  'node.d.mts',
  'host-support.v1.json',
];
for (const file of requiredPackageFiles) {
  assert(manifest.files.includes(file), `package.json files allowlist is missing ${file}`);
}

const actualSurface = JSON.parse(fs.readFileSync(surfaceActualPath, 'utf8'));
assert(
  actualSurface.artifact === 'pkg/burn_research.d.ts',
  `surface diagnostic artifact mismatch: ${actualSurface.artifact}`,
);

const support = JSON.parse(fs.readFileSync(supportPath, 'utf8'));
assert(support.schema === 'burn-research.host-support.v1', 'host support schema mismatch');
assert(support.verified_hosts?.node?.status === 'supported', 'Node host must be declared supported');
assert(support.verified_hosts?.node?.adapter === 'node.mjs', 'Node adapter discovery mismatch');
assert(support.verified_hosts?.node?.types === 'node.d.mts', 'Node type discovery mismatch');

const adapter = await import(pathToFileURL(adapterPath).href);
const runtime = await adapter.loadBurnRuntime();

const programCaps = JSON.parse(runtime.programCapabilities());
const bundleCaps = JSON.parse(runtime.programBundleCapabilities());
assert(programCaps.execution_binding === 'required', 'program execution binding capability mismatch');
assert(bundleCaps.schema === 'burn-research.program-bundle.v1', 'program bundle capability mismatch');

const reg = new runtime.LayerRegistry();
const spec = runtime.AgentLayerSpec.relu(1);
reg.initAgentLayer(spec);
const builder = new runtime.AgentGraphBuilder(2);
builder.addUnary(spec, 0, 1);
builder.setOutput(1);
const graph = builder.compile(reg);
const input = new runtime.WasmTensor(
  new Float32Array([-2, 3]),
  new Uint32Array([1, 2, 1, 1]),
);
const output = graph.run(reg, input);
const got = Array.from(output.to_array());
assert(got.length === 2 && got[0] === 0 && got[1] === 3, `Node host execution mismatch: ${got}`);

output.free();
input.free();
graph.free();
builder.free();
spec.free();
reg.free();

console.log(JSON.stringify({
  verdict: 'PASS',
  host: 'node',
  adapter: 'node.mjs',
  types: 'node.d.mts',
  packageDir: adapter.burnRuntimePackageDir(),
  execution: got,
  programIdentitySchema: programCaps.identity_schema,
  programBundleSchema: bundleCaps.schema,
  packageFiles: requiredPackageFiles,
  surfaceDiagnostic: path.basename(surfaceActualPath),
  backgroundTypes: path.basename(backgroundTypesPath),
}, null, 2));
