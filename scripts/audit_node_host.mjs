import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const adapterPath = path.join(pkgDir, 'node.mjs');
const typesPath = path.join(pkgDir, 'node.d.mts');
const supportPath = path.join(pkgDir, 'host-support.v1.json');
const communicationPath = path.join(pkgDir, 'wasm-host-communication.md');
const interactivePath = path.join(pkgDir, 'interactive_multi_input_ingress.mjs');
const provenancePath = path.join(pkgDir, 'ingress_provenance.mjs');
const replayLedgerPath = path.join(pkgDir, 'ingress_replay_ledger.mjs');
const executionReceiptPath = path.join(pkgDir, 'ingress_execution_receipt.mjs');
const replayLedgerInitPath = path.join(pkgDir, 'init_ingress_replay_ledger.mjs');
const provenanceContractPath = path.join(pkgDir, 'ingress-provenance.v1.json');
const replayLedgerContractPath = path.join(pkgDir, 'ingress-replay-ledger.v1.json');
const executionReceiptContractPath = path.join(pkgDir, 'host-execution-receipt.v1.json');
const packageJsonPath = path.join(pkgDir, 'package.json');
const surfaceActualPath = path.join(pkgDir, 'wasm-surface.actual.json');
const backgroundTypesPath = path.join(pkgDir, 'burn_research_bg.wasm.d.ts');

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

assert(fs.existsSync(adapterPath), 'packaged node.mjs is missing');
assert(fs.existsSync(typesPath), 'packaged node.d.mts is missing');
assert(fs.existsSync(supportPath), 'packaged host-support.v1.json is missing');
assert(fs.existsSync(communicationPath), 'packaged wasm-host-communication.md is missing');
assert(fs.existsSync(interactivePath), 'packaged interactive_multi_input_ingress.mjs is missing');
assert(fs.existsSync(provenancePath), 'packaged ingress_provenance.mjs is missing');
assert(fs.existsSync(replayLedgerPath), 'packaged ingress_replay_ledger.mjs is missing');
assert(fs.existsSync(executionReceiptPath), 'packaged ingress_execution_receipt.mjs is missing');
assert(fs.existsSync(replayLedgerInitPath), 'packaged init_ingress_replay_ledger.mjs is missing');
assert(fs.existsSync(provenanceContractPath), 'packaged ingress-provenance.v1.json is missing');
assert(fs.existsSync(replayLedgerContractPath), 'packaged ingress-replay-ledger.v1.json is missing');
assert(fs.existsSync(executionReceiptContractPath), 'packaged host-execution-receipt.v1.json is missing');
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
  'wasm-host-communication.md',
  'interactive_multi_input_ingress.mjs',
  'ingress_provenance.mjs',
  'ingress_replay_ledger.mjs',
  'ingress_execution_receipt.mjs',
  'init_ingress_replay_ledger.mjs',
  'ingress-provenance.v1.json',
  'ingress-replay-ledger.v1.json',
  'host-execution-receipt.v1.json',
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
assert(support.verified_hosts?.node?.interactive_runner === 'interactive_multi_input_ingress.mjs', 'Node interactive runner discovery mismatch');
assert(support.verified_hosts?.node?.ingress_provenance_contract === 'ingress-provenance.v1.json', 'Node signed ingress contract discovery mismatch');
assert(support.verified_hosts?.node?.ingress_replay_ledger_contract === 'ingress-replay-ledger.v1.json', 'Node replay ledger contract discovery mismatch');
assert(support.verified_hosts?.node?.ingress_replay_ledger_initializer === 'init_ingress_replay_ledger.mjs', 'Node ledger initializer discovery mismatch');
assert(support.verified_hosts?.node?.host_execution_receipt_contract === 'host-execution-receipt.v1.json', 'Node execution receipt contract discovery mismatch');
assert(support.verified_hosts?.node?.types === 'node.d.mts', 'Node type discovery mismatch');
assert(
  support.support_semantics?.packaged_communication_contract === 'wasm-host-communication.md',
  'packaged communication-contract discovery mismatch',
);
const communication = fs.readFileSync(communicationPath, 'utf8');
assert(
  communication.includes('Node application') &&
    communication.includes('initSync({ module: wasmBytes })'),
  'packaged communication contract is missing the verified Node -> WASM path',
);
assert(
  communication.includes('Python does not communicate through the WASM surface.'),
  'packaged communication contract is missing the Python/WASM boundary',
);

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
  communicationContract: path.basename(communicationPath),
}, null, 2));
