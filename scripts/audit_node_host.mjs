import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {verifyWasmSurfaceActual} from './generate_wasm_surface_actual.mjs';

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
const stateBoundProvenanceContractPath = path.join(pkgDir, 'ingress-provenance.v2.json');
const replayLedgerContractPath = path.join(pkgDir, 'ingress-replay-ledger.v1.json');
const executionReceiptContractPath = path.join(pkgDir, 'host-execution-receipt.v2.json');
const checkpointRestoreContractPath = path.join(pkgDir, 'host-checkpoint-restore.v1.json');
const legacyExecutionReceiptContractPath = path.join(pkgDir, 'host-execution-receipt.v1.json');
const stateHandoffContractPath = path.join(pkgDir, 'host-state-handoff.v1.json');
const wasmSurfaceContractPath = path.join(pkgDir, 'wasm-surface.v1.json');
const runtimeSurfaceContractPath = path.join(pkgDir, 'runtime-surface.v1.json');
const packageJsonPath = path.join(pkgDir, 'package.json');
const surfaceActualPath = path.join(pkgDir, 'wasm-surface.actual.json');
const bindingSurfaceActualPath = path.join(pkgDir, 'wasm-surface.bindings.actual.json');
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
assert(fs.existsSync(stateBoundProvenanceContractPath), 'packaged ingress-provenance.v2.json is missing');
assert(fs.existsSync(replayLedgerContractPath), 'packaged ingress-replay-ledger.v1.json is missing');
assert(fs.existsSync(executionReceiptContractPath), 'packaged host-execution-receipt.v2.json is missing');
assert(fs.existsSync(checkpointRestoreContractPath), 'packaged host-checkpoint-restore.v1.json is missing');
assert(fs.existsSync(legacyExecutionReceiptContractPath), 'packaged legacy host-execution-receipt.v1.json is missing');
assert(fs.existsSync(stateHandoffContractPath), 'packaged host-state-handoff.v1.json is missing');
assert(fs.existsSync(wasmSurfaceContractPath), 'packaged wasm-surface.v1.json is missing');
assert(fs.existsSync(path.join(pkgDir, 'multi-input-execution-trace.v1.json')), 'packaged trace contract is missing');
assert(fs.existsSync(runtimeSurfaceContractPath), 'packaged runtime-surface.v1.json is missing');
assert(fs.existsSync(packageJsonPath), 'packaged package.json is missing');
assert(fs.existsSync(surfaceActualPath), 'packaged wasm-surface.actual.json is missing');
assert(fs.existsSync(bindingSurfaceActualPath), 'packaged wasm-surface.bindings.actual.json is missing');
assert(fs.existsSync(backgroundTypesPath), 'packaged burn_research_bg.wasm.d.ts is missing');

const manifest = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
assert(Array.isArray(manifest.files), 'package.json files allowlist is missing');
const requiredPackageFiles = [
  'burn_research_bg.wasm',
  'burn_research.js',
  'burn_research.d.ts',
  'burn_research_bg.wasm.d.ts',
  'wasm-surface.actual.json',
  'wasm-surface.bindings.actual.json',
  'wasm-surface.v1.json',
  'multi-input-execution-trace.v1.json',
  'runtime-surface.v1.json',
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
  'ingress-provenance.v2.json',
  'ingress-replay-ledger.v1.json',
  'host-execution-receipt.v1.json',
  'host-execution-receipt.v2.json',
  'host-checkpoint-restore.v1.json',
  'host-state-handoff.v1.json',
];
for (const file of requiredPackageFiles) {
  assert(manifest.files.includes(file), `package.json files allowlist is missing ${file}`);
}

const actualSurface = JSON.parse(fs.readFileSync(surfaceActualPath, 'utf8'));
assert(actualSurface.schema === 'burn-research.wasm-surface.actual.v1', 'runtime surface schema mismatch');
assert(actualSurface.fingerprint?.algorithm === 'sha256', 'runtime surface fingerprint algorithm mismatch');
assert(/^sha256:[0-9a-f]{64}$/.test(actualSurface.fingerprint?.value ?? ''), 'runtime surface fingerprint is malformed');

const support = JSON.parse(fs.readFileSync(supportPath, 'utf8'));
assert(support.schema === 'burn-research.host-support.v1', 'host support schema mismatch');
assert(support.verified_hosts?.node?.status === 'supported', 'Node host must be declared supported');
assert(support.verified_hosts?.node?.adapter === 'node.mjs', 'Node adapter discovery mismatch');
assert(support.verified_hosts?.node?.interactive_runner === 'interactive_multi_input_ingress.mjs', 'Node interactive runner discovery mismatch');
assert(support.verified_hosts?.node?.ingress_provenance_contract === 'ingress-provenance.v1.json', 'Node signed ingress contract discovery mismatch');
assert(support.verified_hosts?.node?.ingress_replay_ledger_contract === 'ingress-replay-ledger.v1.json', 'Node replay ledger contract discovery mismatch');
assert(support.verified_hosts?.node?.ingress_replay_ledger_initializer === 'init_ingress_replay_ledger.mjs', 'Node ledger initializer discovery mismatch');
assert(support.verified_hosts?.node?.host_execution_receipt_contract === 'host-execution-receipt.v2.json', 'Node execution receipt contract discovery mismatch');
assert(support.verified_hosts?.node?.host_checkpoint_restore_contract === 'host-checkpoint-restore.v1.json', 'Node checkpoint restore contract discovery mismatch');
assert(support.verified_hosts?.node?.legacy_host_execution_receipt_contract === 'host-execution-receipt.v1.json', 'Node legacy receipt contract discovery mismatch');
assert(support.verified_hosts?.node?.host_state_handoff_contract === 'host-state-handoff.v1.json', 'Node state handoff contract discovery mismatch');
assert(support.verified_hosts?.node?.wasm_surface_contract === 'wasm-surface.v1.json', 'Node WASM surface contract discovery mismatch');
assert(support.verified_hosts?.node?.runtime_surface_contract === 'runtime-surface.v1.json', 'Node runtime surface contract discovery mismatch');
assert(support.verified_hosts?.node?.wasm_binding_projection === 'wasm-surface.bindings.actual.json', 'Node binding projection discovery mismatch');
assert(support.verified_hosts?.node?.runtime_surface_actual === 'wasm-surface.actual.json', 'Node runtime surface discovery mismatch');
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
const verifiedSurface = await verifyWasmSurfaceActual(pkgDir, runtime);
assert(verifiedSurface.capability_groups?.multi_input_semantic_ingress?.capabilities?.semantic_ingress_v2?.schema
  === 'burn-research.semantic-ingress-manifest.v2', 'runtime surface omits multi-input semantic ingress v2');
assert(verifiedSurface.wasm_binary_surface?.imports?.length > 0
  && verifiedSurface.wasm_binary_surface?.exports?.length > 0, 'runtime surface omits raw WebAssembly module inventory');
assert(verifiedSurface.capability_groups?.input_contract_and_provenance?.capabilities?.proof_provenance?.schema
  === 'burn-research.proof-provenance.v1', 'runtime surface omits input provenance capability');
assert(verifiedSurface.capability_groups?.program_bundle_checkpoint?.capabilities?.multi_input_program_bundle?.schema
  === 'burn-research.multi-input-program-bundle.v1', 'runtime surface omits multi-input checkpoint capability');
assert(verifiedSurface.math_version_channels?.math_program_surface_version?.schema === 'burn-research.math-program.v9'
  && verifiedSurface.math_version_channels?.math_interaction_protocol_version?.schema === 'burn-research.math-interaction.v1',
'MathProgram semantic generation and interaction protocol version channels are conflated or missing');
assert(verifiedSurface.host_capabilities?.contracts?.state_handoff?.schema === 'burn-research.host-state-handoff.v1',
  'runtime surface omits Node host state-handoff contract');
assert(verifiedSurface.host_capabilities?.contracts?.checkpoint_restore?.schema === 'burn-research.host-checkpoint-restore.v1',
  'runtime surface omits Node host checkpoint restore contract');
assert(verifiedSurface.host_capabilities?.contracts?.state_bound_signed_ingress_provenance?.schema === 'burn-research.ingress-provenance.v2',
  'runtime surface omits the state-bound signed ingress contract');

const programCaps = JSON.parse(runtime.programCapabilities());
const bundleCaps = JSON.parse(runtime.programBundleCapabilities());
const multiInputBundleCaps = JSON.parse(runtime.multiInputProgramBundleCapabilities());
assert(programCaps.execution_binding === 'required', 'program execution binding capability mismatch');
assert(bundleCaps.schema === 'burn-research.program-bundle.v1', 'program bundle capability mismatch');
assert(multiInputBundleCaps.schema === 'burn-research.multi-input-program-bundle.v1'
  && multiInputBundleCaps.state_integrity === 'no_signature_or_authentication'
  && multiInputBundleCaps.authorization === false, 'multi-input state bundle authority boundary mismatch');

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
  surfaceFingerprint: verifiedSurface.fingerprint.value,
  backgroundTypes: path.basename(backgroundTypesPath),
  communicationContract: path.basename(communicationPath),
}, null, 2));
