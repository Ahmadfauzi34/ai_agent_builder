import fs from 'node:fs';
import path from 'node:path';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const packageJsonPath = path.join(pkgDir, 'package.json');
const nodeAdapterSource = path.resolve('hosts/node/node.mjs');
const nodeTypesSource = path.resolve('hosts/node/node.d.mts');
const hostSupportSource = path.resolve('docs/host-support.v1.json');
const communicationSource = path.resolve('docs/wasm-host-communication.md');
const interactiveSource = path.resolve('scripts/interactive_multi_input_ingress.mjs');
const provenanceSource = path.resolve('scripts/ingress_provenance.mjs');
const replayLedgerSource = path.resolve('scripts/ingress_replay_ledger.mjs');
const executionReceiptSource = path.resolve('scripts/ingress_execution_receipt.mjs');
const replayLedgerInitSource = path.resolve('scripts/init_ingress_replay_ledger.mjs');
const provenanceContractSource = path.resolve('docs/ingress-provenance.v1.json');
const replayLedgerContractSource = path.resolve('docs/ingress-replay-ledger.v1.json');
const executionReceiptContractSource = path.resolve('docs/host-execution-receipt.v2.json');
const checkpointRestoreContractSource = path.resolve('docs/host-checkpoint-restore.v1.json');
const legacyExecutionReceiptContractSource = path.resolve('docs/host-execution-receipt.v1.json');
const stateHandoffContractSource = path.resolve('docs/host-state-handoff.v1.json');
const wasmSurfaceContractSource = path.resolve('docs/wasm-surface.v1.json');
const runtimeSurfaceContractSource = path.resolve('docs/runtime-surface.v1.json');
const nodeAdapterTarget = path.join(pkgDir, 'node.mjs');
const nodeTypesTarget = path.join(pkgDir, 'node.d.mts');
const hostSupportTarget = path.join(pkgDir, 'host-support.v1.json');
const communicationTarget = path.join(pkgDir, 'wasm-host-communication.md');
const interactiveTarget = path.join(pkgDir, 'interactive_multi_input_ingress.mjs');
const provenanceTarget = path.join(pkgDir, 'ingress_provenance.mjs');
const replayLedgerTarget = path.join(pkgDir, 'ingress_replay_ledger.mjs');
const executionReceiptTarget = path.join(pkgDir, 'ingress_execution_receipt.mjs');
const replayLedgerInitTarget = path.join(pkgDir, 'init_ingress_replay_ledger.mjs');
const provenanceContractTarget = path.join(pkgDir, 'ingress-provenance.v1.json');
const replayLedgerContractTarget = path.join(pkgDir, 'ingress-replay-ledger.v1.json');
const executionReceiptContractTarget = path.join(pkgDir, 'host-execution-receipt.v2.json');
const checkpointRestoreContractTarget = path.join(pkgDir, 'host-checkpoint-restore.v1.json');
const legacyExecutionReceiptContractTarget = path.join(pkgDir, 'host-execution-receipt.v1.json');
const stateHandoffContractTarget = path.join(pkgDir, 'host-state-handoff.v1.json');
const wasmSurfaceContractTarget = path.join(pkgDir, 'wasm-surface.v1.json');
const runtimeSurfaceContractTarget = path.join(pkgDir, 'runtime-surface.v1.json');

const generatedPackageFiles = [
  'burn_research_bg.wasm.d.ts',
  'wasm-surface.bindings.actual.json',
];

const packagedHostFiles = [
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
  'host-execution-receipt.v2.json',
  'host-checkpoint-restore.v1.json',
  'host-state-handoff.v1.json',
  'wasm-surface.v1.json',
  'runtime-surface.v1.json',
];

const requiredManifestFiles = [
  ...generatedPackageFiles,
  ...packagedHostFiles,
  'wasm-surface.actual.json',
];

if (!fs.existsSync(packageJsonPath)) {
  throw new Error(`package.json not found in ${pkgDir}`);
}

for (const file of generatedPackageFiles) {
  const generatedPath = path.join(pkgDir, file);
  if (!fs.existsSync(generatedPath)) {
    throw new Error(`required generated package file is missing: ${generatedPath}`);
  }
}

fs.copyFileSync(nodeAdapterSource, nodeAdapterTarget);
fs.copyFileSync(nodeTypesSource, nodeTypesTarget);
fs.copyFileSync(hostSupportSource, hostSupportTarget);
fs.copyFileSync(communicationSource, communicationTarget);
fs.copyFileSync(interactiveSource, interactiveTarget);
fs.copyFileSync(provenanceSource, provenanceTarget);
fs.copyFileSync(replayLedgerSource, replayLedgerTarget);
fs.copyFileSync(executionReceiptSource, executionReceiptTarget);
fs.copyFileSync(replayLedgerInitSource, replayLedgerInitTarget);
fs.copyFileSync(provenanceContractSource, provenanceContractTarget);
fs.copyFileSync(replayLedgerContractSource, replayLedgerContractTarget);
fs.copyFileSync(executionReceiptContractSource, executionReceiptContractTarget);
fs.copyFileSync(checkpointRestoreContractSource, checkpointRestoreContractTarget);
fs.copyFileSync(legacyExecutionReceiptContractSource, legacyExecutionReceiptContractTarget);
fs.copyFileSync(stateHandoffContractSource, stateHandoffContractTarget);
fs.copyFileSync(wasmSurfaceContractSource, wasmSurfaceContractTarget);
fs.copyFileSync(runtimeSurfaceContractSource, runtimeSurfaceContractTarget);

const manifest = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
const files = Array.isArray(manifest.files) ? [...manifest.files] : [];
for (const file of requiredManifestFiles) {
  if (!files.includes(file)) files.push(file);
}
manifest.files = files;
fs.writeFileSync(packageJsonPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(JSON.stringify({
  packaged: true,
  pkgDir,
  files: requiredManifestFiles,
}, null, 2));
