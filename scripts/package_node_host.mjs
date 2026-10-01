import fs from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const docsDir = path.join(pkgDir, 'docs');
const packageJsonPath = path.join(pkgDir, 'package.json');
const nodeAdapterSource = path.resolve('hosts/node/node.mjs');
const nodeTypesSource = path.resolve('hosts/node/node.d.mts');
const readmeSource = path.resolve('README.md');
const interactiveSource = path.resolve('scripts/interactive_multi_input_ingress.mjs');
const provenanceSource = path.resolve('scripts/ingress_provenance.mjs');
const replayLedgerSource = path.resolve('scripts/ingress_replay_ledger.mjs');
const executionReceiptSource = path.resolve('scripts/ingress_execution_receipt.mjs');
const branchPromotionLineageSource = path.resolve('scripts/branch_promotion_lineage.mjs');
const operationRegistrySource = path.resolve('scripts/operation_contract_registry.mjs');
const replayLedgerInitSource = path.resolve('scripts/init_ingress_replay_ledger.mjs');
// Distribution documents live under docs/ so the packaged README.md keeps its
// docs/... relative links working (complaint #13). Every docs/... target the
// README links to is packaged here.
const docSources = {
  'host-support.v1.json': path.resolve('docs/contracts/host-support.v1.json'),
  'wasm-host-communication.md': path.resolve('docs/wasm-host-communication.md'),
  'interactive-multi-input-ingress.md': path.resolve('docs/interactive-multi-input-ingress.md'),
  'ingress-provenance.v1.json': path.resolve('docs/contracts/ingress-provenance.v1.json'),
  'ingress-provenance.v2.json': path.resolve('docs/contracts/ingress-provenance.v2.json'),
  'ingress-replay-ledger.v1.json': path.resolve('docs/contracts/ingress-replay-ledger.v1.json'),
  'host-execution-receipt.v1.json': path.resolve('docs/contracts/host-execution-receipt.v1.json'),
  'host-execution-receipt.v2.json': path.resolve('docs/contracts/host-execution-receipt.v2.json'),
  'host-checkpoint-restore.v1.json': path.resolve('docs/contracts/host-checkpoint-restore.v1.json'),
  'host-state-handoff.v1.json': path.resolve('docs/contracts/host-state-handoff.v1.json'),
  'host-branch-promotion-lineage.v1.json': path.resolve('docs/contracts/host-branch-promotion-lineage.v1.json'),
  'host-operation-contract-registry.v1.json': path.resolve('docs/contracts/host-operation-contract-registry.v1.json'),
  'agent-contracts.v1.json': path.resolve('docs/contracts/agent-contracts.v1.json'),
  'agent-workspace.md': path.resolve('docs/agent-workspace.md'),
  'burn-contract-baseline.md': path.resolve('docs/burn-contract-baseline.md'),
  'es-graph-host-orchestration.md': path.resolve('docs/es-graph-host-orchestration.md'),
  'rust-package-support.md': path.resolve('docs/rust-package-support.md'),
  'python-host-layer.md': path.resolve('docs/python-host-layer.md'),
  'python-ffi-v1.md': path.resolve('docs/python-ffi-v1.md'),
  'python-facade-v1.md': path.resolve('docs/python-facade-v1.md'),
  'python-wheel-support.md': path.resolve('docs/python-wheel-support.md'),
  'agent-layer-catalog.v1.json': path.resolve('docs/contracts/agent-layer-catalog.v1.json'),
  'agent-layout-contracts.v1.json': path.resolve('docs/contracts/agent-layout-contracts.v1.json'),
  'agent-input-port.v1.json': path.resolve('docs/contracts/agent-input-port.v1.json'),
  'agent-fault-contract.v1.json': path.resolve('docs/contracts/agent-fault-contract.v1.json'),
  'multi-input-graph-plan.v1.json': path.resolve('docs/contracts/multi-input-graph-plan.v1.json'),
  'multi-input-plan-explain.v1.json': path.resolve('docs/contracts/multi-input-plan-explain.v1.json'),
  'multi-input-execution-trace.v1.json': path.resolve('docs/contracts/multi-input-execution-trace.v1.json'),
  'baseline-candidate-verification.v1.json': path.resolve('docs/contracts/baseline-candidate-verification.v1.json'),
  'transactional-graph-mutation.v1.json': path.resolve('docs/contracts/transactional-graph-mutation.v1.json'),
  'transactional-graph-checkpoint-branch.v1.json': path.resolve('docs/contracts/transactional-graph-checkpoint-branch.v1.json'),
  'semantic-ingress-manifest.v2.json': path.resolve('docs/contracts/semantic-ingress-manifest.v2.json'),
  'wasm-surface.v1.json': path.resolve('docs/contracts/wasm-surface.v1.json'),
  'runtime-surface.v1.json': path.resolve('docs/contracts/runtime-surface.v1.json'),
  'runtime-architecture-artifact-proof.md': path.resolve('docs/runtime-architecture-artifact-proof.md'),
};
const nodeAdapterTarget = path.join(pkgDir, 'node.mjs');
const nodeTypesTarget = path.join(pkgDir, 'node.d.mts');
const readmeTarget = path.join(pkgDir, 'README.md');
const interactiveTarget = path.join(pkgDir, 'interactive_multi_input_ingress.mjs');
const provenanceTarget = path.join(pkgDir, 'ingress_provenance.mjs');
const replayLedgerTarget = path.join(pkgDir, 'ingress_replay_ledger.mjs');
const executionReceiptTarget = path.join(pkgDir, 'ingress_execution_receipt.mjs');
const branchPromotionLineageTarget = path.join(pkgDir, 'branch_promotion_lineage.mjs');
const operationRegistryTarget = path.join(pkgDir, 'operation_contract_registry.mjs');
const replayLedgerInitTarget = path.join(pkgDir, 'init_ingress_replay_ledger.mjs');

const generatedPackageFiles = [
  'burn_research_bg.wasm.d.ts',
  'wasm-surface.bindings.actual.json',
];

const packagedHostFiles = [
  'node.mjs',
  'node.d.mts',
  'README.md',
  'interactive_multi_input_ingress.mjs',
  'ingress_provenance.mjs',
  'ingress_replay_ledger.mjs',
  'ingress_execution_receipt.mjs',
  'branch_promotion_lineage.mjs',
  'operation_contract_registry.mjs',
  'init_ingress_replay_ledger.mjs',
  ...Object.keys(docSources).map(name => `docs/${name}`),
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

fs.mkdirSync(docsDir, {recursive: true});
fs.copyFileSync(nodeAdapterSource, nodeAdapterTarget);
fs.copyFileSync(nodeTypesSource, nodeTypesTarget);
// Repo sources reference contracts as docs/contracts/<name>.json, but the
// packaged layout keeps them at docs/<name>.json (Opsi D). Rewrite .md copies
// so packaged README/docs links resolve against the packaged layout.
function copyDocWithPackagedLinks(source, target) {
  let text = fs.readFileSync(source, 'utf8');
  text = text.replaceAll('docs/contracts/', 'docs/');
  fs.writeFileSync(target, text);
}
copyDocWithPackagedLinks(readmeSource, readmeTarget);
fs.copyFileSync(interactiveSource, interactiveTarget);
fs.copyFileSync(provenanceSource, provenanceTarget);
fs.copyFileSync(replayLedgerSource, replayLedgerTarget);
fs.copyFileSync(executionReceiptSource, executionReceiptTarget);
fs.copyFileSync(branchPromotionLineageSource, branchPromotionLineageTarget);
fs.copyFileSync(operationRegistrySource, operationRegistryTarget);
fs.copyFileSync(replayLedgerInitSource, replayLedgerInitTarget);
for (const [name, source] of Object.entries(docSources)) {
  const target = path.join(docsDir, name);
  if (name.endsWith('.md')) {
    copyDocWithPackagedLinks(source, target);
  } else {
    fs.copyFileSync(source, target);
  }
}

const manifest = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
const files = Array.isArray(manifest.files) ? [...manifest.files] : [];
for (const file of requiredManifestFiles) {
  if (!files.includes(file)) files.push(file);
}
manifest.files = files;
fs.writeFileSync(packageJsonPath, `${JSON.stringify(manifest, null, 2)}\n`);

const registryAudit = spawnSync(
  process.execPath,
  [path.resolve('scripts/audit_operation_contract_registry.mjs'), pkgDir],
  {encoding: 'utf8'},
);
if (registryAudit.status !== 0) {
  throw new Error(`operation registry package audit failed: ${registryAudit.stderr || registryAudit.stdout}`);
}

console.log(JSON.stringify({
  packaged: true,
  pkgDir,
  files: requiredManifestFiles,
  operation_registry_audit: JSON.parse(registryAudit.stdout.trim().split('\n').at(-1)),
}, null, 2));
