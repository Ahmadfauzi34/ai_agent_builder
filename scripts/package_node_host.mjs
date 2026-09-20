import fs from 'node:fs';
import path from 'node:path';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const packageJsonPath = path.join(pkgDir, 'package.json');
const nodeAdapterSource = path.resolve('hosts/node/node.mjs');
const nodeTypesSource = path.resolve('hosts/node/node.d.mts');
const hostSupportSource = path.resolve('docs/host-support.v1.json');
const communicationSource = path.resolve('docs/wasm-host-communication.md');
const nodeAdapterTarget = path.join(pkgDir, 'node.mjs');
const nodeTypesTarget = path.join(pkgDir, 'node.d.mts');
const hostSupportTarget = path.join(pkgDir, 'host-support.v1.json');
const communicationTarget = path.join(pkgDir, 'wasm-host-communication.md');

const generatedPackageFiles = [
  'burn_research_bg.wasm.d.ts',
  'wasm-surface.actual.json',
];

const packagedHostFiles = [
  'node.mjs',
  'node.d.mts',
  'host-support.v1.json',
  'wasm-host-communication.md',
];

const requiredManifestFiles = [
  ...generatedPackageFiles,
  ...packagedHostFiles,
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
