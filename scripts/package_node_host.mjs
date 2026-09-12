import fs from 'node:fs';
import path from 'node:path';

const pkgDir = path.resolve(process.argv[2] ?? 'pkg');
const packageJsonPath = path.join(pkgDir, 'package.json');
const nodeAdapterSource = path.resolve('hosts/node/node.mjs');
const hostSupportSource = path.resolve('docs/host-support.v1.json');
const nodeAdapterTarget = path.join(pkgDir, 'node.mjs');
const hostSupportTarget = path.join(pkgDir, 'host-support.v1.json');

if (!fs.existsSync(packageJsonPath)) {
  throw new Error(`package.json not found in ${pkgDir}`);
}

fs.copyFileSync(nodeAdapterSource, nodeAdapterTarget);
fs.copyFileSync(hostSupportSource, hostSupportTarget);

const manifest = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
const files = Array.isArray(manifest.files) ? [...manifest.files] : [];
for (const file of ['node.mjs', 'host-support.v1.json']) {
  if (!files.includes(file)) files.push(file);
}
manifest.files = files;
fs.writeFileSync(packageJsonPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(JSON.stringify({
  packaged: true,
  pkgDir,
  files: ['node.mjs', 'host-support.v1.json'],
}, null, 2));
