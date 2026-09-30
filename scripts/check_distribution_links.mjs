// Local-link checker for the distributed package (complaint #13, R-15).
// Verifies every local markdown link in the packaged README.md resolves to an
// existing file or directory inside the package. External URLs and pure
// anchors are ignored. Fails closed: CI must fail when a link target is gone.
//
// Usage: node scripts/check_distribution_links.mjs [packageDir]
// (point it at the repo root to check the repo README instead.)
import fs from 'node:fs';
import path from 'node:path';

const root = path.resolve(process.argv[2] ?? 'pkg');
const readmePath = path.join(root, 'README.md');
if (!fs.existsSync(readmePath)) {
  console.error(JSON.stringify({verdict: 'FAIL', reason: 'README.md missing', root}));
  process.exit(1);
}
const readme = fs.readFileSync(readmePath, 'utf8');
const linkPattern = /\[[^\]]*\]\(([^)\s]+)\)/g;
const checked = [];
const missing = [];
let match;
for (;;) {
  match = linkPattern.exec(readme);
  if (!match) break;
  let target = match[1].trim();
  if (/^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(target) || target.startsWith('#')) continue; // external URL or pure anchor
  target = target.split('#')[0];
  if (!target) continue;
  const resolved = path.resolve(root, target);
  if (!resolved.startsWith(root + path.sep)) {
    missing.push({target, reason: 'escapes package root'});
    continue;
  }
  const stat = fs.existsSync(resolved) ? fs.statSync(resolved) : null;
  if (!stat) {
    missing.push({target, reason: 'target missing'});
    continue;
  }
  if (target.endsWith('/') && !stat.isDirectory()) {
    missing.push({target, reason: 'expected a directory'});
    continue;
  }
  checked.push(target);
}

const result = {
  verdict: missing.length === 0 ? 'PASS' : 'FAIL',
  root,
  links_checked: checked.length,
  missing,
};
console.log(JSON.stringify(result, null, 2));
process.exit(missing.length === 0 ? 0 : 1);
