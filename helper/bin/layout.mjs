#!/usr/bin/env node
// layout.mjs — return ONE constructor's layout contract, complete.
// Usage:
//   node helper/bin/layout.mjs <constructor-name>  -> {name, input, output, ...}
//   node helper/bin/layout.mjs --list              -> one line per constructor:
//                                              "<name> <input> -> <output>"
// Exit 0 on found, 2 on not found / bad usage.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { printJsonAndExit, exitAfterFlush } from '../lib/runner.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
import { layoutContractsPath } from '../lib/engine.mjs';
const FILE = layoutContractsPath();
const args = process.argv.slice(2);
const [arg] = args;
if (!arg || args.includes('--help') || args.includes('-h')) {
  console.error('usage: node helper/bin/layout.mjs <constructor-name> | --list');
  process.exit(2);
}
let doc;
try { doc = JSON.parse(fs.readFileSync(FILE, 'utf8')); }
catch (e) { printJsonAndExit({ ok: false, error: `cannot read layout contracts: ${e instanceof Error ? e.message : String(e)}` }, 1); }
const ctors = doc.constructors ?? {};
const layouts = doc.layouts ?? {};
const compat = doc.compatibility ?? {};
if (arg === '--list') {
  for (const name of Object.keys(ctors).sort()) {
    const c = ctors[name];
    console.log(`${name} ${c.input} -> ${c.output}`);
  }
  exitAfterFlush(0);
}
const entry = Object.hasOwn(ctors, arg) ? ctors[arg] : null;
if (!entry) {
  printJsonAndExit({ ok: false, error: `unknown constructor: ${arg}`, hint: 'run: node helper/bin/layout.mjs --list' }, 2);
}
const describe = (key) => {
  const l = Object.hasOwn(layouts, key) ? layouts[key] : null;
  const c = Object.hasOwn(compat, key) ? compat[key] : null;
  if (!l) return { layout: key };
  return {
    layout: key,
    ...l,
    ...(c ? { compatibility: c } : {}),
  };
};
const { input: inKey, output: outKey, ...extra } = entry;
console.log(JSON.stringify({
  name: arg,
  input: describe(inKey),
  output: describe(outKey),
  ...extra,
  policy: doc.policy ?? null,
}, null, 2));
