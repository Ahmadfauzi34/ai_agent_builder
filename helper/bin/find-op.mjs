#!/usr/bin/env node
// find-op.mjs — return ONE layer constructor entry, complete.
// Usage:
//   node helper/bin/find-op.mjs <constructor-name>   -> the full catalog entry as JSON
//   node helper/bin/find-op.mjs --list               -> one line per constructor:
//                                                "<name> <signature> <family>"
// Exit 0 on found, 2 on not found / bad usage.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { printJsonAndExit, exitAfterFlush } from '../lib/runner.mjs';

const ROOT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
import { catalogPath } from '../lib/engine.mjs';
const CATALOG = catalogPath();
const args = process.argv.slice(2);
const [arg] = args;
if (!arg || args.includes('--help') || args.includes('-h')) {
  console.error('usage: node helper/bin/find-op.mjs <constructor-name> | --list');
  process.exit(2);
}
let catalog;
try { catalog = JSON.parse(fs.readFileSync(CATALOG, 'utf8')); }
catch (e) { printJsonAndExit({ ok: false, error: `cannot read catalog: ${e instanceof Error ? e.message : String(e)}` }, 1); }
const ctors = catalog.constructors ?? {};
if (arg === '--list') {
  for (const name of Object.keys(ctors).sort()) {
    const c = ctors[name];
    console.log(`${name} ${c.signature} ${c.family}`);
  }
  exitAfterFlush(0);
}
const entry = Object.hasOwn(ctors, arg) ? ctors[arg] : null;
if (!entry) {
  printJsonAndExit({ ok: false, error: `unknown constructor: ${arg}`, hint: 'run: node helper/bin/find-op.mjs --list' }, 2);
}
console.log(JSON.stringify({ name: arg, ...entry }, null, 2));
