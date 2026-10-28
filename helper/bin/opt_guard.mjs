#!/usr/bin/env node
// wasm-opt opt_guard: probe the engine's tiered-limit guard (complaint #14).
//
// Calls the engine's own guard decision function on arbitrary bytes: would a
// state-record payload of this size be accepted, and under which bincode byte
// limit tier? Only the input length is inspected — probing is always safe.
//
// Usage:
//   node opt_guard.mjs <file>        # probe a file's bytes
//   node opt_guard.mjs --bytes N     # probe a synthetic N-byte payload
// Prints the verdict JSON; exit 0 when accepted, 2 when rejected/error.
import fs from 'node:fs';
import { loadEngineWithOracle, guardCheck } from '../lib/opt.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';

const argv = process.argv.slice(2);
if (argv.length === 0) {
  console.error('usage: node opt_guard.mjs <file> | --bytes N');
  process.exit(2);
}

let bytes;
let label;
if (argv[0] === '--bytes') {
  const n = Number(argv[1]);
  if (!Number.isInteger(n) || n < 0) {
    console.error('--bytes needs a non-negative integer');
    process.exit(2);
  }
  bytes = new Uint8Array(n);
  label = `${n} synthetic bytes`;
} else {
  try {
    bytes = fs.readFileSync(argv[0]);
  } catch (e) {
    console.error(`cannot read ${argv[0]}: ${e.message}`);
    process.exit(2);
  }
  label = argv[0];
}

try {
  const { rt } = await loadEngineWithOracle();
  const verdict = guardCheck(rt, bytes);
  printJsonAndExit({ ok: true, input: label, ...verdict }, verdict.accepted ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
