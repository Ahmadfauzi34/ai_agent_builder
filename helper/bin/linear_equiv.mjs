#!/usr/bin/env node
// burn-math #11: exact equivalence for linear subgraphs.
//
// Usage: node linear_equiv.mjs <specA.json> <specB.json>
// Prints {ok, verdict: 'ACCEPT'|'REJECT', reason, ...}.
// ACCEPT = the two specs denote a bit-identical affine map (no tolerance).
// Exit 0 iff ACCEPT.
//
// This is a proof about the specs' symbolic form: with identical (W, b),
// any faithful executor must agree on every input. It says nothing about
// engine numerics beyond that.
import fs from 'node:fs';
import { loadSpec } from '../lib/spec.mjs';
import { equivLinear, effectiveAffine } from '../lib/linear_equiv.mjs';
import { printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';

const [aFile, bFile] = process.argv.slice(2);
if (!aFile || !bFile || process.argv.slice(2).includes('--help') || process.argv.slice(2).includes('-h')) {
  console.error('usage: node linear_equiv.mjs <specA.json> <specB.json>');
  process.exit(2);
}
let specA, specB;
try { specA = loadSpec(aFile); } catch (e) { failCli(`specA: ${e.message}`); }
try { specB = loadSpec(bFile); } catch (e) { failCli(`specB: ${e.message}`); }

const r = equivLinear(specA, specB);
const detail = {};
if (r.verdict === 'ACCEPT') {
  // Include the effective map fingerprints (not the full matrices) for the record.
  const A = effectiveAffine(specA);
  const flat = [...A.W.flat(), ...A.b];
  let h = 0;
  for (const x of flat) { const s = String(x); for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0; }
  detail.effective_fingerprint = `fnv1a32:${h.toString(16).padStart(8, '0')}`;
  detail.dims = r.dims;
  detail.chainA = r.chainA;
  detail.chainB = r.chainB;
} else if (r.firstDiff) {
  detail.firstDiff = r.firstDiff;
}
printJsonAndExit({ ok: r.verdict === 'ACCEPT', verdict: r.verdict, reason: r.reason, ...detail }, r.verdict === 'ACCEPT' ? 0 : 2);
