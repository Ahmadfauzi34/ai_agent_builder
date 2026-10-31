#!/usr/bin/env node
// burn-math fit_sensitivity: ranked sensitivity REPORT on top of gradcheck.
//
// gradcheck prints a raw Jacobian ("how much does each output move when
// each input moves"). This tool turns it into a report an agent can act on:
// per-output top-K, a global top-K, threshold flags, and a summary. It does
// NOT recompute the Jacobian: it runs the sibling gradcheck.mjs (same
// directory, same Node via process.execPath) and ranks its output. Reuse,
// not a second implementation — the Jacobian has exactly one producer.
//
// Usage: node fit_sensitivity.mjs <spec.json> [topK=10] [threshold=0]
//
//   spec.json : graph spec with continuous-valued inputs.
//   topK      : entries kept per output and globally (default 10).
//   threshold : |J| at or above this is flagged (default 0 = everything).
//
// Exit 0 when the report was produced; exit 2 when gradcheck failed or the
// input is bad. The report itself never "fails": a low-sensitivity spec is
// a finding, not an error.
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { printJsonAndExit } from '../lib/runner.mjs';
import { rankJacobian } from '../lib/fit.mjs';

const [specFile, topKArg, thresholdArg] = process.argv.slice(2);
if (!specFile || process.argv.slice(2).includes('--help') || process.argv.slice(2).includes('-h')) {
  console.error('usage: node fit_sensitivity.mjs <spec.json> [topK=10] [threshold=0]');
  process.exit(2);
}
const topK = topKArg !== undefined ? Number(topKArg) : 10;
const threshold = thresholdArg !== undefined ? Number(thresholdArg) : 0;
if (!(Number.isInteger(topK) && topK > 0) || !(threshold >= 0)) {
  printJsonAndExit({ ok: false, error: 'topK must be a positive integer; threshold must be non-negative' }, 2);
}

try {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const gradcheckPath = path.join(here, 'gradcheck.mjs');
  let raw;
  try {
    raw = execFileSync(process.execPath, [gradcheckPath, specFile], { encoding: 'utf8' });
  } catch (e) {
    const stderr = e.stderr ? String(e.stderr) : '';
    const stdout = e.stdout ? String(e.stdout) : '';
    throw new Error(`gradcheck failed: ${(stdout || stderr).trim() || e.message}`);
  }
  const gc = JSON.parse(raw);
  if (!gc.ok) throw new Error(`gradcheck failed: ${gc.error}`);

  const perInputSlot = {};
  for (const [slotStr, J] of Object.entries(gc.jacobians)) {
    perInputSlot[slotStr] = rankJacobian(J, topK, threshold);
  }
  // Global top-K across slots (same topK budget, entries tagged by slot).
  const all = [];
  for (const [slotStr, r] of Object.entries(perInputSlot)) {
    for (const e of r.entries) all.push({ slot: slotStr, ...e });
  }
  all.sort((a, b) => b.abs - a.abs);

  printJsonAndExit({
    ok: true,
    top_k: topK,
    threshold,
    eps: gc.eps,
    per_input_slot: perInputSlot,
    global_top: all.slice(0, topK),
    notes: [
      'Jacobian produced by gradcheck (central differences); this tool only ranks it',
      'rank by |dOut/dIn|: a large entry means that input moves that output a lot',
      'non-smooth ops (relu at the kink) yield a subgradient average — see gradcheck notes',
    ],
  }, 0);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
