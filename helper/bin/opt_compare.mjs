#!/usr/bin/env node
// wasm-opt opt_compare: engine-provided optimization design loop.
//
// An agent designing a low-level optimization proposes a candidate spec;
// the engine itself judges it:
//   0. static shape/dataflow gate on both specs (ported from the burn-math
//      skill during the 2026-10-20 two-way reconciliation),
//   1. numerical equivalence vs the baseline (MultiInputVerificationCases),
//   2. profiled time/memory delta for both (Fase A memory oracle).
// Verdict is numbers, not feelings.
//
// Usage: node opt_compare.mjs <baseline.json> <candidate.json> [cases.json] [iterations]
// Prints a JSON envelope on stdout; exit 0 when equivalent, 2 otherwise.
import fs from 'node:fs';
import { loadEngineWithOracle, profileSpec } from '../lib/opt.mjs';
import { buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { checkSpec } from '../lib/shapecore.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';

const [baseFile, candFile, casesFile, iterArg] = process.argv.slice(2);
if (!baseFile || !candFile) {
  console.error('usage: node opt_compare.mjs <baseline.json> <candidate.json> [cases.json] [iterations]');
  process.exit(2);
}
const iterations = iterArg !== undefined ? Number(iterArg) : 11;

try {
  const baseSpec = loadSpec(baseFile);
  const candSpec = loadSpec(candFile);
  const cases = casesFile
    ? JSON.parse(fs.readFileSync(casesFile, 'utf8'))
    : [baseSpec.inputs];
  if (!Array.isArray(cases) || cases.length === 0) throw new Error('cases must be a non-empty array');

  // 0. Static gate: shape + dataflow check BEFORE the engine spins up.
  //    Catches dim mismatch in milliseconds with a precise step index.
  for (const [label, s] of [['baseline', baseSpec], ['candidate', candSpec]]) {
    const chk = checkSpec(s);
    if (!chk.ok) {
      printJsonAndExit({
        ok: false,
        error: `${label} spec failed static shape/dataflow check`,
        error_envelope: {
          code: 'shape_mismatch',
          phase: 'staticGate',
          mismatches: chk.mismatches,
          dataflow_issues: chk.dataflow.issues,
          remediation_hint: 'fix the listed steps, then re-run; shapecheck.mjs shows the same report standalone',
        },
    }, 2);
    }
  }

  const { rt } = await loadEngineWithOracle();

  // 1. Numerical equivalence (engine-provided).
  const base = buildGraph(rt, baseSpec);
  const cand = buildGraph(rt, candSpec);
  const basePorts = portBySlot(baseSpec);
  const candPorts = portBySlot(candSpec);
  const vc = new rt.MultiInputVerificationCases(base.g, cand.g);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, base.plan, c, basePorts), makeBundle(rt, cand.plan, c, candPorts));
  }
  const ver = JSON.parse(vc.verify(base.reg, base.g, cand.reg, cand.g, 1e-6, 1e-6));
  if (vc.free) vc.free();

  // 2. Profile both.
  const baseProfile = profileSpec(rt, baseSpec, { iterations });
  const candProfile = profileSpec(rt, candSpec, { iterations });

  const bMs = baseProfile.steady.run_ms_median;
  const cMs = candProfile.steady.run_ms_median;
  const speedup = cMs > 0 ? bMs / cMs : null;
  const verdict = !ver.equivalent
    ? 'REJECT: candidate is not numerically equivalent to baseline'
    : speedup !== null && speedup >= 1
      ? `ACCEPT: equivalent, ${speedup.toFixed(2)}x faster, pages Δ ${candProfile.steady.pages_delta_total >= 0 ? '+' : ''}${candProfile.steady.pages_delta_total}`
      : `EQUIVALENT but slower (${speedup !== null ? speedup.toFixed(2) : 'n/a'}x) — reject unless it buys something else`;

  printJsonAndExit({
    ok: true,
    equivalent: ver.equivalent,
    max_abs_error: ver.max_abs_error,
    max_rel_error: ver.max_rel_error,
    cases: cases.length,
    baseline: {
      run_ms_median: bMs,
      run_ms_min: baseProfile.steady.run_ms_min,
      pages_delta_total: baseProfile.steady.pages_delta_total,
      leak_free: baseProfile.steady.leak_free,
    },
    candidate: {
      run_ms_median: cMs,
      run_ms_min: candProfile.steady.run_ms_min,
      pages_delta_total: candProfile.steady.pages_delta_total,
      leak_free: candProfile.steady.leak_free,
    },
    speedup: speedup !== null ? +speedup.toFixed(3) : null,
    verdict,
  }, ver.equivalent ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}
