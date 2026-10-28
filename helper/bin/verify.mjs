#!/usr/bin/env node
// burn-math verify: baseline-vs-candidate GRAPH verification (WASM-direct).
// Usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]
//        node verify.mjs --batch <pairs.json>
import { loadWasm, buildGraph, makeBundle } from '../lib/wasm_direct.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';
import fs from 'node:fs';

function tagLoadError(e) {
  if (e && typeof e === 'object') e.phase = 'load';
  return e;
}

function loadOrTag(file) {
  try { return loadSpec(file); }
  catch (e) { throw tagLoadError(e); }
}

function loadCasesOrTag(casesRef, baseSpec) {
  if (casesRef === undefined || casesRef === null || casesRef === '') return [baseSpec.inputs];
  let cases;
  try {
    cases = typeof casesRef === 'string'
      ? JSON.parse(fs.readFileSync(casesRef, 'utf8'))
      : casesRef;
  } catch (e) {
    throw tagLoadError(new Error(`cannot read cases file ${casesRef}: ${e instanceof Error ? e.message : String(e)}`));
  }
  if (!Array.isArray(cases) || cases.length === 0) {
    throw tagLoadError(new Error('cases must be a non-empty array'));
  }
  return cases;
}

function freeSafe(obj) {
  try { if (obj && typeof obj.free === 'function') obj.free(); } catch { /* ignore */ }
}

// v2.5 (F5): tolerances must be finite numbers >= 0 (0 = exact).
// Previously Number("abc") = NaN flowed into vc.verify unchecked.
function toTolerance(value, name) {
  if (value === undefined) return 1e-6;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 0) throw new Error(`${name} must be a finite number >= 0, got ${JSON.stringify(value)}`);
  return n;
}

function verifyOne(rt, pair) {
  const { baseline: baseFile, candidate: candFile } = pair;
  let base = null, cand = null, vc = null;
  try {
    const absTol = toTolerance(pair.absTol, 'absTol');
    const relTol = toTolerance(pair.relTol, 'relTol');
    const baseSpec = loadOrTag(baseFile);
    const candSpec = loadOrTag(candFile);
    const cases = loadCasesOrTag(pair.cases, baseSpec);

    base = buildGraph(rt, baseSpec);
    cand = buildGraph(rt, candSpec);
    const basePorts = portBySlot(baseSpec);
    const candPorts = portBySlot(candSpec);

    vc = new rt.MultiInputVerificationCases(base.g, cand.g);
    for (const c of cases) {
      const bBundle = makeBundle(rt, base.plan, c, basePorts);
      const cBundle = makeBundle(rt, cand.plan, c, candPorts);
      vc.addCase(bBundle, cBundle);
    }
    const result = JSON.parse(vc.verify(base.reg, base.g, cand.reg, cand.g, absTol, relTol));
    return {
      ok: true,
      ...result,
      cases: cases.length,
      absTol,
      relTol,
      case_details: result.cases,
      verdict: result.equivalent ? 'ACCEPT' : 'REJECT',
    };
  } catch (e) {
    const out = { ok: false, error: e instanceof Error ? e.message : String(e) };
    if (e?.code) { out.code = e.code; out.phase = e.phase ?? null; }
    else if (e?.phase) { out.phase = e.phase; }
    return out;
  } finally {
    freeSafe(vc);
    if (base) { freeSafe(base.g); freeSafe(base.plan); freeSafe(base.reg); }
    if (cand) { freeSafe(cand.g); freeSafe(cand.plan); freeSafe(cand.reg); }
  }
}

const argv = process.argv.slice(2);
if (!argv.length || argv.includes('--help') || argv.includes('-h')) {
  console.error('usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol] | node verify.mjs --batch <pairs.json>');
  process.exit(2);
}

if (argv[0] === '--batch') {
  const pairsFile = argv[1];
  if (!pairsFile) {
    console.error('usage: node verify.mjs --batch <pairs.json>');
    process.exit(2);
  }
  let pairs;
  try {
    pairs = JSON.parse(fs.readFileSync(pairsFile, 'utf8'));
  } catch (e) {
    printJsonAndExit({ ok: false, error: `cannot load pairs file: ${e instanceof Error ? e.message : String(e)}` }, 1);
  }
  if (!Array.isArray(pairs) || pairs.length === 0) {
    printJsonAndExit({ ok: false, error: 'pairs.json must be a non-empty array' }, 1);
  }
  let rt;
  try { rt = await loadWasm(); }
  catch (e) {
    printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e), ...(e?.code ? { code: e.code, phase: e.phase ?? null } : {}) }, 1);
  }
  const results = pairs.map((p) => ({
    pair: { baseline: p?.baseline, candidate: p?.candidate },
    ...verifyOne(rt, p ?? {}),
  }));
  const allOk = results.every((r) => r.ok && r.equivalent);
  printJsonAndExit(results, allOk ? 0 : 2);
}

const [baseFile, candFile, casesFile, absTolArg, relTolArg] = argv;
if (!baseFile || !candFile) {
  console.error('usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]');
  process.exit(2);
}

let absTol, relTol;
try {
  absTol = toTolerance(absTolArg, 'absTol');
  relTol = toTolerance(relTolArg, 'relTol');
} catch (e) {
  printJsonAndExit({ ok: false, error: e.message }, 2);
}

let base = null, cand = null, vc = null;
try {
  const baseSpec = loadOrTag(baseFile);
  const candSpec = loadOrTag(candFile);
  const cases = loadCasesOrTag(casesFile, baseSpec);

  const rt = await loadWasm();
  base = buildGraph(rt, baseSpec);
  cand = buildGraph(rt, candSpec);
  const basePorts = portBySlot(baseSpec);
  const candPorts = portBySlot(candSpec);

  vc = new rt.MultiInputVerificationCases(base.g, cand.g);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, base.plan, c, basePorts), makeBundle(rt, cand.plan, c, candPorts));
  }
  const result = JSON.parse(vc.verify(base.reg, base.g, cand.reg, cand.g, absTol, relTol));
  printJsonAndExit({ ok: true, ...result, cases: cases.length, absTol, relTol, case_details: result.cases, verdict: result.equivalent ? 'ACCEPT' : 'REJECT' }, result.equivalent ? 0 : 2);
} catch (e) {
  const out = { ok: false, error: e instanceof Error ? e.message : String(e) };
  if (e?.code) { out.code = e.code; out.phase = e.phase ?? null; }
  printJsonAndExit(out, e?.phase === 'load' || e?.phase === 'loadWasm' ? 1 : 2);
} finally {
  freeSafe(vc);
  if (base) { freeSafe(base.g); freeSafe(base.plan); freeSafe(base.reg); }
  if (cand) { freeSafe(cand.g); freeSafe(cand.plan); freeSafe(cand.reg); }
}
