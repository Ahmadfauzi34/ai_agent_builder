#!/usr/bin/env node
// burn-math verify: baseline-vs-candidate GRAPH verification (WASM-direct).
// Usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]
//        node verify.mjs <baseline.json> <candidate.json> --fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H] [absTol] [relTol]
//        node verify.mjs --batch <pairs.json> [--fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H]]
// --fuzz generates its cases in-process via lib/fuzz.mjs (the same
// generator as fuzz_cases.mjs): golden + 6 boundary + N seeded-random
// cases, with discrete (embedding-id) ports clamped to [0, vocab).
// --fuzz conflicts with a cases file. In batch mode the conflict is a
// per-pair {ok:false} record (the batch continues); a pair without
// explicit cases gets its cases generated from its own baseline spec,
// with the same global fuzz options, and its result gains the same
// `fuzz` key as single mode. Tolerances stay per-pair.
import { loadWasm, buildGraph, makeBundle } from '../lib/wasm_direct.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { generateCases } from '../lib/fuzz.mjs';
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

function verifyOne(rt, pair, fuzz = null) {
  const { baseline: baseFile, candidate: candFile } = pair;
  let base = null, cand = null, vc = null;
  try {
    const absTol = toTolerance(pair.absTol, 'absTol');
    const relTol = toTolerance(pair.relTol, 'relTol');
    const baseSpec = loadOrTag(baseFile);
    const candSpec = loadOrTag(candFile);
    // Batch --fuzz: a pair with explicit cases conflicts (per-pair
    // record, batch continues); otherwise cases come from this pair's
    // own baseline spec. Generation happens before any engine object
    // is built, so a generation failure is a clean pair record.
    let cases;
    let fuzzOut = null;
    if (fuzz) {
      if (pair.cases !== undefined && pair.cases !== null && pair.cases !== '') {
        throw new Error('cannot combine --fuzz with cases');
      }
      const gen = generateCases(baseSpec, fuzz);
      cases = gen.cases;
      fuzzOut = { seed: gen.meta.seed, count: fuzz.count, cases: cases.length, discrete_ports: gen.meta.discrete_ports };
    } else {
      cases = loadCasesOrTag(pair.cases, baseSpec);
    }

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
      ...(fuzzOut ? { fuzz: fuzzOut } : {}),
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
  console.error('usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol] | node verify.mjs <baseline.json> <candidate.json> --fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H] [absTol] [relTol] | node verify.mjs --batch <pairs.json> [--fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H]]');
  process.exit(2);
}

if (argv[0] === '--batch') {
  const pairsFile = argv[1];
  if (!pairsFile || pairsFile.startsWith('--')) {
    console.error('usage: node verify.mjs --batch <pairs.json> [--fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H]]');
    process.exit(2);
  }
  // Batch fuzz flags come ONLY after the pairs file (same grammar as
  // single mode: bare integer after --fuzz is its count). Anything
  // else after the pairs file is usage, never silently ignored.
  let fuzz = null;
  const rest = argv.slice(2);
  for (let i = 0; i < rest.length; i++) {
    const a = rest[i];
    const needFuzz = () => (fuzz ??= { count: 8, seed: 42, lo: -2, hi: 2 });
    if (a === '--fuzz') {
      needFuzz();
      const nxt = rest[i + 1];
      if (nxt !== undefined && /^\d+$/.test(nxt)) { fuzz.count = Number(nxt); i++; }
    }
    else if (a === '--fuzz-seed') needFuzz().seed = Number(rest[++i]);
    else if (a === '--fuzz-lo') needFuzz().lo = Number(rest[++i]);
    else if (a === '--fuzz-hi') needFuzz().hi = Number(rest[++i]);
    else {
      console.error('usage: node verify.mjs --batch <pairs.json> [--fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H]]');
      process.exit(2);
    }
  }
  if (fuzz) {
    if (!Number.isInteger(fuzz.seed) || !Number.isInteger(fuzz.count) || fuzz.count < 0)
      printJsonAndExit({ ok: false, error: '--fuzz-seed and --fuzz count must be integers, count >= 0' }, 2);
    if (!Number.isFinite(fuzz.lo) || !Number.isFinite(fuzz.hi))
      printJsonAndExit({ ok: false, error: '--fuzz-lo and --fuzz-hi must be finite numbers' }, 2);
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
    ...verifyOne(rt, p ?? {}, fuzz),
  }));
  const allOk = results.every((r) => r.ok && r.equivalent);
  printJsonAndExit(results, allOk ? 0 : 2);
}

const [baseFile, candFile, ...rest] = argv;
if (!baseFile || !candFile) {
  console.error('usage: node verify.mjs <baseline.json> <candidate.json> [cases.json] [absTol] [relTol]');
  process.exit(2);
}

// Single-mode flags after the two specs. A bare integer after --fuzz is
// its count (default 8). Positional order with --fuzz: a positional
// BEFORE the first --fuzz flag is the cases file (conflict, exit 2);
// positionals AFTER it are [absTol, relTol], so `--fuzz 1e-6` means
// count=default, absTol=1e-6. Without --fuzz the order stays
// [casesFile, absTol, relTol].
let fuzz = null;
const prePositionals = [];
const postPositionals = [];
for (let i = 0; i < rest.length; i++) {
  const a = rest[i];
  const needFuzz = () => (fuzz ??= { count: 8, seed: 42, lo: -2, hi: 2 });
  if (a === '--fuzz') {
    needFuzz();
    const nxt = rest[i + 1];
    if (nxt !== undefined && /^\d+$/.test(nxt)) { fuzz.count = Number(nxt); i++; }
  }
  else if (a === '--fuzz-seed') needFuzz().seed = Number(rest[++i]);
  else if (a === '--fuzz-lo') needFuzz().lo = Number(rest[++i]);
  else if (a === '--fuzz-hi') needFuzz().hi = Number(rest[++i]);
  else (fuzz ? postPositionals : prePositionals).push(a);
}
const [casesFile, absTolArg, relTolArg] = fuzz
  ? [undefined, ...postPositionals]
  : prePositionals;
if (fuzz && prePositionals.length > 0) {
  printJsonAndExit({ ok: false, error: 'cannot combine --fuzz with a cases file' }, 2);
}
if (fuzz && postPositionals.length > 0 && !Number.isFinite(Number(postPositionals[0]))) {
  // With --fuzz, the only non-numeric positional in verify's grammar is
  // the cases file — a tolerance must be a finite number.
  printJsonAndExit({ ok: false, error: 'cannot combine --fuzz with a cases file' }, 2);
}
if (fuzz) {
  if (!Number.isInteger(fuzz.seed) || !Number.isInteger(fuzz.count) || fuzz.count < 0)
    printJsonAndExit({ ok: false, error: '--fuzz-seed and --fuzz count must be integers, count >= 0' }, 2);
  if (!Number.isFinite(fuzz.lo) || !Number.isFinite(fuzz.hi))
    printJsonAndExit({ ok: false, error: '--fuzz-lo and --fuzz-hi must be finite numbers' }, 2);
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
  let cases;
  let fuzzOut = null;
  if (fuzz) {
    const gen = generateCases(baseSpec, fuzz);
    cases = gen.cases;
    fuzzOut = { seed: gen.meta.seed, count: fuzz.count, cases: cases.length, discrete_ports: gen.meta.discrete_ports };
  } else {
    cases = loadCasesOrTag(casesFile, baseSpec);
  }

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
  printJsonAndExit({ ok: true, ...result, cases: cases.length, absTol, relTol, case_details: result.cases, verdict: result.equivalent ? 'ACCEPT' : 'REJECT', ...(fuzzOut ? { fuzz: fuzzOut } : {}) }, result.equivalent ? 0 : 2);
} catch (e) {
  const out = { ok: false, error: e instanceof Error ? e.message : String(e) };
  if (e?.code) { out.code = e.code; out.phase = e.phase ?? null; }
  printJsonAndExit(out, e?.phase === 'load' || e?.phase === 'loadWasm' ? 1 : 2);
} finally {
  freeSafe(vc);
  if (base) { freeSafe(base.g); freeSafe(base.plan); freeSafe(base.reg); }
  if (cand) { freeSafe(cand.g); freeSafe(cand.plan); freeSafe(cand.reg); }
}
