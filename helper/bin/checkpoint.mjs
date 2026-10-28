#!/usr/bin/env node
// burn-math checkpoint: export / verify-integrity / restore of program bundles.
// Usage:
//   node checkpoint.mjs export <spec.json>          -> {ok, bundle_base64, sha256}
//   node checkpoint.mjs verify <bundle.b64> <expected-sha256> -> {ok, verdict, integrity_ok, ...}
//   node checkpoint.mjs restore <bundle.b64> <spec.json> [expected-sha256]
//        -> import the bundle into a FRESH registry, run the spec's inputs,
//           print {ok, restored, bundle_sha256, program_identity, outputs}
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { printJson, printJsonAndExit, exitAfterFlush, failCli } from '../lib/runner.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { loadWasm, buildGraph, makeBundle, verifyEngineIntegritySync } from '../lib/wasm_direct.mjs';

const args = process.argv.slice(2);
const [sub, file] = args;
if (!sub || !file || !['export', 'verify', 'restore'].includes(sub) || args.includes('--help') || args.includes('-h')) {
  console.error('usage: node checkpoint.mjs export <spec.json> | node checkpoint.mjs verify <bundle.b64> <expected-sha256> | node checkpoint.mjs restore <bundle.b64> <spec.json> [expected-sha256]');
  process.exit(2);
}

// Strict canonical base64 decode: standard alphabet, length multiple of 4,
// correct '=' padding (including zeroed padding bits via round-trip check),
// and no whitespace.
function decodeCanonicalBase64(b64, phase = 'verifyCheckpointIntegrity') {
  if (typeof b64 !== 'string' || b64.length === 0) {
    const msg = 'bundle_f32le_base64 must be a non-empty base64 string';
    failCli(msg, {
      code: 'invalid_argument', phase, message: msg, execution_started: false,
      path: 'bundle_f32le_base64', remediation_hint: 'use the bundle_f32le_base64 value from op=checkpoint',
    });
  }
  if (b64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) {
    const msg = 'bundle_f32le_base64 must be canonical base64 (standard alphabet, length multiple of 4, correct padding, no whitespace)';
    failCli(msg, {
      code: 'invalid_argument', phase, message: msg, execution_started: false,
      path: 'bundle_f32le_base64', constraint: 'canonical_base64',
      remediation_hint: 'use the bundle_f32le_base64 value from op=checkpoint unchanged',
    });
  }
  const bytes = Buffer.from(b64, 'base64');
  if (bytes.length === 0 || bytes.toString('base64') !== b64) {
    const msg = 'bundle_f32le_base64 must be canonical base64 (standard alphabet, length multiple of 4, correct padding, no whitespace)';
    failCli(msg, {
      code: 'invalid_argument', phase, message: msg, execution_started: false,
      path: 'bundle_f32le_base64', constraint: 'canonical_base64',
      remediation_hint: 'use the bundle_f32le_base64 value from op=checkpoint unchanged',
    });
  }
  return bytes;
}

function normalizeExpectedSha(digestArg, phase = 'verifyCheckpointIntegrity') {
  const expected = digestArg.startsWith('sha256:') ? digestArg : `sha256:${digestArg}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(expected)) {
    const msg = 'expected_digest must be a lowercase sha256 digest like sha256:<64 hex chars>';
    failCli(msg, {
      code: 'invalid_argument', phase, message: msg, execution_started: false,
      path: 'expected_digest',
      remediation_hint: 'use the checkpoint_bytes_sha256 value recorded when the bundle was exported',
    });
  }
  return expected;
}

if (sub === 'verify') {
  const digestArg = process.argv[4];
  if (!digestArg) failCli('usage: node checkpoint.mjs verify <bundle.b64> <expected-sha256>');
  try { verifyEngineIntegritySync(); }
  catch (e) { failCli(e instanceof Error ? e.message : String(e)); }
  let b64;
  try { b64 = fs.readFileSync(file, 'utf8').trim(); }
  catch (e) { failCli(`cannot read bundle file: ${e instanceof Error ? e.message : String(e)}`); }
  const bytes = decodeCanonicalBase64(b64, 'verifyCheckpointIntegrity');
  const expected_digest = normalizeExpectedSha(digestArg, 'verifyCheckpointIntegrity');
  const actual_digest = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
  const integrity_ok = actual_digest === expected_digest;
  printJsonAndExit({
    ok: true,
    integrity_ok,
    actual_digest,
    expected_digest,
    byte_length: bytes.length,
    verified_against: 'expected_digest',
    note: 'integrity_ok=false means at least one byte differs from the exported bundle',
    verdict: integrity_ok ? 'ACCEPT' : 'REJECT',
  }, integrity_ok ? 0 : 2);
}

if (sub === 'restore') {
  const specFile = process.argv[4];
  const expectedSha = process.argv[5];
  if (!specFile) failCli('usage: node checkpoint.mjs restore <bundle.b64> <spec.json> [expected-sha256]');
  let spec;
  try { spec = loadSpec(specFile); }
  catch (e) { failCli(e instanceof Error ? e.message : String(e)); }
  let b64;
  try { b64 = fs.readFileSync(file, 'utf8').trim(); }
  catch (e) { failCli(`cannot read bundle file: ${e instanceof Error ? e.message : String(e)}`); }
  const bytes = decodeCanonicalBase64(b64, 'restore');
  const expected_digest = expectedSha ? normalizeExpectedSha(expectedSha, 'restore') : null;
  let reg = null, g = null, plan = null, bundle = null;
  try {
    const sha256 = createHash('sha256').update(bytes).digest('hex');
    if (expected_digest) {
      const want = expected_digest.slice(7);
      if (want !== sha256) {
        printJsonAndExit({
          ok: false,
          verdict: 'REJECT',
          restored: false,
          integrity_ok: false,
          bundle_sha256: sha256,
          expected_sha256: want,
          actual_digest: `sha256:${sha256}`,
          expected_digest,
        }, 2);
      }
    }
    const rt = await loadWasm();
    reg = new rt.LayerRegistry();
    g = rt.importMultiInputProgramBundle(reg, bytes);
    plan = rt.MultiInputGraphPlan.fromBytes(g.inputPlanV1());
    bundle = makeBundle(rt, plan, spec.inputs, portBySlot(spec));
    const out = g.run(reg, bundle);
    const values = Array.from(out.to_array());
    const shape = Array.from(typeof out.shape === 'function' ? out.shape() : out.shape ?? []);
    out.free();
    let programIdentity = g.programIdentity();
    try { programIdentity = JSON.parse(programIdentity); } catch { /* keep raw string */ }
    printJson({
      ok: true,
      verdict: 'ACCEPT',
      restored: true,
      ...(expected_digest ? { integrity_ok: true } : {}),
      bundle_sha256: sha256,
      program_identity: programIdentity,
      outputs: { shape, values },
    });
  } catch (e) {
    if (e?.code) {
      printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e), code: e.code, phase: e.phase ?? null }, e.phase === 'loadWasm' ? 1 : 2);
    }
    printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
  } finally {
    try { if (bundle?.free) bundle.free(); } catch { /* ignore */ }
    try { if (plan?.free) plan.free(); } catch { /* ignore */ }
    try { if (g?.free) g.free(); } catch { /* ignore */ }
    try { if (reg?.free) reg.free(); } catch { /* ignore */ }
  }
  // v2.5 (F6): the success print above may still be buffered; exit only
  // after it drains instead of truncating it with a bare process.exit(0).
  exitAfterFlush(0);
}

// export
let spec;
try { spec = loadSpec(file); }
catch (e) { failCli(e instanceof Error ? e.message : String(e)); }
let built = null;
try {
  const rt = await loadWasm();
  built = buildGraph(rt, spec);
  const bytes = Buffer.from(rt.exportMultiInputProgramBundle(built.g, built.reg, true));
  printJson({
    ok: true,
    schema: 'burn-research.multi-input-program-bundle.v1',
    sha256: createHash('sha256').update(bytes).digest('hex'),
    bundle_base64: bytes.toString('base64'),
  });
} catch (e) {
  if (e?.code) {
    printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e), code: e.code, phase: e.phase ?? null }, e.phase === 'loadWasm' ? 1 : 2);
  }
  printJsonAndExit({ ok: false, error: e instanceof Error ? e.message : String(e) }, 2);
} finally {
  if (built) {
    try { if (built.g?.free) built.g.free(); } catch { /* ignore */ }
    try { if (built.plan?.free) built.plan.free(); } catch { /* ignore */ }
    try { if (built.reg?.free) built.reg.free(); } catch { /* ignore */ }
  }
}
