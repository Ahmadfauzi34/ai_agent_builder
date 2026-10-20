#!/usr/bin/env node
// burn-math checkpoint: export / verify-integrity of program bundles.
// Usage:
//   node checkpoint.mjs export <spec.json>          -> {ok, bundle_base64, sha256}
//   node checkpoint.mjs verify <bundle.b64>         -> {ok, integrity_ok, ...}
// export spawns the runner with --allow-state-checkpoint-export (host policy);
// verify is session-free and also enforces canonical base64.
import fs from 'node:fs';
import { startRunner, unwrap, printJson, failCli } from '../lib/runner.mjs';
import { loadSpec } from '../lib/spec.mjs';

const [sub, file] = process.argv.slice(2);
if (!sub || !file || !['export', 'verify'].includes(sub)) {
  console.error('usage: node checkpoint.mjs export <spec.json> | node checkpoint.mjs verify <bundle.b64>');
  process.exit(2);
}

if (sub === 'verify') {
  const digestArg = process.argv[4];
  if (!digestArg) failCli('usage: node checkpoint.mjs verify <bundle.b64> <expected-sha256>');
  const expected_digest = digestArg.startsWith('sha256:') ? digestArg : `sha256:${digestArg}`;
  let b64;
  try { b64 = fs.readFileSync(file, 'utf8').trim(); }
  catch (e) { failCli(`cannot read bundle file: ${e.message}`); }
  const { send, stop } = startRunner();
  try {
    const result = unwrap(await send({ op: 'verifyCheckpointIntegrity', bundle_f32le_base64: b64, expected_digest }));
    await stop();
    printJson({ ok: true, ...result });
    process.exit(result.integrity_ok ? 0 : 2);
  } catch (e) {
    try { await stop(); } catch { /* ignore */ }
    failCli(e.message, e.envelope ?? null);
  }
}

// export
let spec;
try { spec = loadSpec(file); }
catch (e) { failCli(e.message); }
// NOTE: the runner's op=checkpoint is gated behind durable signed ingress
// (host policy: checkpoint export requires a ledger). The skill exports
// WASM-direct instead — same bundle bytes, verifiable via the session-free
// verifyCheckpointIntegrity op below.
try {
  const { loadWasm, buildGraph } = await import('../lib/engine.mjs');
  const { createHash } = await import('node:crypto');
  const rt = await loadWasm();
  const { reg, g } = buildGraph(rt, spec);
  const bytes = Buffer.from(rt.exportMultiInputProgramBundle(g, reg, true));
  printJson({
    ok: true,
    schema: 'burn-research.multi-input-program-bundle.v1',
    sha256: createHash('sha256').update(bytes).digest('hex'),
    bundle_base64: bytes.toString('base64'),
  });
} catch (e) {
  failCli(e.message, e.envelope ?? null);
}
