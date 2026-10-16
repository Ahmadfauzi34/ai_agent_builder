#!/usr/bin/env node
// burn-math prng_check: certify and re-verify deterministic PRNG streams.
//
// Usage: node prng_check.mjs [algo=mulberry32] [seed=1] [count=1000]
//        node prng_check.mjs verify <manifest.json>
//
// The engine itself is deterministic (audited zero-init, no stochastic ops),
// so PRNG risk lives in the AGENT's harness: initializers, dropout masks,
// data shuffling, fuzz seeds. This tool certifies a stream: given
// (algorithm, seed, count) it regenerates the exact f64 values and their
// sha256. Record the manifest alongside any stochastic experiment; anyone
// can re-verify the stream bit-exactly later.
//
// Generate prints {ok, algorithm, seed, count, sha256, preview} and exits 0.
// `verify` re-generates from the manifest and compares hashes; exits 0 iff
// the recorded hash matches, 2 otherwise.
import { printJson } from '../lib/runner.mjs';
import { generateStream, PRNGS } from '../lib/prng.mjs';
import fs from 'node:fs';

const [cmd, a2, a3, a4] = process.argv.slice(2);

try {
  if (cmd === 'verify') {
    if (!a2) throw new Error('usage: node prng_check.mjs verify <manifest.json>');
    const m = JSON.parse(fs.readFileSync(a2, 'utf8'));
    if (!m.algorithm || m.seed === undefined || !m.count || !m.sha256)
      throw new Error('manifest must contain {algorithm, seed, count, sha256}');
    const { sha256 } = generateStream(m.algorithm, m.seed, m.count);
    const verified = sha256 === m.sha256;
    printJson({
      ok: verified,
      verified,
      algorithm: m.algorithm,
      seed: m.seed,
      count: m.count,
      expected_sha256: m.sha256,
      actual_sha256: sha256,
    });
    process.exit(verified ? 0 : 2);
  }

  const algo = cmd ?? 'mulberry32';
  const seed = a2 !== undefined ? Number(a2) : 1;
  const count = a3 !== undefined ? Number(a3) : 1000;
  if (!PRNGS[algo]) throw new Error(`unknown prng '${algo}' (available: ${Object.keys(PRNGS).join(', ')})`);
  if (!Number.isInteger(seed) || !Number.isInteger(count) || count <= 0)
    throw new Error('seed and count must be positive integers');
  const { values, sha256 } = generateStream(algo, seed, count);
  printJson({
    ok: true,
    algorithm: algo,
    seed,
    count,
    sha256,
    preview: values.slice(0, 5),
    note: 'save this manifest; `prng_check.mjs verify <manifest.json>` reproduces the stream exactly',
  });
  process.exit(0);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
