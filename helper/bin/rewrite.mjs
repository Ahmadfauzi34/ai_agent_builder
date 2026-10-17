#!/usr/bin/env node
// burn-math rewrite: apply a cataloged, proof-backed graph rewrite.
//
// Usage: node rewrite.mjs list
//        node rewrite.mjs apply <name> <spec.json> [out.json] [--verify]
//
// `list` shows the catalog (references/rewrites.v1.json) with each entry's
// recorded proof. `apply` runs the transform from ../lib/rewrites.mjs, then
// gates the result through shapecheck (always); with --verify it also proves
// the transformed graph against the ORIGINAL via verify.mjs and reports the
// verdict. Prints {ok, rewrite, shapecheck, verify?, spec?|out_file}.
// Exit 0 iff applied (+ shapecheck clean, + verify equivalent when asked).
import { loadSpec } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import { listRewrites, applyRewrite, RewriteError } from '../lib/rewrites.mjs';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const [cmd, ...rest] = process.argv.slice(2);

function runCli(script, args) {
  try {
    const out = execFileSync('node', [path.join(HERE, script), ...args], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
    return { json: JSON.parse(out), exit: 0 };
  } catch (e) {
    const stdout = e.stdout ? e.stdout.toString() : '';
    let json = null;
    try { json = JSON.parse(stdout); } catch { /* not JSON */ }
    return { json, exit: e.status ?? 1, stderr: e.stderr ? e.stderr.toString().slice(0, 300) : '' };
  }
}

try {
  if (cmd === 'list') {
    const catalog = JSON.parse(fs.readFileSync(path.join(HERE, '..', 'references', 'rewrites.v1.json'), 'utf8'));
    printJson({
      ok: true,
      engine_build: catalog.engine_build,
      proof_date: catalog.proof_date,
      rewrites: catalog.rewrites.map((r) => ({
        name: r.name, description: r.description, preconditions: r.preconditions, proof: r.proof,
      })),
    });
    process.exit(0);
  }

  if (cmd === 'apply') {
    const name = rest[0], specFile = rest[1];
    if (!name || !specFile) throw new RewriteError('usage: node rewrite.mjs apply <name> <spec.json> [out.json] [--verify]');
    const verify = rest.includes('--verify');
    const outFile = rest.find((a) => a.endsWith('.json') && a !== specFile);

    const spec = loadSpec(specFile);
    let next;
    try {
      next = applyRewrite(name, spec);
    } catch (e) {
      if (e instanceof RewriteError) { printJson({ ok: false, error: e.message }); process.exit(2); }
      throw e;
    }

    // Gate 1 (always): shapecheck the transformed graph.
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rw-'));
    const tmpSpec = path.join(tmp, 'after.json');
    fs.writeFileSync(tmpSpec, JSON.stringify(next));
    const sc = runCli('shapecheck.mjs', [tmpSpec]);
    const shapeOk = sc.exit === 0 && sc.json?.ok === true;

    // Gate 2 (optional): prove equivalence vs the original.
    let verifyRes = null, verifyOk = true;
    if (verify) {
      const v = runCli('verify.mjs', [specFile, tmpSpec]);
      verifyOk = v.exit === 0 && v.json?.equivalent === true;
      verifyRes = { exit: v.exit, equivalent: v.json?.equivalent ?? null, max_abs_error: v.json?.max_abs_error ?? null, error: v.json?.error ?? v.stderr ?? null };
    }

    const okAll = shapeOk && verifyOk;
    if (outFile) fs.writeFileSync(outFile, JSON.stringify(next, null, 1));
    printJson({
      ok: okAll,
      rewrite: name,
      shapecheck: { ok: shapeOk, mismatches: sc.json?.mismatches ?? null, warnings: sc.json?.warnings ?? null },
      verify: verify ? verifyRes : null,
      ...(outFile ? { out_file: outFile } : { spec: next }),
    });
    try { fs.rmSync(tmp, { recursive: true }); } catch { /* best effort */ }
    process.exit(okAll ? 0 : 2);
  }

  printJson({ ok: false, error: `unknown command '${cmd}' (use: list | apply)` });
  process.exit(2);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
