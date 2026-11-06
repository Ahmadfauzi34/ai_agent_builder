// Shared helper: spawn the burn-research host runner and speak its
// JSON-lines op protocol. The runner script and engine package are located
// via lib/engine.mjs (packed <pkg>/ layout or repo source layout).
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import { pkgDir, runnerPath } from './engine.mjs';

export const PKG_DIR = pkgDir();
export const RUNNER = runnerPath();

const OP_TIMEOUT_MS = 120_000;

export function startRunner(extraArgs = []) {
  // Runner argv layout: [2]=pkgdir [3]=trust-policy [4]=ledger [5]=subject
  // [6..]=startup options. Pad the middle slots when passing options.
  const argv = extraArgs.length
    ? [PKG_DIR, '', '', '', ...extraArgs]
    : [PKG_DIR];
  const child = spawn('node', [RUNNER, ...argv],
    { stdio: ['pipe', 'pipe', 'pipe'] });
  let buf = '';
  const pending = new Map();
  let seq = 0;
  let stderr = '';
  // Upstream port of the skill's S7 hang fix (quirk #424, 2026-11-04):
  // the runner child can die before answering (missing engine at
  // import, missing runner file) and the parent used to wait for a
  // stdout line that never comes: send() pend up to the 120s op
  // timeout, then stop() awaited an 'exit' that had already fired
  // and parked FOREVER. Fail-closed now: child death rejects every
  // pending op at once, with the stderr tail so the cause (e.g.
  // MODULE_NOT_FOUND) surfaces instead of a bare timeout, and
  // stop() returns immediately when already exited.
  let exited = false;
  let exitCode = null;
  const deathError = (op) => {
    // Headline first: prefer the line that names the cause
    // (e.g. Cannot find module); a bare stack frame does not.
    const lines = stderr.split('\n').map((l) => l.trim()).filter(Boolean);
    const head = (lines.find((l) => /Error|Cannot find|MODULE_NOT_FOUND/.test(l)) ?? lines[0] ?? '').slice(0, 300);
    return new Error(`runner died (code ${exitCode}) before responding to '${op}'` +
      (head ? `; runner stderr: ${head}` : ' (no runner stderr)'));
  };
  const failAllPending = () => {
    for (const [id, p] of pending) {
      clearTimeout(p.timer);
      p.reject(deathError(p.op));
      pending.delete(id);
    }
  };
  child.on('exit', (code) => { exited = true; exitCode = code; failAllPending(); });
  child.on('error', (err) => {
    exited = true; exitCode = err.code ?? 'spawn-error'; failAllPending();
  });
  child.stderr.on('data', (d) => { stderr += d.toString(); });
  child.stdout.on('data', (d) => {
    buf += d.toString();
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 1);
      if (!line.trim()) continue;
      let msg;
      try { msg = JSON.parse(line); } catch { continue; }
      const p = pending.get(msg.request_id);
      if (p) { pending.delete(msg.request_id); p.resolve(msg); }
    }
  });
  const send = (cmd) => new Promise((resolve, reject) => {
    if (exited) { reject(deathError(cmd.op)); return; }
    const request_id = `r${++seq}`;
    const timer = setTimeout(() => {
      if (pending.delete(request_id)) reject(new Error(`runner op timeout: ${cmd.op}`));
    }, OP_TIMEOUT_MS);
    pending.set(request_id, {
      op: cmd.op, timer,
      resolve: (msg) => { clearTimeout(timer); resolve(msg); },
      reject: (err) => { clearTimeout(timer); reject(err); },
    });
    child.stdin.write(JSON.stringify({ request_id, ...cmd }) + '\n', (err) => {
      if (err) { clearTimeout(timer); pending.delete(request_id); reject(err); }
    });
  });
  const stop = async () => {
    if (exited) return;
    try { await send({ op: 'close' }); } catch { /* already gone */ }
    if (exited) return;
    try { child.stdin.end(); } catch { /* already gone */ }
    await Promise.race([
      new Promise((r) => child.on('exit', r)),
      new Promise((r) => setTimeout(() => {
        try { child.kill('SIGKILL'); } catch { /* already gone */ }
        r();
      }, 5000)),
    ]);
  };
  return { send, stop, get stderr() { return stderr; } };
}

// Throw the structured error envelope on ok:false so CLIs print it verbatim.
export function unwrap(resp) {
  if (resp && resp.ok === true) return resp.result;
  const env = resp?.error_envelope;
  const err = new Error(resp?.error || 'runner op failed');
  err.envelope = env ?? null;
  err.raw = resp ?? null;
  throw err;
}

export function printJson(obj) {
  process.stdout.write(JSON.stringify(obj, null, 2) + '\n');
}

// Drain-safe stdout (ported from the agent pack, F6): a plain async write
// followed by process.exit() truncates large outputs (a single writeSync
// to a pipe accepted only ~146KB of a 323KB batch; a deferred drain-exit
// lets execution fall through to later code). This loops synchronously
// until every byte is accepted, so the JSON is never truncated and no
// later code runs after the print. EAGAIN waits briefly; any other error
// (e.g. EPIPE, reader gone) ends the loop and we exit anyway.
export function printJsonAndExit(obj, code) {
  const buf = Buffer.from(JSON.stringify(obj, null, 2) + '\n', 'utf8');
  const wait = new Int32Array(new SharedArrayBuffer(4));
  let off = 0;
  while (off < buf.length) {
    try {
      const n = fs.writeSync(process.stdout.fd, buf, off, buf.length - off);
      if (n === 0) Atomics.wait(wait, 0, 0, 10);
      else off += n;
    } catch (e) {
      // EAGAIN: pipe buffer full, reader has not drained yet — wait and retry.
      // Anything else (EPIPE: reader gone) means the JSON cannot be
      // delivered; exit with the intended code instead of crashing.
      if (e && e.code === 'EAGAIN') { Atomics.wait(wait, 0, 0, 10); continue; }
      break;
    }
  }
  try { process.stdout.on('error', () => {}); } catch { /* ignore */ }
  process.exit(code);
}

// Drain-safe exit after output was already written with printJson: only
// exit immediately when nothing is still buffered, otherwise wait for drain.
export function exitAfterFlush(code) {
  if (process.stdout.writableLength === 0) process.exit(code);
  else process.stdout.once('drain', () => process.exit(code));
}

// errText (PR Tahap 1, 2026-11-06): extract a printable message from
// anything a catch can hold. wasm-bindgen Result<T, String> functions
// (setWeightsFlat, WasmTensor::new, compileMultiInputGraph, ...) throw a
// raw JS string, not an Error — reading e.message on one yields
// undefined, which JSON.stringify drops (bare {ok:false}) or prints as
// "undefined". errText is the single funnel: Error -> message, string ->
// itself, {message} -> message, anything else -> String(...).
export function errText(e) {
  if (e instanceof Error) return e.message;
  if (typeof e === 'string') return e;
  if (e && typeof e === 'object' && typeof e.message === 'string') return e.message;
  return String(e ?? 'unknown error');
}

export function failCli(message, envelope = null, code = 1) {
  const text = errText(message);
  if (envelope) printJsonAndExit({ ok: false, error: text, error_envelope: envelope }, code);
  else printJsonAndExit({ ok: false, error: text }, code);
}

// F2 (strict CLI flags): reject unknown --flags before any file is read
// or written. Unknown flags used to be silently ignored while their
// VALUES leaked into positional slots — e.g. recurse search --beam 2
// wrote best_spec to a file literally named "2", and the same pattern
// lived in every bin that derives a path or parameter from a token
// stream. args are the tokens after the subcommand (or all tokens for
// command-less CLIs). valued flags consume the next token; boolean flags
// take none. A token following a valued flag is its value, even if it
// starts with '--'. Anything else starting with '--' fails fast with
// exit 2 and the known-flag list, so a typo can never become a file.
export function assertKnownFlags(args, { valued = [], boolean = [], prog = 'cli' } = {}) {
  const valuedSet = new Set(valued);
  const known = new Set([...valued, ...boolean]);
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (i > 0 && valuedSet.has(args[i - 1])) continue;
    if (!a.startsWith('--')) continue;
    if (!known.has(a)) {
      console.error(`unknown flag '${a}' for ${prog} (known flags: ${[...known].sort().join(', ') || 'none'})`);
      process.exit(2);
    }
  }
}
