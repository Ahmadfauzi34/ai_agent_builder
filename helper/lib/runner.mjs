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
      const resolve = pending.get(msg.request_id);
      if (resolve) { pending.delete(msg.request_id); resolve(msg); }
    }
  });
  const send = (cmd) => new Promise((resolve, reject) => {
    const request_id = `r${++seq}`;
    const timer = setTimeout(() => {
      if (pending.delete(request_id)) reject(new Error(`runner op timeout: ${cmd.op}`));
    }, OP_TIMEOUT_MS);
    pending.set(request_id, (msg) => { clearTimeout(timer); resolve(msg); });
    child.stdin.write(JSON.stringify({ request_id, ...cmd }) + '\n', (err) => {
      if (err) { clearTimeout(timer); pending.delete(request_id); reject(err); }
    });
  });
  const stop = async () => {
    try { await send({ op: 'close' }); } catch { /* already gone */ }
    try { child.stdin.end(); } catch { /* already gone */ }
    await new Promise((r) => child.on('exit', r));
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

export function failCli(message, envelope = null, code = 1) {
  const text = message instanceof Error ? message.message : String(message);
  if (envelope) printJsonAndExit({ ok: false, error: text, error_envelope: envelope }, code);
  else printJsonAndExit({ ok: false, error: text }, code);
}
