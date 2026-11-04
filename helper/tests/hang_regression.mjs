// Hang regression for helper/lib/runner.mjs (quirk #424, ported from the
// skill's S7 regression): a runner child that dies before answering must
// fail the parent FAST, never park.
// Usage: node hang_regression.mjs <path-to-lib/runner.mjs-under-test>
// The file under test is COPIED into a temp layout (never modified).
// Because the repo runner resolves its child via ./engine.mjs, the temp
// layout also gets a stub engine.mjs whose pkgDir()/runnerPath() point
// at the temp vendor dir — only path resolution is stubbed; the file
// under test itself is verbatim.
// Variants: (V0) healthy child answers + exits on close — happy path;
// (V1) runner script exits 1 at once; (V2) runner script missing
// entirely (node MODULE_NOT_FOUND on stderr, exit 1).
// Per variant asserts: V0 send resolves + stop resolves; V1/V2 send()
// rejects within 10s with a runner-death message, and stop() resolves
// within 10s. Exit 0 iff all pass.
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { pathToFileURL } from 'node:url';

const underTest = process.argv[2];
if (!underTest) { console.error('usage: hang_regression.mjs <runner.mjs>'); process.exit(2); }

const withTimeout = (p, ms, label) => Promise.race([
  p.then((v) => ({ ok: true, v })),
  new Promise((r) => setTimeout(() => r({ ok: false, v: `${label} timed out after ${ms}ms` }), ms)),
]);

const STUB_ENGINE = `import path from 'node:path';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
export function pkgDir() { return path.join(HERE, '..', 'vendor', 'burn-research-pkg'); }
export function runnerPath() { return path.join(pkgDir(), 'interactive_multi_input_ingress.mjs'); }
`;

const HEALTHY_CHILD = `import readline from 'node:readline';
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let msg; try { msg = JSON.parse(line); } catch { return; }
  if (msg.op === 'close') {
    // Like the real ingress script: reply {closed: true}, then exit.
    process.stdout.write(JSON.stringify({ request_id: msg.request_id, ok: true, result: { closed: true } }) + '\\n');
    process.exit(0);
  }
  process.stdout.write(JSON.stringify({ request_id: msg.request_id, ok: true, result: { echo: msg.op } }) + '\\n');
});
`;

let failures = 0;
for (const variant of ['V0-healthy', 'V1-exit1', 'V2-missing']) {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'runner-hang-reg-'));
  const libDir = path.join(base, 'lib');
  const vendorDir = path.join(base, 'vendor', 'burn-research-pkg');
  fs.mkdirSync(libDir, { recursive: true });
  fs.mkdirSync(vendorDir, { recursive: true });
  fs.copyFileSync(underTest, path.join(libDir, 'runner.mjs'));
  fs.writeFileSync(path.join(libDir, 'engine.mjs'), STUB_ENGINE);
  if (variant === 'V0-healthy' || variant === 'V1-exit1') {
    fs.writeFileSync(path.join(vendorDir, 'interactive_multi_input_ingress.mjs'),
      variant === 'V0-healthy' ? HEALTHY_CHILD : 'process.exit(1);\n');
  }
  const { startRunner } = await import(pathToFileURL(path.join(libDir, 'runner.mjs')).href);
  const r = startRunner();
  await new Promise((x) => setTimeout(x, 1200)); // child settled by now
  let sendOk, stopOk, sendV, stopV;
  if (variant === 'V0-healthy') {
    const sendRes = await withTimeout(
      r.send({ op: 'ping' }).then((m) => `resolved ok=${m.ok}`, (e) => `rejected: ${e.message.slice(0, 120)}`),
      10000, 'send()');
    const stopRes = await withTimeout(r.stop().then(() => 'resolved'), 10000, 'stop()');
    sendV = sendRes.v; stopV = stopRes.v;
    sendOk = sendRes.ok && String(sendV).startsWith('resolved ok=true');
    stopOk = stopRes.ok && stopV === 'resolved';
  } else {
    const sendRes = await withTimeout(
      r.send({ op: 'validatePlan' }).then(() => 'UNEXPECTED RESOLVE', (e) => `rejected: ${e.message.slice(0, 160)}`),
      10000, 'send()');
    const stopRes = await withTimeout(r.stop().then(() => 'resolved'), 10000, 'stop()');
    sendV = sendRes.v; stopV = stopRes.v;
    sendOk = sendRes.ok && String(sendV).startsWith('rejected:') && /runner/i.test(String(sendV));
    stopOk = stopRes.ok && stopV === 'resolved';
  }
  console.log(`${variant}: send ${sendOk ? 'PASS' : 'FAIL'} (${sendV}); stop ${stopOk ? 'PASS' : 'FAIL'} (${stopV})`);
  if (!sendOk || !stopOk) failures++;
  fs.rmSync(base, { recursive: true, force: true });
}
console.log(failures === 0 ? 'HANG REGRESSION: ALL PASS' : `HANG REGRESSION: ${failures} variant(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);
