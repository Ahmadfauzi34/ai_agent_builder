import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn, spawnSync} from 'node:child_process';
import {generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, manifestDigest} from './ingress_provenance.mjs';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const runner = path.join(packageDir, 'interactive_multi_input_ingress.mjs');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-ingress-'));
const ledger = path.join(temporary, 'replay.json');
const trustPath = path.join(temporary, 'trust.json');
const initializer = path.join(packageDir, 'init_ingress_replay_ledger.mjs');
const {privateKey, publicKey} = generateKeyPairSync('ed25519');
fs.writeFileSync(trustPath, JSON.stringify({
  schema: 'burn-research.ingress-trust-policy.v1',
  issuers: ['sensor-a', 'memory-b'].map(source => ({source, key_id: 'lab-key',
    subjects: ['run-1', 'run-2'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})})),
}));

const running = new Set();
let sequence = 0;
function check(condition, message) {
  if (!condition) throw new Error(message);
}
function start() {
  const child = spawn(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {stdio: ['pipe', 'pipe', 'pipe']});
  running.add(child);
  let stderr = '';
  child.stderr.on('data', chunk => { stderr += chunk; });
  const pending = [];
  const lines = readline.createInterface({input: child.stdout});
  lines.on('line', line => pending.shift()?.(JSON.parse(line)));
  return {
    ask(command) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`runner response timeout: ${stderr}`)), 5000);
        pending.push(response => { clearTimeout(timer); resolve(response); });
        child.stdin.write(JSON.stringify({...command, request_id: ++sequence}) + '\n');
      });
    },
    async close() {
      check((await this.ask({op: 'close'})).result.closed, 'session close failed');
      const code = await new Promise(resolve => child.once('close', resolve));
      running.delete(child);
      check(code === 0, `runner did not exit cleanly: ${stderr}`);
    },
  };
}
const graph = {
  op: 'create', numSlots: 3, layers: [{constructor: 'add', args: [31]}],
  steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2,
  ports: [
    {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', minimumRevision: 2},
    {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', minimumRevision: 3},
  ],
  logicalPorts: [{id: 'observation', slot: 0, source: 'sensor-a'}, {id: 'memory', slot: 1, source: 'memory-b'}],
};
const left = {op: 'bind', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation',
  layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 2, fingerprint: 'obs'};
const right = {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state',
  layout: 'feature_axis1_singleton', source: 'memory-b', revision: 3, fingerprint: 'state'};
function signed(binding, manifest, nonce, subject = 'run-1') {
  const port = manifest.ports.find(item => item.slot === binding.slot);
  const claim = canonicalInputClaim(binding, {
    plan_hex: manifest.plan_hex, manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest), logical_port_id: port.logical_port_id,
  }, 'lab-key', subject, nonce);
  return {...binding, proof: {claim, signature: sign(null, Buffer.from(JSON.stringify(claim)), privateKey).toString('base64')}};
}
async function create(client) {
  const response = await client.ask(graph);
  check(response.ok && !response.result.host_provenance.ready, 'graph did not start with missing proofs');
  return response.result.manifest;
}
async function runAndVerify(client) {
  const run = await client.ask({op: 'run'});
  check(run.ok && run.result.host_provenance.ready && JSON.stringify(run.result.values) === '[4,6]', 'durable graph result mismatch');
  const verify = await client.ask({op: 'verify', candidate: [4, 6]});
  check(verify.ok && verify.result.reference.verification.passed, 'durable verification failed');
}

try {
  const missing = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(missing.status !== 0, 'missing replay ledger allowed startup');
  const initialized = spawnSync(process.execPath, [initializer, ledger, 'run-1'], {timeout: 5000});
  check(initialized.status === 0, `initializing replay ledger failed: ${initialized.stderr}`);
  const reset = spawnSync(process.execPath, [initializer, ledger, 'run-1'], {timeout: 5000});
  check(reset.status !== 0, 'initializer reset existing replay history');
  const first = start();
  const capabilities = await first.ask({op: 'capabilities'});
  check(capabilities.result.host_provenance.mode === 'ed25519_host_durable'
    && capabilities.result.host_provenance.host_subject === 'run-1', 'durable host gate did not activate');
  const manifest = await create(first);
  check(!(await first.ask(left)).ok, 'unsigned input accepted');
  check(!(await first.ask(signed(left, manifest, 'wrong-subject', 'run-2'))).ok, 'signed claim bypassed host subject');
  const firstLeft = signed(left, manifest, 'left-1');
  const firstRight = signed(right, manifest, 'right-1');
  check((await first.ask(firstLeft)).ok && (await first.ask(firstRight)).ok, 'initial durable bind rejected');
  await runAndVerify(first);
  await first.close();

  const second = start();
  const nextManifest = await create(second);
  check(!(await second.ask(firstLeft)).ok, 'nonce replay survived restart');
  check(!(await second.ask(signed({...left, revision: 2}, nextManifest, 'left-new-nonce'))).ok,
    'stale revision accepted after restart');
  check((await second.ask(signed({...left, revision: 3}, nextManifest, 'left-2'))).ok, 'next observation rejected');
  fs.mkdirSync(`${ledger}.lock`, {mode: 0o700});
  try {
    check(!(await second.ask(signed({...right, revision: 4}, nextManifest, 'right-2'))).ok,
      'bind succeeded despite failed durable commit');
    const inspect = await second.ask({op: 'inspect'});
    check(!inspect.result.host_provenance.ready && inspect.result.status.graph_preflight.inputs.bound_port_count === 1,
      'failed durable commit left an executable tensor');
    check(!(await second.ask({op: 'run'})).ok, 'execution ignored failed commit');
  } finally {
    fs.rmdirSync(`${ledger}.lock`);
  }
  check((await second.ask(signed({...right, revision: 4}, nextManifest, 'right-2'))).ok,
    'retry after failed commit rejected an unconsumed nonce');
  await runAndVerify(second);
  const concurrent = start();
  const concurrentManifest = await create(concurrent);
  check((await concurrent.ask(signed({...left, revision: 4}, concurrentManifest, 'left-3'))).ok,
    'newer revision from another runner rejected');
  check(!(await second.ask({op: 'run'})).ok, 'run accepted a revision superseded in another process');
  check(!(await second.ask({op: 'verify', candidate: [4, 6]})).ok,
    'verify accepted a revision superseded in another process');
  await concurrent.close();
  await second.close();

  const mismatch = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-2'], {timeout: 5000});
  check(mismatch.status !== 0 && /host subject/i.test(mismatch.stderr.toString()), 'ledger allowed a different host subject');
  const missingCopy = path.join(temporary, 'removed-ledger.json');
  fs.renameSync(ledger, missingCopy);
  const removed = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(removed.status !== 0, 'removed replay ledger allowed a reset on restart');
  fs.renameSync(missingCopy, ledger);
  fs.writeFileSync(ledger, '{broken', {mode: 0o600});
  const corrupt = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(corrupt.status !== 0, 'corrupt replay ledger allowed startup');
  console.log(JSON.stringify({verdict: 'PASS', mode: 'ed25519_host_durable',
    restart_replay_rejected: true, stale_revision_rejected: true, subject_pinned: true,
    failed_commit_clears_input: true, cross_process_invalidation: true,
    missing_and_corrupt_ledger_rejected: true, reference: [4, 6]}));
} finally {
  for (const child of running) child.kill();
  fs.rmSync(temporary, {recursive: true, force: true});
}
