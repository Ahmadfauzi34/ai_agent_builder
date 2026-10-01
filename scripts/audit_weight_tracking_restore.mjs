// Fase 4 follow-up: weight revision + state digest across checkpoint restore.
// Session A: create weighted graph, mutate weights, run -> digest D_A.
// Session B (new process, same ledger): restore receipt, mutate to the SAME
// weights -> digest must equal D_A (digest is weight-content, revision is
// session-local). Then rebind signed inputs, run -> [5,5] with the digest.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn, spawnSync} from 'node:child_process';
import {generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, manifestDigest} from './ingress_provenance.mjs';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const runner = path.join(packageDir, 'interactive_multi_input_ingress.mjs');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'fase4-restore-'));
const ledger = path.join(temporary, 'replay.json');
const trustPath = path.join(temporary, 'trust.json');
const initializer = path.join(packageDir, 'init_ingress_replay_ledger.mjs');
const {privateKey, publicKey} = generateKeyPairSync('ed25519');
fs.writeFileSync(trustPath, JSON.stringify({
  schema: 'burn-research.ingress-trust-policy.v1',
  issuers: ['sensor-a'].map(source => ({source, key_id: 'lab-key',
    subjects: ['run-1'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})})),
}));
spawnSync(process.execPath, [initializer, ledger, 'run-1'], {timeout: 5000});

let sequence = 0;
function start() {
  const args = [runner, packageDir, trustPath, ledger, 'run-1',
    '--allow-state-checkpoint-export', '--allow-checkpoint-restore'];
  const child = spawn(process.execPath, args, {stdio: ['pipe', 'pipe', 'pipe']});
  const pending = [];
  const lines = readline.createInterface({input: child.stdout});
  lines.on('line', line => pending.shift()?.(JSON.parse(line)));
  return {
    ask(command) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('timeout')), 8000);
        pending.push(r => { clearTimeout(timer); resolve(r); });
        child.stdin.write(JSON.stringify({...command, request_id: ++sequence}) + '\n');
      });
    },
    async close() {
      await this.ask({op: 'close'});
      child.kill();
      await new Promise(r => child.once('close', r));
    },
  };
}

const layers = [
  {constructor: 'linear', args: [0, 2, 2, false]},
  {constructor: 'linear', args: [1, 2, 2, false]},
  {constructor: 'add', args: [2]},
  {constructor: 'relu', args: [3]},
];
const steps = [
  {kind: 'unary', layer: 0, slots: [0, 2]},
  {kind: 'unary', layer: 1, slots: [1, 3]},
  {kind: 'binary', layer: 2, slots: [2, 3, 4]},
  {kind: 'unary', layer: 3, slots: [4, 5]},
];
const ports = [0, 1].map(slot =>
  ({slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}));
const graphSpec = {op: 'create', numSlots: 6, layers, steps, outputSlot: 5, ports,
  logicalPorts: [{id: 'a', slot: 0, source: 'sensor-a'}, {id: 'b', slot: 1, source: 'sensor-a'}]};

function signed(binding, manifest, nonce) {
  const port = manifest.ports.find(item => item.slot === binding.slot);
  const claim = canonicalInputClaim({...binding}, {
    plan_hex: manifest.plan_hex, manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest), logical_port_id: port.logical_port_id,
  }, 'lab-key', 'run-1', nonce);
  return {...binding, proof: {claim,
    signature: sign(null, Buffer.from(JSON.stringify(claim)), privateKey).toString('base64')}};
}
const mkbind = (slot, values, rev) => ({op: 'bind', slot, values, shape: [1, 2, 1, 1],
  role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: rev,
  fingerprint: 'fp'});

const check = (cond, msg) => { if (!cond) throw new Error('FAIL: ' + msg); };

// ---- Session A ----
const a = start();
const manifestA = (await a.ask(graphSpec)).result.manifest;
await a.ask(signed(mkbind(0, [2, 3], 1), manifestA, 'n1'));
await a.ask(signed(mkbind(1, [0, 0], 1), manifestA, 'n2'));
await a.ask({op: 'setWeights', layer: 0, values: [1, 1, 1, 1]});
await a.ask({op: 'setWeights', layer: 1, values: [0, 0, 0, 0]});
const runA = await a.ask({op: 'run'});
check(runA.ok, 'session A run failed: ' + JSON.stringify(runA.error_envelope));
check(JSON.stringify(runA.result.values) === '[5,5]', 'session A numerics wrong');
const dA = runA.result.execution_state.state_digest;
check(runA.result.execution_state.registry_revision === 2, 'session A revision wrong');
const receiptId = runA.result.execution_receipt.receipt_id;
await a.close();

// ---- Session B: restore, mutate to identical weights ----
const b = start();
const restored = await b.ask({op: 'restore', receipt_id: receiptId});
check(restored.ok, 'restore failed: ' + JSON.stringify(restored.error_envelope));
const swB = await b.ask({op: 'setWeights', layer: 0, values: [1, 1, 1, 1]});
check(swB.ok, 'setWeights after restore failed: ' + JSON.stringify(swB.error_envelope));
const dB = swB.result.execution_state.state_digest;
check(dB === dA, `digest mismatch after restore: ${dB} vs ${dA}`);
check(swB.result.execution_state.registry_revision === 1, 'restored revision should be session-local (1)');
check(JSON.stringify(swB.result.execution_state.layer_weight_revisions) === '[1,0,0,0]',
  'restored per-layer revisions wrong');
// rebind with fresh nonces and run: live weights + restored graph -> [5,5]
const manifestB = restored.result.manifest;
await b.ask(signed(mkbind(0, [2, 3], 2), manifestB, 'n3'));
await b.ask(signed(mkbind(1, [0, 0], 2), manifestB, 'n4'));
const runB = await b.ask({op: 'run'});
check(runB.ok, 'session B run failed: ' + JSON.stringify(runB.error_envelope));
check(JSON.stringify(runB.result.values) === '[5,5]', 'session B numerics wrong');
check(runB.result.execution_state.state_digest === dA, 'session B run digest drifted');
await b.close();

console.log(JSON.stringify({verdict: 'PASS', digest_stable_across_restore: dA === dB,
  revision_session_local: true, restored_run: runB.result.values}));
