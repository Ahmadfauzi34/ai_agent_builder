import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn, spawnSync} from 'node:child_process';
import {generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, manifestDigest} from './ingress_provenance.mjs';
import {IngressReplayLedger} from './ingress_replay_ledger.mjs';
import {encodedF32Matches, executionReceipt, f32ValueBytes, sha256Json, f32ValueDigest} from './ingress_execution_receipt.mjs';

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
  check(run.result.execution_receipt?.authority === 'node_host_observed_wasm_run'
    && run.result.execution_receipt.output.value_sha256 === f32ValueDigest(run.result.values)
    && run.result.output_f32_le_base64 === f32ValueBytes(run.result.values).toString('base64'),
  'durable run did not produce an observed output receipt');
  const verify = await client.ask({op: 'verify', candidate: [4, 6]});
  check(verify.ok && verify.result.reference.verification.passed, 'durable verification failed');
  return run.result.execution_receipt;
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
  const firstReceipt = await runAndVerify(first);
  check(firstReceipt.sequence === 1 && firstReceipt.claim_count === 2
    && firstReceipt.manifest_sha256 === manifestDigest(manifest)
    && JSON.stringify(firstReceipt.input_claims) === JSON.stringify([
      {slot: 0, source: 'sensor-a', revision: '2', claim_sha256: sha256Json(firstLeft.proof.claim)},
      {slot: 1, source: 'memory-b', revision: '3', claim_sha256: sha256Json(firstRight.proof.claim)},
    ]), 'receipt did not bind both signed input claims');
  const {receipt_id: firstId, ...firstRecord} = firstReceipt;
  check(firstId === sha256Json(firstRecord), 'receipt ID is not its canonical content digest');
  await first.close();

  const second = start();
  const recovered = await second.ask({op: 'receipt', receipt_id: firstId, shape: [1, 2, 1, 1], values: [4, 6]});
  check(recovered.ok && recovered.result.output_matches && recovered.result.receipt.receipt_id === firstId,
    'execution receipt did not survive restart');
  const exactRecovered = await second.ask({op: 'receipt', receipt_id: firstId, shape: [1, 2, 1, 1],
    output_f32_le_base64: f32ValueBytes([4, 6]).toString('base64')});
  check(exactRecovered.ok && exactRecovered.result.output_matches, 'exact f32 output did not match persisted receipt');
  const alteredOutput = await second.ask({op: 'receipt', receipt_id: firstId, shape: [1, 2, 1, 1], values: [4, 7]});
  check(alteredOutput.ok && !alteredOutput.result.output_matches, 'modified output matched the recorded receipt');
  check(!(await second.ask({op: 'receipt', receipt_id: 'sha256:' + '0'.repeat(64)})).ok,
    'caller supplied an uncommitted receipt ID');
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
  const secondReceipt = await runAndVerify(second);
  check(secondReceipt.sequence === 2 && secondReceipt.claim_count === 4
    && JSON.stringify(secondReceipt.program_identity) === JSON.stringify(firstReceipt.program_identity),
  'execution sequence, claim count, or program identity mismatch');
  const concurrent = start();
  const concurrentManifest = await create(concurrent);
  check((await concurrent.ask(signed({...left, revision: 4}, concurrentManifest, 'left-3'))).ok,
    'newer revision from another runner rejected');
  check(!(await second.ask({op: 'run'})).ok, 'run accepted a revision superseded in another process');
  check(!(await second.ask({op: 'verify', candidate: [4, 6]})).ok,
    'verify accepted a revision superseded in another process');
  check((await second.ask({op: 'receipt', receipt_id: firstId})).result.receipt.receipt_id === firstId,
    'superseding a claim removed a historical execution receipt');
  await concurrent.close();
  await second.close();

  const mismatch = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-2'], {timeout: 5000});
  check(mismatch.status !== 0 && /host subject/i.test(mismatch.stderr.toString()), 'ledger allowed a different host subject');
  const missingCopy = path.join(temporary, 'removed-ledger.json');
  fs.renameSync(ledger, missingCopy);
  const removed = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(removed.status !== 0, 'removed replay ledger allowed a reset on restart');
  fs.renameSync(missingCopy, ledger);
  const original = fs.readFileSync(ledger);
  const changed = JSON.parse(original);
  changed.executions[0].input_claims[0].claim_sha256 = 'sha256:' + '0'.repeat(64);
  const {receipt_id: ignored, ...changedRecord} = changed.executions[0];
  changed.executions[0].receipt_id = sha256Json(changedRecord);
  fs.writeFileSync(ledger, JSON.stringify(changed) + '\n', {mode: 0o600});
  const forgedHistory = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(forgedHistory.status !== 0, 'receipt with a changed input claim survived ledger history validation');
  fs.writeFileSync(ledger, original, {mode: 0o600});
  fs.writeFileSync(ledger, '{broken', {mode: 0o600});
  const corrupt = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(corrupt.status !== 0, 'corrupt replay ledger allowed startup');

  // Simulate a commit failure after numerical execution. Legacy v1 snapshots
  // without an executions field must still retain their accepted claims.
  const legacyFile = path.join(temporary, 'legacy.json');
  const legacy = new IngressReplayLedger(legacyFile, 'run-1', {initialize: true});
  const oldState = JSON.parse(fs.readFileSync(legacyFile));
  delete oldState.executions;
  fs.writeFileSync(legacyFile, JSON.stringify(oldState) + '\n');
  const migrated = new IngressReplayLedger(legacyFile, 'run-1');
  const claims = [0, 1].map(slot => ({source: `source-${slot}`, subject: 'run-1', slot,
    revision: '1', manifest_sha256: 'sha256:' + 'a'.repeat(64)}));
  claims.forEach((claim, slot) => migrated.commit({claim,
    nonceKey: JSON.stringify([claim.source, 'key', `nonce-${slot}`]),
    revisionKey: JSON.stringify([claim.source, claim.subject, claim.slot])}));
  const identity = {subject: 'run-1', programIdentity: {schema: 'probe'},
    manifestSha256: claims[0].manifest_sha256};
  let numericalExecutionOccurred = false;
  const write = migrated.write;
  migrated.write = () => { throw new Error('injected receipt commit failure'); };
  try {
    migrated.executeWithReceipt(claims, identity, () => {
      numericalExecutionOccurred = true;
      return {shape: [1, 2, 1, 1], values: [4, 6]};
    });
    throw new Error('failed receipt commit returned success');
  } catch (error) {
    check(error.message === 'injected receipt commit failure', 'receipt failure injection reported the wrong error');
  } finally {
    migrated.write = write;
  }
  check(numericalExecutionOccurred && new IngressReplayLedger(legacyFile, 'run-1').load().state.executions.length === 0,
    'failed receipt commit created a durable receipt or bypassed numerical execution');
  const migratedReceipt = migrated.executeWithReceipt(claims, identity,
    () => ({shape: [1, 2, 1, 1], values: [4, 6]})).execution_receipt;
  check(migrated.getReceipt(migratedReceipt.receipt_id).sequence === 1,
    'pre-receipt ledger did not preserve its claim history during the upgrade');
  const negativeZero = executionReceipt({...identity, claims, shape: [1, 2, 1, 1], values: [-0, 2],
    sequence: 2, claimCount: 2});
  check(f32ValueDigest([-0, 2]) !== f32ValueDigest([0, 2])
    && encodedF32Matches(negativeZero, [1, 2, 1, 1], f32ValueBytes([-0, 2]).toString('base64'))
    && !encodedF32Matches(negativeZero, [1, 2, 1, 1], f32ValueBytes([0, 2]).toString('base64')),
  'f32 wire representation erased the sign of negative zero');
  console.log(JSON.stringify({verdict: 'PASS', mode: 'ed25519_host_durable',
    restart_replay_rejected: true, stale_revision_rejected: true, subject_pinned: true,
    failed_commit_clears_input: true, cross_process_invalidation: true,
    missing_and_corrupt_ledger_rejected: true, execution_receipt_persisted: true,
    modified_output_rejected: true, forged_receipt_history_rejected: true,
    receipt_commit_failure_closed: true, legacy_ledger_migrated: true, reference: [4, 6]}));
} finally {
  for (const child of running) child.kill();
  fs.rmSync(temporary, {recursive: true, force: true});
}
