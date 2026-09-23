import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn, spawnSync} from 'node:child_process';
import {createHash, generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, manifestDigest} from './ingress_provenance.mjs';
import {IngressReplayLedger} from './ingress_replay_ledger.mjs';
import {decodeF32Base64, encodedF32Matches, executionReceipt, f32ValueBytes, sha256Json, f32ValueDigest} from './ingress_execution_receipt.mjs';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const runner = path.join(packageDir, 'interactive_multi_input_ingress.mjs');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'durable-ingress-'));
const ledger = path.join(temporary, 'replay.json');
const trustPath = path.join(temporary, 'trust.json');
const initializer = path.join(packageDir, 'init_ingress_replay_ledger.mjs');
const {privateKey, publicKey} = generateKeyPairSync('ed25519');
fs.writeFileSync(trustPath, JSON.stringify({
  schema: 'burn-research.ingress-trust-policy.v1',
  issuers: ['sensor-a', 'memory-b', 'agent-state'].map(source => ({source, key_id: 'lab-key',
    subjects: ['run-1', 'run-2'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})})),
}));

const running = new Set();
let sequence = 0;
function check(condition, message) {
  if (!condition) throw new Error(message);
}
function start(allowCheckpointExport = true, ledgerPath = ledger) {
  const args = [runner, packageDir, trustPath, ledgerPath, 'run-1'];
  if (allowCheckpointExport) args.push('--allow-state-checkpoint-export');
  const child = spawn(process.execPath, args, {stdio: ['pipe', 'pipe', 'pipe']});
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
  const values = binding.values_f32_le_base64 !== undefined
    ? decodeF32Base64(binding.values_f32_le_base64, binding.shape) : binding.values;
  const claim = canonicalInputClaim({...binding, values}, {
    plan_hex: manifest.plan_hex, manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest), logical_port_id: port.logical_port_id,
  }, 'lab-key', subject, nonce);
  const transport = {...binding};
  if (binding.values_f32_le_base64 !== undefined) delete transport.values;
  return {...transport, proof: {claim, signature: sign(null, Buffer.from(JSON.stringify(claim)), privateKey).toString('base64')}};
}
async function create(client, graphSpec = graph) {
  const response = await client.ask(graphSpec);
  check(response.ok && !response.result.host_provenance.ready, 'graph did not start with missing proofs');
  return response.result.manifest;
}
async function runAndVerify(client, candidate = [4, 6]) {
  const run = await client.ask({op: 'run'});
  check(run.ok && run.result.host_provenance.ready && JSON.stringify(run.result.values) === JSON.stringify(candidate), 'durable graph result mismatch');
  check(run.result.execution_receipt?.authority === 'node_host_observed_wasm_run'
    && run.result.execution_receipt.schema === 'burn-research.host-execution-receipt.v2'
    && run.result.execution_receipt.output.value_sha256 === f32ValueDigest(run.result.values)
    && run.result.execution_receipt.state_checkpoint_bytes_sha256 === run.result.state_checkpoint_bytes_sha256
    && run.result.output_f32_le_base64 === f32ValueBytes(run.result.values).toString('base64'),
  'durable run did not produce an observed output receipt');
  const checkpoint = await client.ask({op: 'checkpoint'});
  const checkpointBytes = Buffer.from(checkpoint.result.bundle_f32le_base64, 'base64');
  check(checkpoint.ok && checkpoint.result.schema === 'burn-research.multi-input-program-bundle.v1'
    && `sha256:${createHash('sha256').update(checkpointBytes).digest('hex')}`
      === checkpoint.result.checkpoint_bytes_sha256
    && checkpoint.result.checkpoint_bytes_sha256 === run.result.state_checkpoint_bytes_sha256,
  'checkpoint operation did not reproduce the receipt-bound state checkpoint');
  const verify = await client.ask({op: 'verify', candidate});
  check(verify.ok && verify.result.reference.verification.passed, 'durable verification failed');
  return {receipt: run.result.execution_receipt, result: run.result};
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
    && capabilities.result.host_provenance.host_subject === 'run-1'
    && capabilities.result.host_provenance.state_checkpoint_export_enabled === true,
  'durable host gate or checkpoint export opt-in did not activate');
  const withoutCheckpoint = start(false);
  check(!(await withoutCheckpoint.ask({op: 'checkpoint'})).ok,
    'state checkpoint export ignored the trusted startup opt-in');
  check(!(await withoutCheckpoint.ask({op: 'restore', receipt_id: 'sha256:' + '0'.repeat(64)})).ok,
    'checkpoint restore ignored the trusted startup opt-in');
  await withoutCheckpoint.close();
  const manifest = await create(first);
  check(!(await first.ask(left)).ok, 'unsigned input accepted');
  check(!(await first.ask(signed(left, manifest, 'wrong-subject', 'run-2'))).ok, 'signed claim bypassed host subject');
  const firstLeft = signed(left, manifest, 'left-1');
  const firstRight = signed(right, manifest, 'right-1');
  check((await first.ask(firstLeft)).ok && (await first.ask(firstRight)).ok, 'initial durable bind rejected');
  const firstRun = await runAndVerify(first);
  const firstReceipt = firstRun.receipt;
  check(firstReceipt.sequence === 1 && firstReceipt.claim_count === 2
    && firstReceipt.manifest_sha256 === manifestDigest(manifest)
    && JSON.stringify(firstReceipt.input_claims) === JSON.stringify([
      {slot: 0, source: 'sensor-a', role: 'observation', shape: [1, 2, 1, 1],
        value_sha256: f32ValueDigest([1, 2]), revision: '2', claim_sha256: sha256Json(firstLeft.proof.claim)},
      {slot: 1, source: 'memory-b', role: 'state', shape: [1, 2, 1, 1],
        value_sha256: f32ValueDigest([3, 4]), revision: '3', claim_sha256: sha256Json(firstRight.proof.claim)},
    ]), 'receipt did not bind both signed input claims');
  const {receipt_id: firstId, ...firstRecord} = firstReceipt;
  check(firstId === sha256Json(firstRecord), 'receipt ID is not its canonical content digest');
  await first.close();

  const second = start();
  const recovered = await second.ask({op: 'receipt', receipt_id: firstId, shape: [1, 2, 1, 1], values: [4, 6]});
  check(recovered.ok && recovered.result.output_matches && recovered.result.receipt.receipt_id === firstId,
    'execution receipt did not survive restart');
  const restored = await second.ask({op: 'restore', receipt_id: firstId});
  check(restored.ok && restored.result.restored_from_receipt_id === firstId
    && restored.result.checkpoint_bytes_sha256 === firstReceipt.state_checkpoint_bytes_sha256
    && JSON.stringify(restored.result.program_identity) === JSON.stringify(firstReceipt.program_identity)
    && !restored.result.host_provenance.ready
    && restored.result.status.graph_preflight.inputs.bound_port_count === 0,
  'receipt-bound restore failed to resume the exact graph without promoting old inputs');
  check(!(await second.ask({op: 'run'})).ok && !(await second.ask(firstLeft)).ok,
    'restored session executed without fresh signed input or replayed a claim');
  check(!(await second.ask({op: 'restore', receipt_id: 'sha256:' + '0'.repeat(64)})).ok
    && (await second.ask({op: 'inspect'})).result.host_provenance.ready === false,
  'unknown receipt altered the live session');
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
  const secondRun = await runAndVerify(second);
  const secondReceipt = secondRun.receipt;
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

  const childGraph = {...graph, layers: [{constructor: 'add', args: [32]}], logicalPorts: [
    {id: 'observation', slot: 0, source: 'sensor-a'},
    {id: 'agent_state', slot: 1, source: 'agent-state'},
  ]};
  const child = start();
  const childManifest = await create(child, childGraph);
  const childObservation = signed({...left, values: [5, 6], revision: 5, fingerprint: 'obs-5'},
    childManifest, 'obs-5');
  const alteredPayload = signed({op: 'bind', slot: 1, values_f32_le_base64: f32ValueBytes([4, 7]).toString('base64'),
    shape: firstReceipt.output.shape, role: 'state', layout: 'feature_axis1_singleton', source: 'agent-state',
    revision: 3, fingerprint: 'state-3', handoff_receipt_id: firstId}, childManifest, 'state-tampered');
  check(!(await child.ask(alteredPayload)).ok, 'state payload different from the parent receipt was accepted');
  check((await child.ask({op: 'inspect'})).result.status.graph_preflight.inputs.bound_port_count === 0,
    'tampered state payload reached the WASM input bundle');
  const childState = signed({op: 'bind', slot: 1,
    values_f32_le_base64: firstRun.result.output_f32_le_base64, shape: firstReceipt.output.shape,
    role: 'state', layout: 'feature_axis1_singleton', source: 'agent-state', revision: 3,
    fingerprint: 'state-3', handoff_receipt_id: firstId}, childManifest, 'state-3');
  check((await child.ask(childObservation)).ok && (await child.ask(childState)).ok,
    'receipt output did not bind as a signed child state input');
  const childRun = await runAndVerify(child, [9, 12]);
  const childReceipt = childRun.receipt;
  const childInput = childReceipt.input_claims.find(input => input.slot === 1);
  check(childReceipt.sequence === 3 && childInput?.handoff_id
    && childInput.role === 'state' && childInput.source === 'agent-state'
    && childInput.value_sha256 === firstReceipt.output.value_sha256,
  'child execution receipt did not preserve parent state lineage');
  const committedLedger = JSON.parse(fs.readFileSync(ledger, 'utf8'));
  const firstHandoff = committedLedger.handoffs[0];
  check(firstHandoff.parent_receipt_id === firstId && firstHandoff.branch_id === 'main'
    && firstHandoff.input.claim_sha256 === sha256Json(childState.proof.claim)
    && firstHandoff.handoff_id === childInput.handoff_id,
  'durable handoff record did not bind the parent receipt and signed child claim');
  await child.close();

  const nextTick = start();
  const laterManifest = await create(nextTick, {...childGraph, layers: [{constructor: 'add', args: [33]}]});
  const nextObservation = signed({...left, values: [2, 3], revision: 6, fingerprint: 'obs-6'},
    laterManifest, 'obs-6');
  check((await nextTick.ask(nextObservation)).ok, 'next tick observation bind rejected');
  const replayedParent = signed({op: 'bind', slot: 1,
    values_f32_le_base64: firstRun.result.output_f32_le_base64, shape: firstReceipt.output.shape,
    role: 'state', layout: 'feature_axis1_singleton', source: 'agent-state', revision: 4,
    fingerprint: 'state-replay', handoff_receipt_id: firstId}, laterManifest, 'state-replay');
  check(!(await nextTick.ask(replayedParent)).ok,
    'an already-consumed parent receipt was replayed on the same branch');
  check((await nextTick.ask({op: 'inspect'})).result.status.graph_preflight.inputs.bound_port_count === 1,
    'rejected replay left a state tensor bound');
  const nextState = signed({op: 'bind', slot: 1,
    values_f32_le_base64: childRun.result.output_f32_le_base64, shape: childReceipt.output.shape,
    role: 'state', layout: 'feature_axis1_singleton', source: 'agent-state', revision: 4,
    fingerprint: 'state-4', handoff_receipt_id: childReceipt.receipt_id}, laterManifest, 'state-4');
  check((await nextTick.ask(nextState)).ok, 'newer parent receipt did not advance the state branch');
  const nextRun = await runAndVerify(nextTick, [11, 15]);
  const nextStateInput = nextRun.receipt.input_claims.find(input => input.slot === 1);
  const finalLedger = JSON.parse(fs.readFileSync(ledger, 'utf8'));
  check(nextRun.receipt.sequence === 4 && finalLedger.handoffs.length === 2
    && finalLedger.handoffs[1].parent_receipt_id === childReceipt.receipt_id
    && nextStateInput?.handoff_id === finalLedger.handoffs[1].handoff_id,
  'two-tick state handoff chain did not persist across a runner restart');
  await nextTick.close();

  const fork = start();
  const forkManifest = await create(fork, {...childGraph, layers: [{constructor: 'add', args: [34]}]});
  const forkObservation = signed({...left, values: [8, 9], revision: 7, fingerprint: 'obs-fork'},
    forkManifest, 'obs-fork');
  const forkState = signed({op: 'bind', slot: 1,
    values_f32_le_base64: firstRun.result.output_f32_le_base64, shape: firstReceipt.output.shape,
    role: 'state', layout: 'feature_axis1_singleton', source: 'agent-state', revision: 5,
    fingerprint: 'state-fork', handoff_receipt_id: firstId, handoff_branch_id: 'experiment'},
  forkManifest, 'state-fork');
  check((await fork.ask(forkObservation)).ok && (await fork.ask(forkState)).ok,
    'independent handoff branch could not fork from the shared parent receipt');
  const forkRun = await runAndVerify(fork, [12, 15]);
  const forkLedger = JSON.parse(fs.readFileSync(ledger, 'utf8'));
  check(forkRun.receipt.sequence === 5 && forkLedger.handoffs.length === 3
    && forkLedger.handoffs[2].parent_receipt_id === firstId
    && forkLedger.handoffs[2].branch_id === 'experiment',
  'explicit branch did not maintain independent receipt replay state');
  await fork.close();

  const mismatch = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-2'], {timeout: 5000});
  check(mismatch.status !== 0 && /host subject/i.test(mismatch.stderr.toString()), 'ledger allowed a different host subject');
  const missingCopy = path.join(temporary, 'removed-ledger.json');
  fs.renameSync(ledger, missingCopy);
  const removed = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(removed.status !== 0, 'removed replay ledger allowed a reset on restart');
  fs.renameSync(missingCopy, ledger);
  const original = fs.readFileSync(ledger);
  const alteredHandoff = JSON.parse(original);
  alteredHandoff.handoffs[0].input.value_sha256 = f32ValueDigest([99, 99]);
  fs.writeFileSync(ledger, JSON.stringify(alteredHandoff) + '\n', {mode: 0o600});
  const forgedHandoff = spawnSync(process.execPath, [runner, packageDir, trustPath, ledger, 'run-1'], {timeout: 5000});
  check(forgedHandoff.status !== 0, 'modified durable state handoff survived ledger history validation');
  fs.writeFileSync(ledger, original, {mode: 0o600});
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
    role: slot === 0 ? 'observation' : 'state', shape: [1, 2, 1, 1],
    value_sha256: f32ValueDigest([4, 6]), revision: '1', manifest_sha256: 'sha256:' + 'a'.repeat(64)}));
  claims.forEach((claim, slot) => migrated.commit({claim,
    nonceKey: JSON.stringify([claim.source, 'key', `nonce-${slot}`]),
    revisionKey: JSON.stringify([claim.source, claim.subject, claim.slot])}));
  const identity = {subject: 'run-1', programIdentity: {schema: 'probe'},
    manifestSha256: claims[0].manifest_sha256};
  const preHandoffFile = path.join(temporary, 'pre-handoff-receipts.json');
  const preHandoff = new IngressReplayLedger(preHandoffFile, 'run-1', {initialize: true});
  claims.forEach((claim, slot) => preHandoff.commit({claim,
    nonceKey: JSON.stringify([claim.source, 'key', `pre-handoff-${slot}`]),
    revisionKey: JSON.stringify([claim.source, claim.subject, claim.slot])}));
  const previousReceipt = preHandoff.executeWithReceipt(claims, identity,
    () => ({shape: [1, 2, 1, 1], values: [4, 6], state_checkpoint_bytes_sha256: 'sha256:' + '1'.repeat(64)})).execution_receipt;
  const previousState = JSON.parse(fs.readFileSync(preHandoffFile, 'utf8'));
  delete previousState.handoffs;
  for (const input of previousState.executions[0].input_claims) {
    delete input.role;
    delete input.shape;
    delete input.value_sha256;
  }
  previousState.executions[0].schema = 'burn-research.host-execution-receipt.v1';
  delete previousState.executions[0].state_checkpoint_bytes_sha256;
  const {receipt_id: previousId, ...previousRecord} = previousState.executions[0];
  previousState.executions[0].receipt_id = sha256Json(previousRecord);
  fs.writeFileSync(preHandoffFile, JSON.stringify(previousState) + '\n', {mode: 0o600});
  const reopenedPrevious = new IngressReplayLedger(preHandoffFile, 'run-1');
  check(previousReceipt.sequence === 1 && reopenedPrevious.getReceipt(previousState.executions[0].receipt_id).receipt_id
    === previousState.executions[0].receipt_id && previousId !== previousState.executions[0].receipt_id,
  'pre-handoff receipt ledger did not reopen under the additive input projection');
  let numericalExecutionOccurred = false;
  const write = migrated.write;
  migrated.write = () => { throw new Error('injected receipt commit failure'); };
  try {
    migrated.executeWithReceipt(claims, identity, () => {
      numericalExecutionOccurred = true;
      return {shape: [1, 2, 1, 1], values: [4, 6], state_checkpoint_bytes_sha256: 'sha256:' + '2'.repeat(64)};
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
    () => ({shape: [1, 2, 1, 1], values: [4, 6], state_checkpoint_bytes_sha256: 'sha256:' + '3'.repeat(64)})).execution_receipt;
  check(migrated.getReceipt(migratedReceipt.receipt_id).sequence === 1,
    'pre-receipt ledger did not preserve its claim history during the upgrade');
  const negativeZero = executionReceipt({...identity, claims, shape: [1, 2, 1, 1], values: [-0, 2],
    sequence: 2, claimCount: 2, checkpointBytesSha256: 'sha256:' + '4'.repeat(64)});
  const negativeZeroBytes = f32ValueBytes([-0, 2]).toString('base64');
  check(f32ValueDigest([-0, 2]) !== f32ValueDigest([0, 2])
    && Object.is(decodeF32Base64(negativeZeroBytes, [1, 2, 1, 1])[0], -0)
    && encodedF32Matches(negativeZero, [1, 2, 1, 1], negativeZeroBytes)
    && !encodedF32Matches(negativeZero, [1, 2, 1, 1], f32ValueBytes([0, 2]).toString('base64')),
  'f32 wire representation erased the sign of negative zero');

  const checkpointFile = path.join(temporary, 'checkpoint-bound-receipts.json');
  const checkpointLedger = new IngressReplayLedger(checkpointFile, 'run-1', {initialize: true});
  claims.forEach((claim, slot) => checkpointLedger.commit({claim,
    nonceKey: JSON.stringify([claim.source, 'key', `checkpoint-${slot}`]),
    revisionKey: JSON.stringify([claim.source, claim.subject, claim.slot])}));
  const sameObservedOutput = () => ({shape: [1, 2, 1, 1], values: [4, 6],
    state_checkpoint_bytes_sha256: 'sha256:' + 'a'.repeat(64)});
  const stateA = checkpointLedger.executeWithReceipt(claims, identity, sameObservedOutput).execution_receipt;
  const stateB = checkpointLedger.executeWithReceipt(claims, identity, () => ({...sameObservedOutput(),
    state_checkpoint_bytes_sha256: 'sha256:' + 'b'.repeat(64)})).execution_receipt;
  check(stateA.output.value_sha256 === stateB.output.value_sha256
    && stateA.state_checkpoint_bytes_sha256 !== stateB.state_checkpoint_bytes_sha256
    && stateA.receipt_id !== stateB.receipt_id,
  'same observed output with different mutable state checkpoint bytes was not distinguished');
  const resumeLedger = path.join(temporary, 'resume-ledger.json');
  check(spawnSync(process.execPath, [initializer, resumeLedger, 'run-1']).status === 0,
    'resume ledger initialization failed');
  const prior = start(true, resumeLedger);
  const priorManifest = await create(prior);
  check((await prior.ask(signed(left, priorManifest, 'resume-left-1'))).ok
    && (await prior.ask(signed(right, priorManifest, 'resume-right-1'))).ok,
  'pre-restart signed input bind failed');
  const priorRun = await runAndVerify(prior);
  await prior.close();
  const storeFile = path.join(`${resumeLedger}.checkpoints`, `${priorRun.receipt.receipt_id.slice(7)}.json`);
  const retained = JSON.parse(fs.readFileSync(storeFile, 'utf8'));
  const retainedBytes = Buffer.from(retained.bundle_f32le_base64, 'base64');
  check(`sha256:${createHash('sha256').update(retainedBytes).digest('hex')}`
    === priorRun.receipt.state_checkpoint_bytes_sha256, 'host failed to retain exact checkpoint bytes');
  const resumed = start(true, resumeLedger);
  const resumedResult = await resumed.ask({op: 'restore', receipt_id: priorRun.receipt.receipt_id});
  check(resumedResult.ok && manifestDigest(resumedResult.result.manifest) === priorRun.receipt.manifest_sha256
    && !(await resumed.ask({op: 'run'})).ok, 'restart restored a runnable old input');
  check((await resumed.ask(signed({...left, revision: 3}, resumedResult.result.manifest, 'resume-left-2'))).ok
    && (await resumed.ask(signed({...right, revision: 4}, resumedResult.result.manifest, 'resume-right-2'))).ok,
  'restored graph did not accept fresh signed inputs');
  const afterResume = await runAndVerify(resumed);
  check(afterResume.receipt.sequence === 2
    && afterResume.receipt.state_checkpoint_bytes_sha256 === priorRun.receipt.state_checkpoint_bytes_sha256,
  'restart did not run the same exact reference state');
  await resumed.close();
  const manifestTamper = {...retained, manifest: {...retained.manifest, execution_authorized: true}};
  fs.writeFileSync(storeFile, JSON.stringify(manifestTamper) + '\n', {mode: 0o600});
  const manifestClient = start(true, resumeLedger);
  check(!(await manifestClient.ask({op: 'restore', receipt_id: priorRun.receipt.receipt_id})).ok,
    'modified retained manifest passed receipt-bound restore');
  await manifestClient.close();
  const tampered = {...retained, bundle_f32le_base64: Buffer.from('forged-state').toString('base64')};
  fs.writeFileSync(storeFile, JSON.stringify(tampered) + '\n', {mode: 0o600});
  const tamperClient = start(true, resumeLedger);
  check(!(await tamperClient.ask({op: 'restore', receipt_id: priorRun.receipt.receipt_id})).ok,
    'modified retained bytes passed receipt-bound restore');
  await tamperClient.close();
  fs.unlinkSync(storeFile);
  const missingClient = start(true, resumeLedger);
  check(!(await missingClient.ask({op: 'restore', receipt_id: priorRun.receipt.receipt_id})).ok,
    'missing retained checkpoint passed restore');
  await missingClient.close();
  const rollbackFile = path.join(temporary, 'checkpoint-rollback-ledger.json');
  const rollbackLedger = new IngressReplayLedger(rollbackFile, 'run-1', {initialize: true});
  const rollbackManifest = {schema: 'test-manifest'};
  const rollbackClaims = claims.map(claim => ({...claim, manifest_sha256: sha256Json(rollbackManifest)}));
  rollbackClaims.forEach((claim, slot) => rollbackLedger.commit({claim,
    nonceKey: JSON.stringify([claim.source, 'key', `rollback-${slot}`]),
    revisionKey: JSON.stringify([claim.source, claim.subject, claim.slot])}));
  const rollbackBytes = Buffer.from('checkpoint-bytes');
  const originalWrite = rollbackLedger.write;
  rollbackLedger.write = () => { throw new Error('injected checkpoint receipt commit failure'); };
  try {
    rollbackLedger.executeWithReceipt(rollbackClaims,
      {...identity, manifestSha256: sha256Json(rollbackManifest)},
      () => ({shape: [1, 2, 1, 1], values: [4, 6],
        state_checkpoint_bytes_sha256: `sha256:${createHash('sha256').update(rollbackBytes).digest('hex')}`,
        checkpoint_bytes: rollbackBytes, checkpoint_manifest: rollbackManifest}));
    throw new Error('failed checkpoint receipt commit returned success');
  } catch (error) {
    check(error.message === 'injected checkpoint receipt commit failure', 'checkpoint commit failure reported wrong error');
  } finally { rollbackLedger.write = originalWrite; }
  check(rollbackLedger.load().state.executions.length === 0
    && fs.readdirSync(`${rollbackFile}.checkpoints`).length === 0,
  'failed receipt commit left an authorized receipt or retained checkpoint');
  const audit = {restart_replay_rejected: true, stale_revision_rejected: true, subject_pinned: true,
    failed_commit_clears_input: true, cross_process_invalidation: true,
    missing_and_corrupt_ledger_rejected: true, execution_receipt_persisted: true,
    modified_output_rejected: true, forged_receipt_history_rejected: true,
    state_handoff_bytes_bound: true, state_handoff_restart_chain: true,
    state_handoff_replay_rejected: true, state_handoff_branching: true,
    forged_handoff_history_rejected: true, pre_handoff_receipts_compatible: true,
    receipt_commit_failure_closed: true, legacy_ledger_migrated: true,
    state_checkpoint_bytes_bound_to_receipt: true, checkpoint_export_matches_run_receipt: true,
    checkpoint_bytes_retained: true, receipt_bound_restore: true,
    exact_restart_resume_requires_new_signed_inputs: true, tampered_or_missing_checkpoint_rejected: true,
    modified_manifest_rejected: true, failed_receipt_commit_cleans_checkpoint: true,
    reference: [4, 6]};
  console.log(JSON.stringify({verdict: 'PASS', mode: 'ed25519_host_durable', ...audit}));
} finally {
  for (const child of running) child.kill();
  fs.rmSync(temporary, {recursive: true, force: true});
}
