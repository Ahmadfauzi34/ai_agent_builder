import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn, spawnSync} from 'node:child_process';
import {createHash, generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, canonicalStateBoundInputClaim, manifestDigest,
  SIGNED_INPUT_CLAIM_SCHEMA_V1, SIGNED_INPUT_CLAIM_SCHEMA_V2} from './ingress_provenance.mjs';
import {decodeF32Base64, sha256Json} from './ingress_execution_receipt.mjs';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const runner = path.join(packageDir, 'interactive_multi_input_ingress.mjs');
const initializer = path.join(packageDir, 'init_ingress_replay_ledger.mjs');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'state-bound-ingress-'));
const artifactArchive = process.argv[3] ? path.resolve(process.argv[3]) : null;
const artifactSha256 = artifactArchive
  ? createHash('sha256').update(fs.readFileSync(artifactArchive)).digest('hex') : null;
const subject = 'state-bound-audit';
const {privateKey, publicKey} = generateKeyPairSync('ed25519');
const children = new Set();
let requestSequence = 0;

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function writePolicy(name, sources) {
  const file = path.join(temporary, `${name}.json`);
  fs.writeFileSync(file, JSON.stringify({schema: 'burn-research.ingress-trust-policy.v1',
    issuers: sources.map(({source, schemas}) => ({source, key_id: 'audit-key', subjects: [subject],
      ...(schemas ? {claim_schemas: schemas} : {}),
      public_key_pem: publicKey.export({type: 'spki', format: 'pem'})}))}));
  return file;
}

function initLedger(name) {
  const file = path.join(temporary, `${name}.ledger.json`);
  const result = spawnSync(process.execPath, [initializer, file, subject], {timeout: 5000});
  check(result.status === 0, `ledger initialization failed: ${result.stderr}`);
  return file;
}

function start(policy, ledger, flags = []) {
  const child = spawn(process.execPath,
    [runner, packageDir, policy, ledger ?? '', ledger ? subject : '', ...flags],
    {stdio: ['pipe', 'pipe', 'pipe']});
  children.add(child);
  let stderr = '';
  child.stderr.on('data', chunk => { stderr += chunk; });
  const pending = [];
  readline.createInterface({input: child.stdout}).on('line', line => pending.shift()?.(JSON.parse(line)));
  return {
    ask(command) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error(`runner response timeout: ${stderr}`)), 10000);
        pending.push(response => { clearTimeout(timer); resolve(response); });
        child.stdin.write(JSON.stringify({...command, request_id: ++requestSequence}) + '\n');
      });
    },
    async close() {
      const result = await this.ask({op: 'close'});
      check(result.ok && result.result.closed, `runner did not close cleanly: ${JSON.stringify(result)} ${stderr}`);
      const code = await new Promise(resolve => child.once('close', resolve));
      children.delete(child);
      check(code === 0, `runner exited with ${code}: ${stderr}`);
    },
  };
}

function graph() {
  return {op: 'create', numSlots: 4, outputSlot: 3,
    layers: [{constructor: 'add', args: [31]}, {constructor: 'linear', args: [32, 2, 2, true]}],
    steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}, {kind: 'unary', layer: 1, slots: [2, 3]}],
    ports: [
      {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', minimumRevision: 2},
      {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', minimumRevision: 3},
    ],
    logicalPorts: [{id: 'observation', slot: 0, source: 'sensor-a'}, {id: 'memory', slot: 1, source: 'memory-b'}]};
}

function inputs(leftRevision, rightRevision) {
  return [
    {op: 'bind', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation',
      layout: 'feature_axis1_singleton', source: 'sensor-a', revision: leftRevision, fingerprint: `obs-${leftRevision}`},
    {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state',
      layout: 'feature_axis1_singleton', source: 'memory-b', revision: rightRevision, fingerprint: `state-${rightRevision}`},
  ];
}

function signed(binding, manifest, nonce, schema, activeStateDigest) {
  const port = manifest.ports.find(item => item.slot === binding.slot);
  const values = binding.values_f32_le_base64 === undefined
    ? binding.values : decodeF32Base64(binding.values_f32_le_base64, binding.shape);
  const context = {plan_hex: manifest.plan_hex, manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest), logical_port_id: port.logical_port_id,
    ...(schema === SIGNED_INPUT_CLAIM_SCHEMA_V2
      ? {active_state_checkpoint_bytes_sha256: activeStateDigest} : {})};
  const claim = schema === SIGNED_INPUT_CLAIM_SCHEMA_V2
    ? canonicalStateBoundInputClaim({...binding, values}, context, 'audit-key', subject, nonce)
    : canonicalInputClaim({...binding, values}, context, 'audit-key', subject, nonce);
  return {...binding, proof: {claim,
    signature: sign(null, Buffer.from(JSON.stringify(claim)), privateKey).toString('base64')}};
}

async function create(client) {
  const result = await client.ask(graph());
  check(result.ok && !result.result.host_provenance.ready, `graph creation failed: ${JSON.stringify(result)}`);
  return result.result;
}

async function bindAll(client, manifest, stateDigest, revisions, noncePrefix,
  schema = SIGNED_INPUT_CLAIM_SCHEMA_V2) {
  const commands = inputs(...revisions).map((binding, index) =>
    signed(binding, manifest, `${noncePrefix}-${index}`, schema, stateDigest));
  for (const command of commands) {
    const result = await client.ask(command);
    check(result.ok, `signed bind rejected: ${JSON.stringify(result)}`);
  }
  return commands;
}

async function run(client) {
  const result = await client.ask({op: 'run'});
  check(result.ok && result.result.host_provenance.ready && Array.isArray(result.result.values),
    `graph run failed: ${JSON.stringify(result)}`);
  return result.result;
}

const schemas = [SIGNED_INPUT_CLAIM_SCHEMA_V1, SIGNED_INPUT_CLAIM_SCHEMA_V2];
const v2Policy = writePolicy('v2-policy', [
  {source: 'sensor-a', schemas}, {source: 'memory-b', schemas},
]);
const v1DefaultPolicy = writePolicy('v1-default-policy', [
  {source: 'sensor-a'}, {source: 'memory-b'},
]);
const mixedPolicy = writePolicy('mixed-policy', [
  {source: 'sensor-a', schemas}, {source: 'memory-b', schemas: [SIGNED_INPUT_CLAIM_SCHEMA_V1]},
]);
const report = {schema: 'burn-research.state-bound-ingress-audit.v1', scenarios: {}};

try {
  // Fixed graph and f32 inputs; only independent mutable layer initialization differs.
  const ledgerA = initLedger('host-a');
  const ledgerB = initLedger('host-b');
  const hostA = start(v2Policy, ledgerA, ['--allow-checkpoint-restore']);
  const hostB = start(v2Policy, ledgerB, ['--allow-checkpoint-restore']);
  const createdA = await create(hostA);
  const createdB = await create(hostB);
  const digestA = createdA.host_provenance.active_state_checkpoint_bytes_sha256;
  const digestB = createdB.host_provenance.active_state_checkpoint_bytes_sha256;
  check(/^sha256:[0-9a-f]{64}$/.test(digestA ?? '') && /^sha256:[0-9a-f]{64}$/.test(digestB ?? ''),
    'v2-capable host did not expose a valid active state digest');
  check(JSON.stringify(createdA.manifest) === JSON.stringify(createdB.manifest),
    'the fixed structural graph produced different manifests');
  check(JSON.stringify(createdA.program_identity) === JSON.stringify(createdB.program_identity),
    'negative control changed structural program identity instead of mutable state');
  check(digestA !== digestB, 'negative control did not create distinct active model state');

  const claimsA = inputs(2, 3).map((binding, index) =>
    signed(binding, createdA.manifest, `host-a-${index}`, SIGNED_INPUT_CLAIM_SCHEMA_V2, digestA));
  const bindA0 = await hostA.ask(claimsA[0]);
  check(bindA0.ok, `state-matched v2 input was rejected: ${JSON.stringify(bindA0)}`);
  const crossState = await hostB.ask(claimsA[0]);
  check(!crossState.ok && /checkpoint digest differs from the active program state/i.test(crossState.error),
    `state-mismatched proof did not fail at the state gate: ${JSON.stringify(crossState)}`);
  const untouchedB = await hostB.ask({op: 'inspect'});
  check(untouchedB.ok && untouchedB.result.status.graph_preflight.inputs.bound_port_count === 0,
    'rejected cross-state proof mutated the peer input bundle');
  const bindA1 = await hostA.ask(claimsA[1]);
  check(bindA1.ok, `second state-matched v2 input was rejected: ${JSON.stringify(bindA1)}`);
  const runA = await run(hostA);
  const receiptA = runA.execution_receipt;
  check(receiptA?.input_state_checkpoint_bytes_sha256 === digestA
    && receiptA.input_claims.every(input => input.claim_schema === SIGNED_INPUT_CLAIM_SCHEMA_V2
      && input.active_state_checkpoint_bytes_sha256 === digestA)
    && /^sha256:[0-9a-f]{64}$/.test(receiptA.state_checkpoint_bytes_sha256),
  'durable receipt did not distinguish the signed pre-run digest from its post-run checkpoint');
  const repeatedRunA = await run(hostA);
  check(repeatedRunA.execution_receipt.sequence === receiptA.sequence + 1
    && repeatedRunA.execution_receipt.state_parent_receipt_id === receiptA.receipt_id
    && repeatedRunA.execution_receipt.state_checkpoint_bytes_sha256 === receiptA.state_checkpoint_bytes_sha256
    && repeatedRunA.execution_receipt.input_state_checkpoint_bytes_sha256 === digestA,
  'unchanged-state repeated run did not preserve the expected state-bound execution behavior');
  const tamperedLedger = path.join(temporary, 'tampered-v2-receipt.ledger.json');
  fs.copyFileSync(ledgerA, tamperedLedger);
  fs.chmodSync(tamperedLedger, 0o600);
  const tamperedState = JSON.parse(fs.readFileSync(tamperedLedger, 'utf8'));
  const {receipt_id: ignoredReceiptId, ...tamperedReceipt} = tamperedState.executions[0];
  tamperedReceipt.input_state_checkpoint_bytes_sha256 = `sha256:${'0'.repeat(64)}`;
  tamperedState.executions[0] = {...tamperedReceipt, receipt_id: sha256Json(tamperedReceipt)};
  fs.writeFileSync(tamperedLedger, `${JSON.stringify(tamperedState)}\n`, {mode: 0o600});
  fs.chmodSync(tamperedLedger, 0o600);
  const tamperProbe = spawnSync(process.execPath,
    [runner, packageDir, v2Policy, tamperedLedger, subject], {timeout: 10000});
  check(tamperProbe.status !== 0 && /common input checkpoint digest is inconsistent/i.test(tamperProbe.stderr),
    'ledger accepted a rehashed receipt with a mismatched common input checkpoint digest');

  // Reinitialize the same program on host A, then restore A's earlier receipt.
  const createdA2 = await create(hostA);
  const digestA2 = createdA2.host_provenance.active_state_checkpoint_bytes_sha256;
  check(digestA2 !== receiptA.state_checkpoint_bytes_sha256,
    'restore freshness fixture did not produce a checkpoint distinct from the later session');
  const claimsA2 = await bindAll(hostA, createdA2.manifest, digestA2, [3, 4], 'host-a-later');
  const runA2 = await run(hostA);
  const receiptA2 = runA2.execution_receipt;
  const restored = await hostA.ask({op: 'restore', receipt_id: receiptA.receipt_id});
  check(restored.ok && restored.result.host_provenance.active_state_checkpoint_bytes_sha256
    === receiptA.state_checkpoint_bytes_sha256,
  `receipt restore did not reestablish the recorded state digest: ${JSON.stringify(restored)}`);
  const staleAfterRestore = await hostA.ask(claimsA2[0]);
  check(!staleAfterRestore.ok && /checkpoint digest differs from the active program state/i.test(staleAfterRestore.error),
    `pre-restore signed input was not rejected against restored state: ${JSON.stringify(staleAfterRestore)}`);
  const afterReject = await hostA.ask({op: 'inspect'});
  check(afterReject.ok && afterReject.result.status.graph_preflight.inputs.bound_port_count === 0,
    'stale post-restore claim reached the restored input bundle');
  const claimsRestored = await bindAll(hostA, restored.result.manifest,
    restored.result.host_provenance.active_state_checkpoint_bytes_sha256, [5, 6], 'host-a-restored');
  const runRestored = await run(hostA);
  check(runRestored.execution_receipt.state_parent_receipt_id === receiptA.receipt_id
    && runRestored.execution_receipt.restore_event_id === restored.result.restore_event_id
    && runRestored.execution_receipt.input_state_checkpoint_bytes_sha256
      === restored.result.host_provenance.active_state_checkpoint_bytes_sha256,
  'restored execution receipt did not bind the restored input state and ancestry');
  report.scenarios.state_binding_and_restore = {
    verdict: 'PASS_WITH_PROVEN_LIMITS',
    fixed_manifest_equal_across_peers: true,
    structural_program_identity_equal_across_peers: true,
    state_digests_distinct: [digestA, digestB],
    exact_signed_v2_proof_rejected_on_other_state: true,
    rejected_bind_left_input_count: 0,
    receipt_input_digest: receiptA.input_state_checkpoint_bytes_sha256,
    receipt_post_run_digest: receiptA.state_checkpoint_bytes_sha256,
    same_claim_set_can_run_again_while_checkpoint_is_unchanged: true,
    repeated_run_sequence: repeatedRunA.execution_receipt.sequence,
    rehashed_receipt_with_mismatched_input_digest_rejected: true,
    pre_restore_claim_rejected_after_restore: true,
    restored_state_digest: restored.result.host_provenance.active_state_checkpoint_bytes_sha256,
    restored_receipt_parent: runRestored.execution_receipt.state_parent_receipt_id,
    restored_receipt_event: runRestored.execution_receipt.restore_event_id,
    stale_claims: claimsA2.length,
    post_restore_claims: claimsRestored.length,
    interpretation: 'v2 binds each accepted input to the exact active state bytes; restore requires fresh signatures for the restored digest',
  };

  const wrongIssuerHost = start(mixedPolicy, null);
  const wrongIssuerCreated = await create(wrongIssuerHost);
  const wrongIssuerBinding = inputs(2, 3)[1];
  const wrongIssuerClaim = signed(wrongIssuerBinding, wrongIssuerCreated.manifest, 'v2-not-allowed',
    SIGNED_INPUT_CLAIM_SCHEMA_V2, wrongIssuerCreated.host_provenance.active_state_checkpoint_bytes_sha256);
  const wrongIssuer = await wrongIssuerHost.ask(wrongIssuerClaim);
  check(!wrongIssuer.ok && /issuer policy does not allow/i.test(wrongIssuer.error),
    `issuer without v2 opt-in accepted the claim: ${JSON.stringify(wrongIssuer)}`);
  const mixedRequired = spawnSync(process.execPath,
    [runner, packageDir, mixedPolicy, '', '', '--require-state-bound-inputs'], {timeout: 10000});
  check(mixedRequired.status !== 0 && /every configured issuer.*allow signed claim v2/i.test(mixedRequired.stderr),
    'required state-bound startup did not fail fast for a v1-only configured issuer');
  report.scenarios.issuer_allowlist = {
    verdict: 'PASS', v2_from_v1_only_issuer_rejected: true,
    required_mode_rejects_mixed_issuer_policy_at_startup: true,
    expected_rejector: 'issuer claim_schemas allowlist',
  };

  const requiredLedger = initLedger('required-host');
  const requiredHost = start(v2Policy, requiredLedger,
    ['--require-state-bound-inputs', '--allow-checkpoint-restore']);
  const requiredCreated = await create(requiredHost);
  const v1Attempt = await requiredHost.ask(signed(inputs(2, 3)[0], requiredCreated.manifest,
    'required-v1', SIGNED_INPUT_CLAIM_SCHEMA_V1));
  check(!v1Attempt.ok && /requires every signed input to use state-bound claim v2/i.test(v1Attempt.error),
    `required-state host accepted or misclassified v1: ${JSON.stringify(v1Attempt)}`);
  const requiredUntouched = await requiredHost.ask({op: 'inspect'});
  check(requiredUntouched.ok && requiredUntouched.result.status.graph_preflight.inputs.bound_port_count === 0,
    'required-mode v1 rejection mutated the input bundle');
  await bindAll(requiredHost, requiredCreated.manifest,
    requiredCreated.host_provenance.active_state_checkpoint_bytes_sha256, [2, 3], 'required-v2');
  const requiredRun = await run(requiredHost);
  check(requiredRun.execution_receipt.input_state_checkpoint_bytes_sha256
    === requiredCreated.host_provenance.active_state_checkpoint_bytes_sha256,
  'required-state host did not execute a complete v2 input set');
  report.scenarios.required_gate = {
    verdict: 'PASS', v1_rejected_before_bind: true, v2_complete_input_set_accepted: true,
    input_count_after_reject: 0, required_receipt_input_digest: requiredRun.execution_receipt.input_state_checkpoint_bytes_sha256,
  };

  const legacyHost = start(v1DefaultPolicy, null);
  const legacyCreated = await create(legacyHost);
  const legacyBindings = inputs(2, 3);
  for (const [index, binding] of legacyBindings.entries()) {
    const result = await legacyHost.ask(signed(binding, legacyCreated.manifest, `legacy-${index}`,
      SIGNED_INPUT_CLAIM_SCHEMA_V1));
    check(result.ok, `legacy v1-only trust policy regressed: ${JSON.stringify(result)}`);
  }
  const legacyRun = await run(legacyHost);
  check(legacyRun.execution_receipt === undefined && legacyRun.host_provenance.ready,
    'legacy non-durable v1 mode changed its receipt or readiness behavior');
  report.scenarios.v1_compatibility = {
    verdict: 'PASS', missing_claim_schemas_defaults_to_v1: true,
    v1_only_host_runs: true, durable_receipt_absent: true,
  };

  await legacyHost.close();
  await requiredHost.close();
  await wrongIssuerHost.close();
  await hostB.close();
  await hostA.close();
  report.verdict = 'PASS_WITH_PROVEN_LIMITS';
  report.baseline = {artifactArchiveSha256: artifactSha256,
    state_digest_algorithm: 'sha256 over exact stateful multi-input ProgramBundle bytes',
    run_receipt_count_for_state_restore: 4};
  console.log(JSON.stringify(report, null, 2));
} finally {
  for (const child of children) child.kill();
  fs.rmSync(temporary, {recursive: true, force: true});
}
