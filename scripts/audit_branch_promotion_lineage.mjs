import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const {IngressReplayLedger} = await import(pathToFileURL(path.join(packageDir, 'ingress_replay_ledger.mjs')).href);
const {BranchPromotionLineage} = await import(pathToFileURL(path.join(packageDir, 'branch_promotion_lineage.mjs')).href);
const {f32ValueDigest, sha256Json} = await import(pathToFileURL(path.join(packageDir, 'ingress_execution_receipt.mjs')).href);

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256Bytes(bytes) {
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

function exactReceipt(body) {
  const bodyJson = JSON.stringify(body);
  const receiptDigest = sha256Bytes(Buffer.from(bodyJson));
  return JSON.stringify({...body, receipt_digest: receiptDigest});
}

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'branch-promotion-lineage-'));
fs.chmodSync(temporary, 0o700);
const subject = 'branch-run-1';
const executionPath = path.join(temporary, 'executions.json');
const lineagePath = path.join(temporary, 'promotions.json');

try {
  const executionLedger = new IngressReplayLedger(executionPath, subject, {initialize: true});
  const manifest = {schema: 'audit-manifest', ports: ['observation', 'state']};
  const manifestSha256 = sha256Json(manifest);
  const programIdentity = {schema: 'burn-research.multi-input-program-identity.v1', input_plan_hex: '0102', output_slot: 2};

  function claim(slot, source, role, logicalPortId, revision, value) {
    return {
      schema: 'burn-research.signed-input-claim.v1',
      key_id: 'audit-key',
      subject,
      source,
      slot,
      logical_port_id: logicalPortId,
      role,
      shape: [1, 1, 1, 1],
      value_sha256: f32ValueDigest([value]),
      revision: String(revision),
      manifest_sha256: manifestSha256,
      nonce: `${source}-${revision}`,
    };
  }
  function ticket(value) {
    return {
      claim: value,
      nonceKey: JSON.stringify([value.source, value.subject, value.nonce]),
      revisionKey: JSON.stringify([value.source, value.subject, value.slot]),
    };
  }

  const observation = claim(0, 'sensor', 'observation', 'observation', 1, 1);
  const memory = claim(1, 'memory', 'state', 'memory', 1, 2);
  executionLedger.commit(ticket(observation));
  executionLedger.commit(ticket(memory));

  const baselineBundle = Buffer.from('audit-baseline-stateful-program-bundle-v1');
  const baselineDigest = sha256Bytes(baselineBundle);
  const baselineRun = executionLedger.executeWithReceipt(
    [observation, memory],
    {subject, programIdentity, manifestSha256},
    () => ({shape: [1, 1, 1, 1], values: [3], state_checkpoint_bytes_sha256: baselineDigest}),
  );
  const baselineReceipt = baselineRun.execution_receipt;
  check(baselineReceipt.state_checkpoint_bytes_sha256 === baselineDigest, 'baseline receipt digest mismatch');

  const candidateBundle = Buffer.from('audit-candidate-stateful-program-bundle-v2');
  const candidateDigest = sha256Bytes(candidateBundle);
  const candidateIdentity = {schema: 'burn-research.multi-input-program-identity.v1', input_plan_hex: '0103', output_slot: 3};
  const verifierReceipt = JSON.parse(exactReceipt({
    schema_version: 1,
    schema_id: 'burn-research.baseline-candidate-verification.v1',
    authority: 'wasm_burn_reference_observation',
    baseline_program_identity: programIdentity,
    candidate_program_identity: candidateIdentity,
    baseline_state_checkpoint_bytes_sha256: baselineDigest,
    candidate_state_checkpoint_bytes_sha256: candidateDigest,
    abs_tol: 0,
    rel_tol: 0,
    tested_vector_count: 1,
    compared_f32_count: 1,
    max_abs_error: 0,
    max_rel_error: 0,
    first_failure_case_index: null,
    observed_output_bytes: 8,
    equivalent: true,
    promotion_authorized: false,
    cases: [{index: 0, input_sha256: sha256Bytes(Buffer.from('input')), input_bytes: 8,
      baseline_shape: [1, 1, 1, 1], candidate_shape: [1, 1, 1, 1],
      baseline_value_sha256: sha256Bytes(Buffer.from('out')), candidate_value_sha256: sha256Bytes(Buffer.from('out')),
      shape_matches: true, passed: true, max_abs_error: 0, max_rel_error: 0, first_failure: null}],
  }));
  const stateDiff = {
    schema_version: 1,
    schema_id: 'burn-research.checkpoint-state-diff.v1',
    baseline_bundle_sha256: baselineDigest,
    candidate_bundle_sha256: candidateDigest,
    baseline_bytes: baselineBundle.length,
    candidate_bytes: candidateBundle.length,
    byte_length_delta: candidateBundle.length - baselineBundle.length,
    changed_byte_count: 7,
    same_length: baselineBundle.length === candidateBundle.length,
    ranges: [],
    ranges_truncated: false,
    semantic_equivalence_asserted: false,
  };
  const branchReceiptJson = exactReceipt({
    schema_version: 1,
    schema_id: 'burn-research.checkpoint-branch-verification.v1',
    authority: 'wasm_burn_reference_observation',
    branch_id: 'candidate-a',
    baseline_program_identity: programIdentity,
    candidate_program_identity: candidateIdentity,
    baseline_state_checkpoint_bytes_sha256: baselineDigest,
    candidate_state_checkpoint_bytes_sha256: candidateDigest,
    equivalent: true,
    promotion_authorized: false,
    verifier_receipt: verifierReceipt,
    state_diff: stateDiff,
  });

  const lineage = new BranchPromotionLineage(lineagePath, subject, executionLedger, {initialize: true});
  const pending = lineage.beginPromotion({
    baselineReceiptId: baselineReceipt.receipt_id,
    branchReceiptJson,
    candidateBundle,
  });
  check(pending.status === 'pending' && pending.branch_id === 'candidate-a'
    && pending.baseline_receipt_id === baselineReceipt.receipt_id
    && pending.candidate_checkpoint_bytes_sha256 === candidateDigest,
  'pending branch promotion intent did not bind baseline/branch/candidate');
  check(lineage.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id, branchReceiptJson, candidateBundle}).intent_id
    === pending.intent_id, 'identical beginPromotion was not idempotent');

  const restarted = new BranchPromotionLineage(lineagePath, subject, executionLedger);
  check(restarted.getPromotion(pending.intent_id).completion === null, 'restart converted pending intent into success');
  let wrongCandidateRejected = false;
  try {
    restarted.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id, branchReceiptJson, candidateBundle: Buffer.from('wrong')});
  } catch { wrongCandidateRejected = true; }
  check(wrongCandidateRejected, 'candidate bytes not covered by branch receipt were accepted');

  let wrongCompletionRejected = false;
  try { restarted.completePromotion(pending.intent_id, Buffer.from('wrong'), candidateIdentity); }
  catch { wrongCompletionRejected = true; }
  check(wrongCompletionRejected, 'completion accepted the wrong promoted checkpoint bytes');

  const completed = restarted.completePromotion(pending.intent_id, candidateBundle, candidateIdentity);
  check(completed.status === 'completed' && completed.intent_id === pending.intent_id
    && restarted.completePromotion(pending.intent_id, candidateBundle, candidateIdentity).completion_id === completed.completion_id,
  'promotion completion was not exact or idempotent');
  const finalRestart = new BranchPromotionLineage(lineagePath, subject, executionLedger);
  check(finalRestart.getPromotion(pending.intent_id).completion?.completion_id === completed.completion_id,
    'completed promotion lineage did not survive restart');

  const secondCandidate = Buffer.from('audit-candidate-pending-after-crash');
  const secondDigest = sha256Bytes(secondCandidate);
  const secondIdentity = {...candidateIdentity, input_plan_hex: '0104'};
  const secondVerifier = {...verifierReceipt,
    candidate_program_identity: secondIdentity,
    candidate_state_checkpoint_bytes_sha256: secondDigest};
  const secondVerifierBody = {...secondVerifier}; delete secondVerifierBody.receipt_digest;
  secondVerifier.receipt_digest = sha256Bytes(Buffer.from(JSON.stringify(secondVerifierBody)));
  const secondBranchJson = exactReceipt({
    schema_version: 1,
    schema_id: 'burn-research.checkpoint-branch-verification.v1',
    authority: 'wasm_burn_reference_observation',
    branch_id: 'candidate-pending',
    baseline_program_identity: programIdentity,
    candidate_program_identity: secondIdentity,
    baseline_state_checkpoint_bytes_sha256: baselineDigest,
    candidate_state_checkpoint_bytes_sha256: secondDigest,
    equivalent: true,
    promotion_authorized: false,
    verifier_receipt: secondVerifier,
    state_diff: {...stateDiff, candidate_bundle_sha256: secondDigest},
  });
  const stillPending = finalRestart.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id,
    branchReceiptJson: secondBranchJson, candidateBundle: secondCandidate});
  const afterCrash = new BranchPromotionLineage(lineagePath, subject, executionLedger);
  check(afterCrash.getPromotion(stillPending.intent_id).completion === null,
    'pending promotion intent was inferred completed after restart');

  const beforeTamper = fs.readFileSync(lineagePath);
  const tampered = JSON.parse(beforeTamper);
  tampered.intents[0].branch_receipt_json = tampered.intents[0].branch_receipt_json.replace('candidate-a', 'candidate-x');
  tampered.intents[0].branch_receipt_sha256 = sha256Bytes(Buffer.from(tampered.intents[0].branch_receipt_json));
  const {intent_id: ignored, ...tamperedRecord} = tampered.intents[0];
  tampered.intents[0].intent_id = sha256Json(tamperedRecord);
  fs.writeFileSync(lineagePath, `${JSON.stringify(tampered)}\n`, {mode: 0o600});
  let tamperRejected = false;
  try { new BranchPromotionLineage(lineagePath, subject, executionLedger); } catch { tamperRejected = true; }
  check(tamperRejected, 'tampered exact branch receipt survived lineage reload');
  fs.writeFileSync(lineagePath, beforeTamper, {mode: 0o600});

  console.log(JSON.stringify({
    verdict: 'PASS',
    schema: 'burn-research.host-branch-promotion-lineage.v1',
    pending_survives_restart: true,
    exact_candidate_required: true,
    completion_survives_restart: true,
    duplicate_begin_idempotent: true,
    duplicate_completion_idempotent: true,
    exact_branch_receipt_tamper_rejected: true,
    automatic_promotion: false,
  }));
} finally {
  fs.rmSync(temporary, {recursive: true, force: true});
}
