import {createHash} from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const {loadBurnRuntime} = await import(pathToFileURL(path.join(packageDir, 'node.mjs')).href);
const wasm = await loadBurnRuntime();
const {IngressReplayLedger} = await import(pathToFileURL(path.join(packageDir, 'ingress_replay_ledger.mjs')).href);
const {BranchPromotionLineage} = await import(pathToFileURL(path.join(packageDir, 'branch_promotion_lineage.mjs')).href);
const {f32ValueDigest, sha256Json} = await import(pathToFileURL(path.join(packageDir, 'ingress_execution_receipt.mjs')).href);

function check(condition, message) {
  if (!condition) throw new Error(message);
}

function sha256Bytes(bytes) {
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

function makeInputs(plan, leftValues = [1, 2], rightValues = [3, 4]) {
  const bundle = new wasm.MultiInputInputBundle(plan);
  const left = new wasm.WasmTensor(new Float32Array(leftValues), new Uint32Array([1, 2, 1, 1]));
  const right = new wasm.WasmTensor(new Float32Array(rightValues), new Uint32Array([1, 2, 1, 1]));
  bundle.bindInput(0, left, 'observation', 'feature_axis1_singleton', 'audit', 0n, '');
  bundle.bindInput(1, right, 'state', 'feature_axis1_singleton', 'audit', 0n, '');
  left.free();
  right.free();
  return bundle;
}

function buildVerifiedBranch(baselineBundle, baselineDigest, branchId, layerId) {
  const liveRegistry = new wasm.LayerRegistry();
  const liveGraph = wasm.importMultiInputProgramBundle(liveRegistry, baselineBundle);
  const identity = liveGraph.programIdentity();
  const branches = new wasm.CheckpointBranchSet(liveRegistry, liveGraph, identity, baselineDigest);
  branches.fork(branchId);
  const relu = wasm.AgentLayerSpec.relu(layerId);
  branches.insertStep(branchId, 1, relu, 2, 2, 3);
  branches.setOutput(branchId, 3);
  branches.stageBranch(branchId);
  const candidateBundle = Buffer.from(branches.candidateBundle(branchId));

  const baselineRegistry = new wasm.LayerRegistry();
  const baselineGraph = wasm.importMultiInputProgramBundle(baselineRegistry, baselineBundle);
  const candidateRegistry = new wasm.LayerRegistry();
  const candidateGraph = wasm.importMultiInputProgramBundle(candidateRegistry, candidateBundle);
  const baselinePlan = wasm.MultiInputGraphPlan.fromBytes(baselineGraph.inputPlanV1());
  const candidatePlan = wasm.MultiInputGraphPlan.fromBytes(candidateGraph.inputPlanV1());
  const cases = new wasm.MultiInputVerificationCases(baselineGraph, candidateGraph);
  const baselineInput = makeInputs(baselinePlan);
  const candidateInput = makeInputs(candidatePlan);
  cases.addCase(baselineInput, candidateInput);
  const branchReceiptJson = branches.verifyBranch(branchId, cases, 0, 0);
  const branchReceipt = JSON.parse(branchReceiptJson);
  check(branchReceipt.equivalent === true && branchReceipt.branch_id === branchId,
    'real WASM branch did not issue a passing branch receipt');

  for (const value of [baselineInput, candidateInput, cases, baselinePlan, candidatePlan,
    baselineGraph, candidateGraph, baselineRegistry, candidateRegistry]) value.free();
  relu.free();
  return {liveRegistry, liveGraph, identity, branches, candidateBundle, branchReceiptJson, branchReceipt};
}

const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'branch-promotion-lineage-'));
fs.chmodSync(temporary, 0o700);
const subject = 'branch-run-1';
const executionPath = path.join(temporary, 'executions.json');
const lineagePath = path.join(temporary, 'promotions.json');
const owned = [];

try {
  const sourceRegistry = new wasm.LayerRegistry();
  const add = wasm.AgentLayerSpec.add(71);
  sourceRegistry.initAgentLayer(add);
  const builder = new wasm.AgentGraphBuilder(4);
  builder.addBinary(add, 0, 1, 2);
  builder.setOutput(2);
  const plan = builder.multiInputPlanV1();
  plan.addInputPort(0, 'observation', 1, 2, 1, 1, 'feature_axis1_singleton', false, 0n);
  plan.addInputPort(1, 'state', 1, 2, 1, 1, 'feature_axis1_singleton', false, 0n);
  const sourceGraph = sourceRegistry.compileMultiInputGraph(plan);
  const baselineBundle = Buffer.from(wasm.exportMultiInputProgramBundle(sourceGraph, sourceRegistry, true));
  const baselineDigest = sha256Bytes(baselineBundle);
  const programIdentity = JSON.parse(sourceGraph.programIdentity());
  owned.push(sourceGraph, plan, builder, add, sourceRegistry);

  const executionLedger = new IngressReplayLedger(executionPath, subject, {initialize: true});
  const manifest = {schema: 'audit-manifest', ports: ['observation', 'state']};
  const manifestSha256 = sha256Json(manifest);

  function claim(slot, source, role, logicalPortId, revision, values) {
    return {
      schema: 'burn-research.signed-input-claim.v1',
      key_id: 'audit-key',
      subject,
      source,
      slot,
      logical_port_id: logicalPortId,
      role,
      shape: [1, 2, 1, 1],
      value_sha256: f32ValueDigest(values),
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

  const observation = claim(0, 'sensor', 'observation', 'observation', 1, [1, 2]);
  const memory = claim(1, 'memory', 'state', 'memory', 1, [3, 4]);
  executionLedger.commit(ticket(observation));
  executionLedger.commit(ticket(memory));
  const baselineRun = executionLedger.executeWithReceipt(
    [observation, memory],
    {subject, programIdentity, manifestSha256},
    () => ({shape: [1, 2, 1, 1], values: [4, 6], state_checkpoint_bytes_sha256: baselineDigest}),
  );
  const baselineReceipt = baselineRun.execution_receipt;
  check(baselineReceipt.state_checkpoint_bytes_sha256 === baselineDigest,
    'durable host baseline receipt differs from real WASM checkpoint');

  const first = buildVerifiedBranch(baselineBundle, baselineDigest, 'candidate-a', 72);
  owned.push(first.branches, first.liveGraph, first.liveRegistry);
  const candidateDigest = sha256Bytes(first.candidateBundle);
  check(candidateDigest === first.branchReceipt.candidate_state_checkpoint_bytes_sha256,
    'real candidate ProgramBundle differs from branch receipt');

  const lineage = new BranchPromotionLineage(lineagePath, subject, executionLedger, {initialize: true});
  const pending = lineage.beginPromotion({
    baselineReceiptId: baselineReceipt.receipt_id,
    branchReceiptJson: first.branchReceiptJson,
    candidateBundle: first.candidateBundle,
  });
  check(pending.status === 'pending' && pending.branch_id === 'candidate-a'
    && pending.baseline_receipt_id === baselineReceipt.receipt_id
    && pending.candidate_checkpoint_bytes_sha256 === candidateDigest,
  'pending promotion intent did not bind the real baseline/branch/candidate');
  check(lineage.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id,
    branchReceiptJson: first.branchReceiptJson, candidateBundle: first.candidateBundle}).intent_id === pending.intent_id,
  'identical beginPromotion was not idempotent');

  const restarted = new BranchPromotionLineage(lineagePath, subject, executionLedger);
  check(restarted.getPromotion(pending.intent_id).completion === null,
    'restart converted pending promotion intent into success');
  let wrongCandidateRejected = false;
  try {
    restarted.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id,
      branchReceiptJson: first.branchReceiptJson, candidateBundle: Buffer.from('wrong')});
  } catch { wrongCandidateRejected = true; }
  check(wrongCandidateRejected, 'candidate bytes not covered by the real branch receipt were accepted');

  let wrongCompletionRejected = false;
  try { restarted.completePromotion(pending.intent_id, Buffer.from('wrong'), first.branchReceipt.candidate_program_identity); }
  catch { wrongCompletionRejected = true; }
  check(wrongCompletionRejected, 'completion accepted the wrong promoted checkpoint bytes');

  const promoted = first.branches.commitBranchByReceipt('candidate-a', first.liveRegistry, first.liveGraph,
    first.identity, baselineDigest, first.branchReceipt.receipt_digest, true);
  owned.push(promoted);
  const promotedBundle = Buffer.from(wasm.exportMultiInputProgramBundle(promoted, first.liveRegistry, true));
  const promotedIdentity = JSON.parse(promoted.programIdentity());
  check(Buffer.compare(promotedBundle, first.candidateBundle) === 0,
    'real WASM promotion did not install the exact verified candidate checkpoint');
  const completed = restarted.completePromotion(pending.intent_id, promotedBundle, promotedIdentity);
  check(completed.status === 'completed' && completed.intent_id === pending.intent_id
    && restarted.completePromotion(pending.intent_id, promotedBundle, promotedIdentity).completion_id === completed.completion_id,
  'promotion completion was not exact or idempotent');
  const completedRestart = new BranchPromotionLineage(lineagePath, subject, executionLedger);
  check(completedRestart.getPromotion(pending.intent_id).completion?.completion_id === completed.completion_id,
    'completed real promotion lineage did not survive restart');

  const second = buildVerifiedBranch(baselineBundle, baselineDigest, 'candidate-pending', 73);
  owned.push(second.branches, second.liveGraph, second.liveRegistry);
  const stillPending = completedRestart.beginPromotion({baselineReceiptId: baselineReceipt.receipt_id,
    branchReceiptJson: second.branchReceiptJson, candidateBundle: second.candidateBundle});
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
  check(tamperRejected, 'tampered exact WASM branch receipt survived lineage reload');
  fs.writeFileSync(lineagePath, beforeTamper, {mode: 0o600});

  console.log(JSON.stringify({
    verdict: 'PASS',
    schema: 'burn-research.host-branch-promotion-lineage.v1',
    baseline_receipt_id: baselineReceipt.receipt_id,
    branch_receipt_digest: first.branchReceipt.receipt_digest,
    candidate_checkpoint_bytes_sha256: candidateDigest,
    real_wasm_commit_by_receipt: true,
    promoted_bundle_exact: Buffer.compare(promotedBundle, first.candidateBundle) === 0,
    pending_survives_restart: true,
    exact_candidate_required: true,
    completion_survives_restart: true,
    duplicate_begin_idempotent: true,
    duplicate_completion_idempotent: true,
    exact_branch_receipt_tamper_rejected: true,
    automatic_promotion: false,
  }));
} finally {
  for (const value of owned.reverse()) {
    try { value.free(); } catch {}
  }
  fs.rmSync(temporary, {recursive: true, force: true});
}
