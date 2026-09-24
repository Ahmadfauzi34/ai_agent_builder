import {createHash, randomUUID} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {EXECUTION_RECEIPT_SCHEMA, sha256Json} from './ingress_execution_receipt.mjs';

const LEDGER_SCHEMA = 'burn-research.host-branch-promotion-lineage-ledger.v1';
const INTENT_SCHEMA = 'burn-research.host-branch-promotion-intent.v1';
const COMPLETION_SCHEMA = 'burn-research.host-branch-promotion-completion.v1';
const BRANCH_RECEIPT_SCHEMA = 'burn-research.checkpoint-branch-verification.v1';
const BRANCH_AUTHORITY = 'wasm_burn_reference_observation';
const MAX_PROMOTIONS = 4096;
const MAX_LEDGER_BYTES = 32 * 1024 * 1024;
const MAX_BRANCH_RECEIPT_BYTES = 1024 * 1024;
const MAX_CANDIDATE_BYTES = 16 * 1024 * 1024;
const SHA256 = /^sha256:[0-9a-f]{64}$/;

function sha256Bytes(bytes) {
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
}

function validBranchId(value) {
  return typeof value === 'string' && value.length >= 1 && value.length <= 64
    && /^[A-Za-z0-9._-]+$/.test(value);
}

function validateOwner(stat, description) {
  if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) {
    throw new Error(`${description} must be owned by the runner user`);
  }
}

function reconstructBranchReceiptBody(raw, receiptDigest) {
  const suffix = `,"receipt_digest":"${receiptDigest}"}`;
  if (!raw.endsWith(suffix)) {
    throw new Error('branch promotion lineage requires the exact compact WASM branch receipt bytes');
  }
  return `${raw.slice(0, -suffix.length)}}`;
}

function validateBranchReceipt(raw, baselineReceipt, candidateBundle) {
  if (typeof raw !== 'string' || raw.length === 0 || Buffer.byteLength(raw) > MAX_BRANCH_RECEIPT_BYTES) {
    throw new Error('branch promotion receipt is missing or exceeds 1 MiB');
  }
  let receipt;
  try { receipt = JSON.parse(raw); } catch { throw new Error('branch promotion receipt is not valid JSON'); }
  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)
    || receipt.schema_id !== BRANCH_RECEIPT_SCHEMA
    || receipt.authority !== BRANCH_AUTHORITY
    || !validBranchId(receipt.branch_id)
    || receipt.equivalent !== true
    || receipt.promotion_authorized !== false
    || !SHA256.test(receipt.receipt_digest ?? '')
    || !SHA256.test(receipt.baseline_state_checkpoint_bytes_sha256 ?? '')
    || !SHA256.test(receipt.candidate_state_checkpoint_bytes_sha256 ?? '')
    || !receipt.baseline_program_identity || typeof receipt.baseline_program_identity !== 'object'
    || Array.isArray(receipt.baseline_program_identity)
    || !receipt.candidate_program_identity || typeof receipt.candidate_program_identity !== 'object'
    || Array.isArray(receipt.candidate_program_identity)
    || !receipt.verifier_receipt || typeof receipt.verifier_receipt !== 'object'
    || receipt.verifier_receipt.schema_id !== 'burn-research.baseline-candidate-verification.v1'
    || receipt.verifier_receipt.authority !== BRANCH_AUTHORITY
    || receipt.verifier_receipt.equivalent !== true
    || receipt.verifier_receipt.promotion_authorized !== false
    || !SHA256.test(receipt.verifier_receipt.receipt_digest ?? '')
    || !receipt.state_diff || typeof receipt.state_diff !== 'object'
    || receipt.state_diff.semantic_equivalence_asserted !== false
    || !SHA256.test(receipt.state_diff.baseline_bundle_sha256 ?? '')
    || !SHA256.test(receipt.state_diff.candidate_bundle_sha256 ?? '')) {
    throw new Error('branch promotion receipt does not satisfy the verified branch contract');
  }
  const body = reconstructBranchReceiptBody(raw, receipt.receipt_digest);
  if (sha256Bytes(Buffer.from(body)) !== receipt.receipt_digest) {
    throw new Error('branch promotion receipt digest does not match its exact WASM body bytes');
  }
  if (!baselineReceipt || baselineReceipt.schema !== EXECUTION_RECEIPT_SCHEMA
    || baselineReceipt.state_checkpoint_bytes_sha256 !== receipt.baseline_state_checkpoint_bytes_sha256
    || JSON.stringify(baselineReceipt.program_identity) !== JSON.stringify(receipt.baseline_program_identity)
    || receipt.state_diff.baseline_bundle_sha256 !== receipt.baseline_state_checkpoint_bytes_sha256
    || receipt.state_diff.candidate_bundle_sha256 !== receipt.candidate_state_checkpoint_bytes_sha256) {
    throw new Error('branch promotion receipt baseline differs from its committed host execution receipt');
  }
  if (!Buffer.isBuffer(candidateBundle) || candidateBundle.length === 0 || candidateBundle.length > MAX_CANDIDATE_BYTES) {
    throw new Error('branch promotion candidate bundle must be 1..=16 MiB');
  }
  const candidateDigest = sha256Bytes(candidateBundle);
  if (candidateDigest !== receipt.candidate_state_checkpoint_bytes_sha256) {
    throw new Error('branch promotion candidate bytes differ from the branch receipt');
  }
  return {receipt, candidateDigest};
}

export class BranchPromotionLineage {
  constructor(lineagePath, subject, executionLedger, {initialize = false} = {}) {
    if (typeof subject !== 'string' || subject.length === 0 || Buffer.byteLength(subject) > 256) {
      throw new Error('branch promotion host subject must be an explicit nonempty string');
    }
    if (!executionLedger || typeof executionLedger.getReceipt !== 'function') {
      throw new Error('branch promotion lineage requires the durable execution ledger');
    }
    this.subject = subject;
    this.executionLedger = executionLedger;
    this.file = path.resolve(lineagePath);
    this.directory = path.dirname(this.file);
    this.lockPath = `${this.file}.lock`;
    const directory = fs.lstatSync(this.directory);
    if (!directory.isDirectory() || directory.isSymbolicLink() || directory.mode & 0o022) {
      throw new Error('branch promotion ledger parent must be a private host-owned directory');
    }
    validateOwner(directory, 'branch promotion ledger parent');
    this.withLock(() => {
      if (initialize) {
        if (fs.existsSync(this.file)) throw new Error('refusing to reset existing branch promotion lineage');
        this.write({schema: LEDGER_SCHEMA, subject: this.subject, intents: [], completions: []});
      } else this.load();
    });
  }

  withLock(operation) {
    fs.mkdirSync(this.lockPath, {mode: 0o700});
    try { return operation(); } finally { fs.rmdirSync(this.lockPath); }
  }

  write(state) {
    const contents = `${JSON.stringify(state)}\n`;
    if (Buffer.byteLength(contents) > MAX_LEDGER_BYTES) {
      throw new Error('branch promotion lineage exceeds its 32 MiB limit');
    }
    const temporary = path.join(this.directory, `.${path.basename(this.file)}.${randomUUID()}.tmp`);
    let fd;
    try {
      fd = fs.openSync(temporary, 'wx', 0o600);
      fs.writeFileSync(fd, contents);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
      fd = undefined;
      fs.renameSync(temporary, this.file);
      const dirFd = fs.openSync(this.directory, 'r');
      try { fs.fsyncSync(dirFd); } finally { fs.closeSync(dirFd); }
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
  }

  load() {
    const stat = fs.lstatSync(this.file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.mode & 0o077 || stat.size > MAX_LEDGER_BYTES) {
      throw new Error('branch promotion lineage file is not a private bounded regular file');
    }
    validateOwner(stat, 'branch promotion lineage file');
    const state = JSON.parse(fs.readFileSync(this.file, 'utf8'));
    if (state.schema !== LEDGER_SCHEMA || state.subject !== this.subject
      || !Array.isArray(state.intents) || state.intents.length > MAX_PROMOTIONS
      || !Array.isArray(state.completions) || state.completions.length > MAX_PROMOTIONS) {
      throw new Error('branch promotion lineage schema, subject, or event limit mismatch');
    }
    const intentsById = new Map();
    for (let index = 0; index < state.intents.length; index++) {
      const intent = state.intents[index];
      if (!intent || typeof intent !== 'object' || Array.isArray(intent)) throw new Error('malformed branch promotion intent');
      const {intent_id: intentId, ...record} = intent;
      const baseline = this.executionLedger.getReceipt(intent.baseline_receipt_id);
      if (intent.schema !== INTENT_SCHEMA || intent.subject !== this.subject || intent.sequence !== index + 1
        || intent.status !== 'pending' || intentId !== sha256Json(record)
        || !baseline || baseline.schema !== EXECUTION_RECEIPT_SCHEMA
        || baseline.sequence !== intent.baseline_execution_sequence
        || baseline.state_checkpoint_bytes_sha256 !== intent.baseline_checkpoint_bytes_sha256
        || !validBranchId(intent.branch_id)
        || !SHA256.test(intent.branch_receipt_digest ?? '')
        || !SHA256.test(intent.branch_receipt_sha256 ?? '')
        || !SHA256.test(intent.candidate_checkpoint_bytes_sha256 ?? '')
        || !SHA256.test(intent.state_diff_sha256 ?? '')
        || !intent.candidate_program_identity || typeof intent.candidate_program_identity !== 'object'
        || Array.isArray(intent.candidate_program_identity)
        || typeof intent.branch_receipt_json !== 'string'
        || sha256Bytes(Buffer.from(intent.branch_receipt_json)) !== intent.branch_receipt_sha256) {
        throw new Error('branch promotion intent differs from durable baseline or receipt identity');
      }
      const validated = validateBranchReceipt(intent.branch_receipt_json, baseline, Buffer.from(intent.candidate_bundle_base64, 'base64'));
      if (validated.receipt.receipt_digest !== intent.branch_receipt_digest
        || validated.candidateDigest !== intent.candidate_checkpoint_bytes_sha256
        || sha256Json(validated.receipt.state_diff) !== intent.state_diff_sha256
        || JSON.stringify(validated.receipt.candidate_program_identity) !== JSON.stringify(intent.candidate_program_identity)) {
        throw new Error('branch promotion intent content does not reproduce its durable identity');
      }
      intentsById.set(intentId, intent);
    }
    const completionsByIntent = new Map();
    for (let index = 0; index < state.completions.length; index++) {
      const completion = state.completions[index];
      if (!completion || typeof completion !== 'object' || Array.isArray(completion)) throw new Error('malformed branch promotion completion');
      const {completion_id: completionId, ...record} = completion;
      const intent = intentsById.get(completion.intent_id);
      if (completion.schema !== COMPLETION_SCHEMA || completion.subject !== this.subject
        || completion.sequence !== index + 1 || completion.status !== 'completed'
        || completionId !== sha256Json(record) || !intent
        || completionsByIntent.has(completion.intent_id)
        || completion.candidate_checkpoint_bytes_sha256 !== intent.candidate_checkpoint_bytes_sha256
        || JSON.stringify(completion.candidate_program_identity) !== JSON.stringify(intent.candidate_program_identity)) {
        throw new Error('branch promotion completion differs from its pending intent');
      }
      completionsByIntent.set(completion.intent_id, completion);
    }
    return {state, intentsById, completionsByIntent};
  }

  beginPromotion({baselineReceiptId, branchReceiptJson, candidateBundle}) {
    return this.withLock(() => {
      const snapshot = this.load();
      if (snapshot.state.intents.length >= MAX_PROMOTIONS) throw new Error('branch promotion lineage is full');
      const baseline = this.executionLedger.getReceipt(baselineReceiptId);
      const {receipt, candidateDigest} = validateBranchReceipt(branchReceiptJson, baseline, candidateBundle);
      const branchReceiptSha256 = sha256Bytes(Buffer.from(branchReceiptJson));
      const existing = snapshot.state.intents.find(intent => intent.baseline_receipt_id === baselineReceiptId
        && intent.branch_receipt_sha256 === branchReceiptSha256
        && intent.candidate_checkpoint_bytes_sha256 === candidateDigest);
      if (existing) return existing;
      const record = {
        schema: INTENT_SCHEMA,
        subject: this.subject,
        sequence: snapshot.state.intents.length + 1,
        status: 'pending',
        baseline_receipt_id: baselineReceiptId,
        baseline_execution_sequence: baseline.sequence,
        baseline_checkpoint_bytes_sha256: baseline.state_checkpoint_bytes_sha256,
        branch_id: receipt.branch_id,
        branch_receipt_digest: receipt.receipt_digest,
        branch_receipt_sha256: branchReceiptSha256,
        branch_receipt_json: branchReceiptJson,
        candidate_bundle_base64: candidateBundle.toString('base64'),
        candidate_checkpoint_bytes_sha256: candidateDigest,
        candidate_program_identity: receipt.candidate_program_identity,
        state_diff_sha256: sha256Json(receipt.state_diff),
      };
      const intent = {...record, intent_id: sha256Json(record)};
      snapshot.state.intents.push(intent);
      this.write(snapshot.state);
      return intent;
    });
  }

  completePromotion(intentId, promotedBundle, promotedProgramIdentity) {
    return this.withLock(() => {
      const snapshot = this.load();
      const intent = snapshot.intentsById.get(intentId);
      if (!intent) throw new Error('branch promotion intent is absent');
      const existing = snapshot.completionsByIntent.get(intentId);
      if (existing) return existing;
      if (!Buffer.isBuffer(promotedBundle) || promotedBundle.length === 0 || promotedBundle.length > MAX_CANDIDATE_BYTES
        || sha256Bytes(promotedBundle) !== intent.candidate_checkpoint_bytes_sha256
        || JSON.stringify(promotedProgramIdentity) !== JSON.stringify(intent.candidate_program_identity)) {
        throw new Error('completed promotion does not match the pending candidate checkpoint and program identity');
      }
      const record = {
        schema: COMPLETION_SCHEMA,
        subject: this.subject,
        sequence: snapshot.state.completions.length + 1,
        status: 'completed',
        intent_id: intentId,
        baseline_receipt_id: intent.baseline_receipt_id,
        branch_id: intent.branch_id,
        candidate_checkpoint_bytes_sha256: intent.candidate_checkpoint_bytes_sha256,
        candidate_program_identity: intent.candidate_program_identity,
      };
      const completion = {...record, completion_id: sha256Json(record)};
      snapshot.state.completions.push(completion);
      this.write(snapshot.state);
      return completion;
    });
  }

  getPromotion(intentId) {
    return this.withLock(() => {
      const snapshot = this.load();
      const intent = snapshot.intentsById.get(intentId);
      if (!intent) throw new Error('branch promotion intent is absent');
      return {intent, completion: snapshot.completionsByIntent.get(intentId) ?? null};
    });
  }
}

export const BRANCH_PROMOTION_LINEAGE_SCHEMAS = Object.freeze({
  ledger: LEDGER_SCHEMA,
  intent: INTENT_SCHEMA,
  completion: COMPLETION_SCHEMA,
});
