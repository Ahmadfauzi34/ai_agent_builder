import {createHash, randomUUID} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {executionReceipt, EXECUTION_RECEIPT_SCHEMA, LEGACY_EXECUTION_RECEIPT_SCHEMA, sha256Json} from './ingress_execution_receipt.mjs';

const SCHEMA = 'burn-research.ingress-replay-ledger.v1';
const MAX_CLAIMS = 50000;
const MAX_EXECUTIONS = 50000;
const MAX_HANDOFFS = 50000;
const MAX_RESTORES = 50000;
const MAX_BYTES = 32 * 1024 * 1024;
const MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024;
const MAX_CHECKPOINT_STORE_BYTES = 512 * 1024 * 1024;
const U64_MAX = (1n << 64n) - 1n;

function digest(claim) {
  return `sha256:${createHash('sha256').update(JSON.stringify(claim)).digest('hex')}`;
}

function validateOwner(stat, description) {
  if (typeof process.getuid === 'function' && stat.uid !== process.getuid()) throw new Error(`${description} must be owned by the runner user`);
}

export class IngressReplayLedger {
  constructor(ledgerPath, subject, {initialize = false} = {}) {
    if (typeof subject !== 'string' || subject.length === 0 || Buffer.byteLength(subject) > 256) throw new Error('host subject must be an explicit nonempty string');
    this.subject = subject;
    this.file = path.resolve(ledgerPath);
    this.directory = path.dirname(this.file);
    this.lockPath = `${this.file}.lock`;
    this.checkpointDirectory = `${this.file}.checkpoints`;
    const directory = fs.lstatSync(this.directory);
    if (!directory.isDirectory() || directory.isSymbolicLink() || directory.mode & 0o022) throw new Error('ledger parent must be a private host-owned directory');
    validateOwner(directory, 'ledger parent');
    this.withLock(() => {
      if (initialize) {
        if (fs.existsSync(this.file)) throw new Error('refusing to reset an existing durable replay ledger');
        this.write({schema: SCHEMA, subject: this.subject, claims: [], executions: [], handoffs: [], restores: []});
      } else this.load();
    });
  }

  withLock(operation) {
    // mkdir is exclusive across processes. A crash leaves a lock behind and
    // deliberately requires a host operator to verify and recover it.
    fs.mkdirSync(this.lockPath, {mode: 0o700});
    try {
      return operation();
    } finally {
      fs.rmdirSync(this.lockPath);
    }
  }

  load() {
    const stat = fs.lstatSync(this.file);
    if (!stat.isFile() || stat.isSymbolicLink() || stat.mode & 0o077 || stat.size > MAX_BYTES) throw new Error('ledger file is not a private bounded regular file');
    validateOwner(stat, 'ledger file');
    const state = JSON.parse(fs.readFileSync(this.file, 'utf8'));
    if (state.schema !== SCHEMA || state.subject !== this.subject || !Array.isArray(state.claims) || state.claims.length > MAX_CLAIMS
      || (state.executions !== undefined && (!Array.isArray(state.executions) || state.executions.length > MAX_EXECUTIONS))
      || (state.handoffs !== undefined && (!Array.isArray(state.handoffs) || state.handoffs.length > MAX_HANDOFFS))
      || (state.restores !== undefined && (!Array.isArray(state.restores) || state.restores.length > MAX_RESTORES))) {
      throw new Error('ledger schema, host subject, or claim limit mismatch');
    }
    // Existing v1 ledger snapshots did not contain executions. The first
    // successful receipt write extends them without resetting claim history.
    state.executions ??= [];
    state.handoffs ??= [];
    state.restores ??= [];
    const nonces = new Set();
    const revisions = new Map();
    const hashes = new Set();
    for (const entry of state.claims) {
      if (typeof entry.nonce_key !== 'string' || typeof entry.revision_key !== 'string'
        || typeof entry.claim_sha256 !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(entry.claim_sha256)
        || typeof entry.revision !== 'string' || !/^(0|[1-9][0-9]*)$/.test(entry.revision)) throw new Error('malformed replay ledger entry');
      if (entry.claim_schema !== undefined
        && !['burn-research.signed-input-claim.v1', 'burn-research.signed-input-claim.v2'].includes(entry.claim_schema)) {
        throw new Error('replay ledger contains an unsupported signed claim schema');
      }
      if ((entry.claim_schema === 'burn-research.signed-input-claim.v2'
          && !/^sha256:[0-9a-f]{64}$/.test(entry.active_state_checkpoint_bytes_sha256 ?? ''))
        || (entry.claim_schema !== 'burn-research.signed-input-claim.v2'
          && entry.active_state_checkpoint_bytes_sha256 !== undefined)) {
        throw new Error('replay ledger state-bound claim digest is malformed');
      }
      const nonceParts = JSON.parse(entry.nonce_key);
      const revisionParts = JSON.parse(entry.revision_key);
      if (!Array.isArray(nonceParts) || nonceParts.length !== 3 || nonceParts.some(part => typeof part !== 'string')
        || !Array.isArray(revisionParts) || revisionParts.length !== 3 || revisionParts[1] !== this.subject
        || typeof revisionParts[0] !== 'string' || !Number.isInteger(revisionParts[2]) || revisionParts[2] < 0 || revisionParts[2] > 63) {
        throw new Error('malformed replay ledger identity');
      }
      const revision = BigInt(entry.revision);
      if (revision > U64_MAX || nonceParts[0] !== revisionParts[0]) throw new Error('invalid replay ledger identity or revision');
      if (nonces.has(entry.nonce_key) || hashes.has(entry.claim_sha256)
        || revision <= (revisions.get(entry.revision_key) ?? -1n)) throw new Error('duplicate or out-of-order replay ledger entry');
      nonces.add(entry.nonce_key);
      hashes.add(entry.claim_sha256);
      revisions.set(entry.revision_key, revision);
    }
    let committedClaims = 0;
    const latestAtReceipt = new Map();
    const receiptsById = new Map();
    for (let index = 0; index < state.executions.length; index++) {
      const entry = state.executions[index];
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) throw new Error('malformed execution receipt');
      const {receipt_id: receiptId, ...record} = entry;
      if (![LEGACY_EXECUTION_RECEIPT_SCHEMA, EXECUTION_RECEIPT_SCHEMA].includes(entry.schema)
        || (entry.schema === EXECUTION_RECEIPT_SCHEMA
          && !/^sha256:[0-9a-f]{64}$/.test(entry.state_checkpoint_bytes_sha256 ?? ''))
        || entry.authority !== 'node_host_observed_wasm_run'
        || entry.subject !== this.subject || entry.sequence !== index + 1
        || !Number.isSafeInteger(entry.claim_count) || entry.claim_count < committedClaims
        || entry.claim_count > state.claims.length || receiptId !== sha256Json(record)
        || typeof entry.program_identity !== 'object' || !entry.program_identity || Array.isArray(entry.program_identity)
        || !/^sha256:[0-9a-f]{64}$/.test(entry.manifest_sha256)
        || !Array.isArray(entry.input_claims) || entry.input_claims.length < 2 || entry.input_claims.length > 64
        || !entry.output || !Array.isArray(entry.output.shape) || entry.output.shape.length !== 4
        || entry.output.shape.some(dim => !Number.isSafeInteger(dim) || dim < 1 || dim > 0xffffffff)
        || !/^sha256:[0-9a-f]{64}$/.test(entry.output.value_sha256)) {
        throw new Error('execution receipt identity or content mismatch');
      }
      while (committedClaims < entry.claim_count) {
        const claim = state.claims[committedClaims++];
        latestAtReceipt.set(claim.revision_key, claim);
      }
      const usedSlots = new Set();
      for (const input of entry.input_claims) {
        if (!Number.isInteger(input.slot) || input.slot < 0 || input.slot > 63
          || usedSlots.has(input.slot) || typeof input.source !== 'string'
          || typeof input.revision !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(input.claim_sha256)) {
          throw new Error('malformed execution receipt input');
        }
        if (input.role !== undefined && (typeof input.role !== 'string' || !Array.isArray(input.shape)
          || input.shape.length !== 4 || input.shape.some(dim => !Number.isSafeInteger(dim) || dim < 1 || dim > 0xffffffff)
          || !/^sha256:[0-9a-f]{64}$/.test(input.value_sha256))) throw new Error('malformed execution receipt input contract');
        if (input.claim_schema !== undefined
          && !['burn-research.signed-input-claim.v1', 'burn-research.signed-input-claim.v2'].includes(input.claim_schema)) {
          throw new Error('execution receipt input has an unsupported signed claim schema');
        }
        if ((input.claim_schema === 'burn-research.signed-input-claim.v2'
            && !/^sha256:[0-9a-f]{64}$/.test(input.active_state_checkpoint_bytes_sha256 ?? ''))
          || (input.claim_schema !== 'burn-research.signed-input-claim.v2'
            && input.active_state_checkpoint_bytes_sha256 !== undefined)) {
          throw new Error('execution receipt state-bound input digest is malformed');
        }
        const key = JSON.stringify([input.source, this.subject, input.slot]);
        const current = latestAtReceipt.get(key);
        if (!current || current.claim_sha256 !== input.claim_sha256 || current.revision !== input.revision
          || (input.claim_schema !== undefined && current.claim_schema !== input.claim_schema)
          || input.active_state_checkpoint_bytes_sha256 !== current.active_state_checkpoint_bytes_sha256) {
          throw new Error('execution receipt input does not match its committed claim history');
        }
        usedSlots.add(input.slot);
      }
      const allStateBoundInputs = entry.input_claims.every(input => input.claim_schema === 'burn-research.signed-input-claim.v2');
      const inputCheckpointDigests = new Set(entry.input_claims.map(input => input.active_state_checkpoint_bytes_sha256));
      if ((entry.input_state_checkpoint_bytes_sha256 !== undefined
          && !/^sha256:[0-9a-f]{64}$/.test(entry.input_state_checkpoint_bytes_sha256))
        || (allStateBoundInputs && (inputCheckpointDigests.size !== 1
          || entry.input_state_checkpoint_bytes_sha256 !== [...inputCheckpointDigests][0]))
        || (!allStateBoundInputs && entry.input_state_checkpoint_bytes_sha256 !== undefined)) {
        throw new Error('execution receipt common input checkpoint digest is inconsistent');
      }
      receiptsById.set(receiptId, entry);
    }
    const restoresById = new Map();
    let previousRestoreExecutions = 0;
    let previousRestoreClaims = 0;
    for (let index = 0; index < state.restores.length; index++) {
      const event = state.restores[index];
      if (!event || typeof event !== 'object' || Array.isArray(event)) throw new Error('malformed checkpoint restore event');
      const {restore_id: restoreId, ...record} = event;
      const parent = receiptsById.get(event.parent_receipt_id);
      if (event.schema !== 'burn-research.host-checkpoint-restore.v1'
        || event.subject !== this.subject || event.sequence !== index + 1
        || !Number.isSafeInteger(event.after_execution_sequence)
        || event.after_execution_sequence < previousRestoreExecutions
        || event.after_execution_sequence > state.executions.length
        || !Number.isSafeInteger(event.claim_count) || event.claim_count < previousRestoreClaims
        || event.claim_count > state.claims.length || restoreId !== sha256Json(record)
        || !parent || parent.schema !== EXECUTION_RECEIPT_SCHEMA
        || parent.sequence > event.after_execution_sequence
        || parent.claim_count > event.claim_count
        || parent.state_checkpoint_bytes_sha256 !== event.checkpoint_bytes_sha256
        || parent.manifest_sha256 !== event.manifest_sha256
        || JSON.stringify(parent.program_identity) !== JSON.stringify(event.program_identity)) {
        throw new Error('checkpoint restore event differs from its committed parent receipt');
      }
      previousRestoreExecutions = event.after_execution_sequence;
      previousRestoreClaims = event.claim_count;
      restoresById.set(restoreId, event);
    }
    for (const receipt of state.executions) {
      const parentId = receipt.state_parent_receipt_id;
      const restoreId = receipt.restore_event_id;
      if (parentId === undefined && restoreId === undefined) continue; // historical receipts and fresh graph sessions
      const parent = receiptsById.get(parentId);
      const restore = restoreId === undefined ? null : restoresById.get(restoreId);
      if (!parent || parent.schema !== EXECUTION_RECEIPT_SCHEMA || parent.sequence >= receipt.sequence
        || JSON.stringify(parent.program_identity) !== JSON.stringify(receipt.program_identity)
        || (restoreId !== undefined && (!restore || receipt.sequence <= restore.after_execution_sequence
          || (parentId !== restore.parent_receipt_id && parent.restore_event_id !== restoreId)))) {
        throw new Error('execution receipt does not preserve its runtime-state ancestry');
      }
    }
    let handoffClaims = 0;
    const latestAtHandoff = new Map();
    const handoffsById = new Map();
    const latestParentByLane = new Map();
    for (const handoff of state.handoffs) {
      if (!handoff || typeof handoff !== 'object' || Array.isArray(handoff)) throw new Error('malformed durable state handoff');
      const {handoff_id: handoffId, ...record} = handoff;
      const target = handoff.input;
      if (handoff.schema !== 'burn-research.host-state-handoff.v1' || handoff.subject !== this.subject
        || typeof handoff.parent_receipt_id !== 'string' || !receiptsById.has(handoff.parent_receipt_id)
        || !Number.isSafeInteger(handoff.claim_count) || handoff.claim_count < handoffClaims
        || handoff.claim_count > state.claims.length || handoffId !== sha256Json(record)
        || typeof handoff.branch_id !== 'string' || handoff.branch_id.length === 0 || Buffer.byteLength(handoff.branch_id) > 256
        || !target || target.role !== 'state' || !Number.isInteger(target.slot) || target.slot < 0 || target.slot > 63
        || typeof target.source !== 'string' || target.source.length === 0 || Buffer.byteLength(target.source) > 256
        || typeof target.logical_port_id !== 'string' || target.logical_port_id.length === 0 || Buffer.byteLength(target.logical_port_id) > 256
        || typeof target.revision !== 'string' || !Array.isArray(target.shape) || target.shape.length !== 4
        || target.shape.some(dim => !Number.isSafeInteger(dim) || dim < 1 || dim > 0xffffffff)
        || !/^sha256:[0-9a-f]{64}$/.test(target.value_sha256) || !/^sha256:[0-9a-f]{64}$/.test(target.claim_sha256)) {
        throw new Error('state handoff identity or content mismatch');
      }
      while (handoffClaims < handoff.claim_count) {
        const claim = state.claims[handoffClaims++];
        latestAtHandoff.set(claim.revision_key, claim);
      }
      const claimKey = JSON.stringify([target.source, this.subject, target.slot]);
      const current = latestAtHandoff.get(claimKey);
      if (!current || current.claim_sha256 !== target.claim_sha256 || current.revision !== target.revision) {
        throw new Error('state handoff does not match its committed input claim');
      }
      const parent = receiptsById.get(handoff.parent_receipt_id);
      if (parent.subject !== this.subject || parent.claim_count >= handoff.claim_count
        || JSON.stringify(parent.output.shape) !== JSON.stringify(target.shape)
        || parent.output.value_sha256 !== target.value_sha256) throw new Error('state handoff parent output differs from the bound state');
      const lane = JSON.stringify([this.subject, target.source, target.logical_port_id, handoff.branch_id]);
      const previousParent = latestParentByLane.get(lane) ?? 0;
      if (parent.sequence <= previousParent) throw new Error('state handoff replays or rewinds its parent receipt');
      latestParentByLane.set(lane, parent.sequence);
      handoffsById.set(handoffId, handoff);
    }
    for (const receipt of state.executions) {
      for (const input of receipt.input_claims) {
        if (input.handoff_id === undefined) continue;
        const handoff = handoffsById.get(input.handoff_id);
        const parent = handoff && receiptsById.get(handoff.parent_receipt_id);
        const target = handoff?.input;
        if (!handoff || !parent || !target || receipt.sequence <= parent.sequence
          || receipt.claim_count < handoff.claim_count
          || input.role !== 'state' || input.claim_sha256 !== target.claim_sha256
          || input.slot !== target.slot || input.source !== target.source || input.revision !== target.revision
          || input.value_sha256 !== target.value_sha256 || JSON.stringify(input.shape) !== JSON.stringify(target.shape)) {
          throw new Error('execution receipt does not preserve its state handoff lineage');
        }
      }
    }
    return {state, nonces, revisions, hashes, receiptsById, restoresById, handoffsById, latestParentByLane};
  }

  check(ticket, snapshot = this.load()) {
    if (snapshot.nonces.has(ticket.nonceKey)) throw new Error('signed claim nonce replay');
    if (snapshot.state.claims.length >= MAX_CLAIMS) throw new Error('durable replay ledger is full; refuse new binds');
    if (BigInt(ticket.claim.revision) <= (snapshot.revisions.get(ticket.revisionKey) ?? -1n)) throw new Error('stale signed input revision');
  }

  write(state) {
    const contents = JSON.stringify(state) + '\n';
    if (Buffer.byteLength(contents) > MAX_BYTES) throw new Error('durable replay ledger exceeds its size limit');
    const temporary = path.join(this.directory, `.${path.basename(this.file)}.${randomUUID()}.tmp`);
    let fd;
    try {
      fd = fs.openSync(temporary, 'wx', 0o600);
      fs.writeFileSync(fd, contents);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
      fd = undefined;
      fs.renameSync(temporary, this.file);
      const directoryFd = fs.openSync(this.directory, 'r');
      try { fs.fsyncSync(directoryFd); } finally { fs.closeSync(directoryFd); }
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
  }

  checkpointPath(receiptId) {
    if (!/^sha256:[0-9a-f]{64}$/.test(receiptId)) throw new Error('invalid checkpoint receipt ID');
    return path.join(this.checkpointDirectory, `${receiptId.slice(7)}.json`);
  }

  checkCheckpointDirectory({create = false} = {}) {
    if (create && !fs.existsSync(this.checkpointDirectory)) {
      fs.mkdirSync(this.checkpointDirectory, {mode: 0o700});
      const fd = fs.openSync(this.directory, 'r');
      try { fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
    }
    const stat = fs.lstatSync(this.checkpointDirectory);
    if (!stat.isDirectory() || stat.isSymbolicLink() || stat.mode & 0o077) throw new Error('checkpoint directory must be private');
    validateOwner(stat, 'checkpoint directory');
  }

  retainCheckpoint(receipt, bytes, manifest) {
    if (!Buffer.isBuffer(bytes) || bytes.length === 0 || bytes.length > MAX_CHECKPOINT_BYTES) throw new Error('checkpoint bytes exceed the host retention limit');
    if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) throw new Error('checkpoint manifest is missing');
    const expected = `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
    if (receipt.state_checkpoint_bytes_sha256 !== expected
      || receipt.manifest_sha256 !== sha256Json(manifest)) throw new Error('checkpoint differs from committed receipt');
    this.checkCheckpointDirectory({create: true});
    const contents = JSON.stringify({schema: 'burn-research.host-retained-checkpoint.v1',
      subject: this.subject, receipt_id: receipt.receipt_id, manifest,
      bundle_f32le_base64: bytes.toString('base64')}) + '\n';
    const size = Buffer.byteLength(contents);
    const used = fs.readdirSync(this.checkpointDirectory).reduce((total, name) => {
      const stat = fs.lstatSync(path.join(this.checkpointDirectory, name));
      if (!stat.isFile() || stat.isSymbolicLink() || stat.mode & 0o077) throw new Error('checkpoint store contains an unsafe entry');
      validateOwner(stat, 'checkpoint entry');
      return total + stat.size;
    }, 0);
    if (used + size > MAX_CHECKPOINT_STORE_BYTES) throw new Error('checkpoint store exceeds its size limit');
    const target = this.checkpointPath(receipt.receipt_id);
    const temporary = path.join(this.checkpointDirectory, `.${randomUUID()}.tmp`);
    let fd;
    try {
      fd = fs.openSync(temporary, 'wx', 0o600);
      fs.writeFileSync(fd, contents);
      fs.fsyncSync(fd);
      fs.closeSync(fd);
      fd = undefined;
      fs.linkSync(temporary, target); // exclusive: never overwrite another committed checkpoint
      const dirFd = fs.openSync(this.checkpointDirectory, 'r');
      try { fs.fsyncSync(dirFd); } finally { fs.closeSync(dirFd); }
      return target;
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
      if (fs.existsSync(temporary)) fs.unlinkSync(temporary);
    }
  }

  getCheckpoint(receiptId) {
    return this.withLock(() => {
      const receipt = this.load().receiptsById.get(receiptId);
      if (!receipt || receipt.schema !== EXECUTION_RECEIPT_SCHEMA) throw new Error('checkpoint receipt is absent from the durable ledger');
      this.checkCheckpointDirectory();
      const file = this.checkpointPath(receiptId);
      const stat = fs.lstatSync(file);
      if (!stat.isFile() || stat.isSymbolicLink() || stat.mode & 0o077
        || stat.size > MAX_CHECKPOINT_BYTES * 2) throw new Error('retained checkpoint is not a private bounded file');
      validateOwner(stat, 'retained checkpoint');
      const record = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (record.schema !== 'burn-research.host-retained-checkpoint.v1'
        || record.subject !== this.subject || record.receipt_id !== receiptId
        || !record.manifest || typeof record.bundle_f32le_base64 !== 'string'
        || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(record.bundle_f32le_base64)) {
        throw new Error('retained checkpoint envelope is invalid');
      }
      const bytes = Buffer.from(record.bundle_f32le_base64, 'base64');
      if (bytes.length === 0 || bytes.length > MAX_CHECKPOINT_BYTES
        || bytes.toString('base64') !== record.bundle_f32le_base64
        || `sha256:${createHash('sha256').update(bytes).digest('hex')}` !== receipt.state_checkpoint_bytes_sha256
        || sha256Json(record.manifest) !== receipt.manifest_sha256) {
        throw new Error('retained checkpoint bytes or manifest differ from receipt');
      }
      return {receipt, bytes, manifest: record.manifest};
    });
  }

  recordRestore(receiptId, identity, manifestSha256, checkpointBytesSha256) {
    return this.withLock(() => {
      const snapshot = this.load();
      if (snapshot.state.restores.length >= MAX_RESTORES) throw new Error('durable checkpoint restore history is full');
      const parent = snapshot.receiptsById.get(receiptId);
      if (!parent || parent.schema !== EXECUTION_RECEIPT_SCHEMA
        || parent.state_checkpoint_bytes_sha256 !== checkpointBytesSha256
        || parent.manifest_sha256 !== manifestSha256
        || JSON.stringify(parent.program_identity) !== JSON.stringify(identity)) {
        throw new Error('checkpoint restore ancestry differs from its receipt');
      }
      const record = {schema: 'burn-research.host-checkpoint-restore.v1', subject: this.subject,
        sequence: snapshot.state.restores.length + 1,
        after_execution_sequence: snapshot.state.executions.length,
        claim_count: snapshot.state.claims.length, parent_receipt_id: receiptId,
        checkpoint_bytes_sha256: checkpointBytesSha256,
        manifest_sha256: manifestSha256, program_identity: identity};
      const event = {...record, restore_id: sha256Json(record)};
      snapshot.state.restores.push(event);
      this.write(snapshot.state);
      return event;
    });
  }

  commit(ticket, {parentReceiptId, branchId = 'main'} = {}) {
    return this.withLock(() => {
      const snapshot = this.load();
      this.check(ticket, snapshot);
      let handoff;
      if (parentReceiptId !== undefined) {
        if (ticket.claim.role !== 'state') throw new Error('receipt handoff is only valid for a state input');
        if (snapshot.state.handoffs.length >= MAX_HANDOFFS) throw new Error('durable state handoff ledger is full');
        if (typeof branchId !== 'string' || branchId.length === 0 || Buffer.byteLength(branchId) > 256) throw new Error('handoff branch ID must be a nonempty string of at most 256 bytes');
        const parent = snapshot.receiptsById.get(parentReceiptId);
        if (!parent) throw new Error('state handoff parent receipt is absent from the durable ledger');
        if (JSON.stringify(parent.output.shape) !== JSON.stringify(ticket.claim.shape)
          || parent.output.value_sha256 !== ticket.claim.value_sha256) throw new Error('state input bytes or shape differ from its parent receipt output');
        const lane = JSON.stringify([this.subject, ticket.claim.source, ticket.claim.logical_port_id, branchId]);
        const lastParentSequence = snapshot.latestParentByLane.get(lane) ?? 0;
        if (parent.sequence <= lastParentSequence) throw new Error('state handoff replays or rewinds its parent receipt');
      }
      snapshot.state.claims.push({
        nonce_key: ticket.nonceKey,
        revision_key: ticket.revisionKey,
        revision: ticket.claim.revision,
        claim_schema: ticket.claim.schema,
        ...(ticket.claim.schema === 'burn-research.signed-input-claim.v2'
          ? {active_state_checkpoint_bytes_sha256: ticket.claim.active_state_checkpoint_bytes_sha256} : {}),
        claim_sha256: digest(ticket.claim),
      });
      if (parentReceiptId !== undefined) {
        const input = {slot: ticket.claim.slot, source: ticket.claim.source,
          logical_port_id: ticket.claim.logical_port_id, role: ticket.claim.role,
          revision: ticket.claim.revision, shape: [...ticket.claim.shape],
          value_sha256: ticket.claim.value_sha256, claim_sha256: digest(ticket.claim)};
        const record = {schema: 'burn-research.host-state-handoff.v1', subject: this.subject,
          parent_receipt_id: parentReceiptId, branch_id: branchId,
          claim_count: snapshot.state.claims.length, input};
        handoff = {...record, handoff_id: sha256Json(record)};
        snapshot.state.handoffs.push(handoff);
        const lane = JSON.stringify([this.subject, ticket.claim.source, ticket.claim.logical_port_id, branchId]);
        snapshot.latestParentByLane.set(lane, snapshot.receiptsById.get(parentReceiptId).sequence);
      }
      this.write(snapshot.state);
      return handoff;
    });
  }

  withCurrent(claims, execute) {
    return this.withLock(() => {
      const snapshot = this.load();
      this.assertCurrent(snapshot, claims);
      return execute();
    });
  }

  assertCurrent(snapshot, claims) {
    for (const claim of claims) {
      const revisionKey = JSON.stringify([claim.source, claim.subject, claim.slot]);
      if (!snapshot.hashes.has(digest(claim)) || snapshot.revisions.get(revisionKey) !== BigInt(claim.revision)) {
        throw new Error('signed input is absent from durable ledger or no longer the current revision');
      }
    }
  }

  executeWithReceipt(claims, identity, execute, handoffs = new Map()) {
    return this.withLock(() => {
      const snapshot = this.load();
      const currentClaims = [...claims];
      this.assertCurrent(snapshot, currentClaims);
      if (snapshot.state.executions.length >= MAX_EXECUTIONS) throw new Error('durable execution receipt ledger is full');
      const parentId = identity.stateParentReceiptId;
      const restoreId = identity.restoreEventId;
      if (parentId !== undefined) {
        const parent = snapshot.receiptsById.get(parentId);
        const restore = restoreId !== undefined ? snapshot.restoresById.get(restoreId) : null;
        if (!parent || parent.schema !== EXECUTION_RECEIPT_SCHEMA
          || JSON.stringify(parent.program_identity) !== JSON.stringify(identity.programIdentity)
          || (restoreId !== undefined && (!restore
            || (parentId !== restore.parent_receipt_id && parent.restore_event_id !== restoreId)))) {
          throw new Error('runtime-state parent is absent or differs from this session');
        }
      } else if (restoreId !== undefined) throw new Error('restore event requires a runtime-state parent');
      const result = execute();
      const receipt = executionReceipt({...identity, claims: currentClaims, handoffs,
        shape: result.shape, values: result.values,
        checkpointBytesSha256: result.state_checkpoint_bytes_sha256,
        sequence: snapshot.state.executions.length + 1, claimCount: snapshot.state.claims.length});
      snapshot.state.executions.push(receipt);
      let retained;
      try {
        if (result.checkpoint_bytes !== undefined) retained = this.retainCheckpoint(receipt, result.checkpoint_bytes, result.checkpoint_manifest);
        this.write(snapshot.state);
      } catch (error) {
        if (retained) fs.unlinkSync(retained);
        throw error;
      }
      const {checkpoint_bytes: ignoredBytes, checkpoint_manifest: ignoredManifest, ...publicResult} = result;
      return {...publicResult, execution_receipt: receipt};
    });
  }

  getReceipt(receiptId) {
    return this.withLock(() => {
      const receipt = this.load().state.executions.find(entry => entry.receipt_id === receiptId);
      if (!receipt) throw new Error('execution receipt is absent from the durable ledger');
      return receipt;
    });
  }
}
