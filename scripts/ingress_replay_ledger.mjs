import {createHash, randomUUID} from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import {executionReceipt, EXECUTION_RECEIPT_SCHEMA, sha256Json} from './ingress_execution_receipt.mjs';

const SCHEMA = 'burn-research.ingress-replay-ledger.v1';
const MAX_CLAIMS = 50000;
const MAX_EXECUTIONS = 50000;
const MAX_BYTES = 32 * 1024 * 1024;
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
    const directory = fs.lstatSync(this.directory);
    if (!directory.isDirectory() || directory.isSymbolicLink() || directory.mode & 0o022) throw new Error('ledger parent must be a private host-owned directory');
    validateOwner(directory, 'ledger parent');
    this.withLock(() => {
      if (initialize) {
        if (fs.existsSync(this.file)) throw new Error('refusing to reset an existing durable replay ledger');
        this.write({schema: SCHEMA, subject: this.subject, claims: [], executions: []});
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
      || (state.executions !== undefined && (!Array.isArray(state.executions) || state.executions.length > MAX_EXECUTIONS))) {
      throw new Error('ledger schema, host subject, or claim limit mismatch');
    }
    // Existing v1 ledger snapshots did not contain executions. The first
    // successful receipt write extends them without resetting claim history.
    state.executions ??= [];
    const nonces = new Set();
    const revisions = new Map();
    const hashes = new Set();
    for (const entry of state.claims) {
      if (typeof entry.nonce_key !== 'string' || typeof entry.revision_key !== 'string'
        || typeof entry.claim_sha256 !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(entry.claim_sha256)
        || typeof entry.revision !== 'string' || !/^(0|[1-9][0-9]*)$/.test(entry.revision)) throw new Error('malformed replay ledger entry');
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
    for (let index = 0; index < state.executions.length; index++) {
      const entry = state.executions[index];
      if (!entry || typeof entry !== 'object' || Array.isArray(entry)) throw new Error('malformed execution receipt');
      const {receipt_id: receiptId, ...record} = entry;
      if (entry.schema !== EXECUTION_RECEIPT_SCHEMA || entry.authority !== 'node_host_observed_wasm_run'
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
        const key = JSON.stringify([input.source, this.subject, input.slot]);
        const current = latestAtReceipt.get(key);
        if (!current || current.claim_sha256 !== input.claim_sha256 || current.revision !== input.revision) {
          throw new Error('execution receipt input does not match its committed claim history');
        }
        usedSlots.add(input.slot);
      }
    }
    return {state, nonces, revisions, hashes};
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

  commit(ticket) {
    this.withLock(() => {
      const snapshot = this.load();
      this.check(ticket, snapshot);
      snapshot.state.claims.push({
        nonce_key: ticket.nonceKey,
        revision_key: ticket.revisionKey,
        revision: ticket.claim.revision,
        claim_sha256: digest(ticket.claim),
      });
      this.write(snapshot.state);
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

  executeWithReceipt(claims, identity, execute) {
    return this.withLock(() => {
      const snapshot = this.load();
      const currentClaims = [...claims];
      this.assertCurrent(snapshot, currentClaims);
      if (snapshot.state.executions.length >= MAX_EXECUTIONS) throw new Error('durable execution receipt ledger is full');
      const result = execute();
      const receipt = executionReceipt({...identity, claims: currentClaims,
        shape: result.shape, values: result.values,
        sequence: snapshot.state.executions.length + 1, claimCount: snapshot.state.claims.length});
      snapshot.state.executions.push(receipt);
      this.write(snapshot.state);
      return {...result, execution_receipt: receipt};
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
