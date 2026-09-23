import {createHash, createPublicKey, verify as verifySignature} from 'node:crypto';
import fs from 'node:fs';
import {IngressReplayLedger} from './ingress_replay_ledger.mjs';
import {f32ValueDigest} from './ingress_execution_receipt.mjs';

export const SIGNED_INPUT_CLAIM_SCHEMA_V1 = 'burn-research.signed-input-claim.v1';
export const SIGNED_INPUT_CLAIM_SCHEMA_V2 = 'burn-research.signed-input-claim.v2';
const U64_MAX = (1n << 64n) - 1n;
const MAX_ACCEPTED_CLAIMS = 50000;
const SIGNED_INPUT_CLAIM_SCHEMAS = new Set([SIGNED_INPUT_CLAIM_SCHEMA_V1, SIGNED_INPUT_CLAIM_SCHEMA_V2]);

export function manifestDigest(manifest) {
  return `sha256:${createHash('sha256').update(JSON.stringify(manifest)).digest('hex')}`;
}

function requiredString(value, name, maxLength = 256) {
  if (typeof value !== 'string' || value.length === 0 || Buffer.byteLength(value) > maxLength) throw new Error(`${name} must be a nonempty string of at most ${maxLength} bytes`);
  return value;
}

function revisionString(value) {
  if (typeof value === 'number' && !Number.isSafeInteger(value)) throw new Error('revision number must be a safe integer; use a decimal string for larger revisions');
  if (!/^(0|[1-9][0-9]*)$/.test(String(value))) throw new Error('revision must be a nonnegative decimal integer');
  const revision = BigInt(value);
  if (revision > U64_MAX) throw new Error('revision exceeds u64');
  return revision.toString();
}

// Signed bytes use JSON.stringify on this exact insertion order and the SHA-256
// of little-endian f32 values, which are the bytes submitted to WasmTensor.
export function canonicalInputClaim(binding, context, keyId, subject, nonce) {
  const shape = binding.shape;
  if (!Array.isArray(shape) || shape.length !== 4 || shape.some(dim => !Number.isSafeInteger(dim) || dim < 1 || dim > 0xffffffff)) {
    throw new Error('shape must contain four positive u32 dimensions');
  }
  if (!Number.isInteger(binding.slot) || binding.slot < 0 || binding.slot > 63) throw new Error('slot must be 0..63');
  return {
    schema: SIGNED_INPUT_CLAIM_SCHEMA_V1,
    key_id: requiredString(keyId, 'key_id'),
    plan_hex: requiredString(context.plan_hex, 'plan_hex', 131072),
    manifest_fingerprint: requiredString(context.manifest_fingerprint, 'manifest_fingerprint'),
    manifest_sha256: requiredString(context.manifest_sha256, 'manifest_sha256'),
    logical_port_id: requiredString(context.logical_port_id, 'logical_port_id'),
    slot: binding.slot,
    source: requiredString(binding.source, 'source'),
    subject: requiredString(subject, 'subject'),
    role: requiredString(binding.role, 'role'),
    layout: requiredString(binding.layout, 'layout'),
    shape: [...shape],
    revision: revisionString(binding.revision),
    fingerprint: requiredString(binding.fingerprint, 'fingerprint'),
    nonce: requiredString(nonce, 'nonce'),
    value_sha256: f32ValueDigest(binding.values),
  };
}

function checkpointDigest(value) {
  if (typeof value !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(value)) {
    throw new Error('active state checkpoint digest must be a lowercase SHA-256 value');
  }
  return value;
}

export function canonicalStateBoundInputClaim(binding, context, keyId, subject, nonce) {
  const claim = canonicalInputClaim(binding, context, keyId, subject, nonce);
  return {
    ...claim,
    schema: SIGNED_INPUT_CLAIM_SCHEMA_V2,
    active_state_checkpoint_bytes_sha256: checkpointDigest(context.active_state_checkpoint_bytes_sha256),
  };
}

export class SignedIngressVerifier {
  constructor(configPath, {ledgerPath, subject} = {}) {
    if (Boolean(ledgerPath) !== Boolean(subject)) throw new Error('durable replay requires both a host-owned ledger path and a host subject');
    const policy = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    if (policy.schema !== 'burn-research.ingress-trust-policy.v1' || !Array.isArray(policy.issuers) || !policy.issuers.length) {
      throw new Error('invalid ingress trust policy');
    }
    if (policy.issuers.length > 256) throw new Error('ingress trust policy exceeds 256 issuers');
    this.issuers = new Map();
    this.usedNonces = new Set();
    this.lastRevision = new Map();
    for (const issuer of policy.issuers) {
      const source = requiredString(issuer.source, 'issuer.source');
      const keyId = requiredString(issuer.key_id, 'issuer.key_id');
      const key = createPublicKey(requiredString(issuer.public_key_pem, 'issuer.public_key_pem'));
      if (key.asymmetricKeyType !== 'ed25519') throw new Error('issuer key must be Ed25519');
      if (!Array.isArray(issuer.subjects) || !issuer.subjects.length || issuer.subjects.length > 1024) throw new Error('issuer subjects must be explicit and bounded');
      const subjects = new Set(issuer.subjects.map(subject => requiredString(subject, 'issuer.subject')));
      if (issuer.claim_schemas !== undefined && !Array.isArray(issuer.claim_schemas)) throw new Error('issuer claim_schemas must be an array');
      const claimSchemaList = issuer.claim_schemas ?? [SIGNED_INPUT_CLAIM_SCHEMA_V1];
      const claimSchemas = new Set(claimSchemaList);
      if (claimSchemaList.length === 0 || claimSchemaList.length > SIGNED_INPUT_CLAIM_SCHEMAS.size
        || claimSchemas.size !== claimSchemaList.length
        || [...claimSchemas].some(schema => !SIGNED_INPUT_CLAIM_SCHEMAS.has(schema))) {
        throw new Error('issuer claim_schemas must contain unique supported signed input claim schemas');
      }
      const id = JSON.stringify([source, keyId]);
      if (this.issuers.has(id)) throw new Error('duplicate issuer');
      this.issuers.set(id, {key, subjects, claimSchemas});
    }
    this.hostSubject = subject ?? null;
    this.ledger = ledgerPath ? new IngressReplayLedger(ledgerPath, subject) : null;
    this.acceptedClaimSchemas = [...new Set([...this.issuers.values()].flatMap(issuer => [...issuer.claimSchemas]))];
    this.stateBoundInputClaimsSupported = this.acceptedClaimSchemas.includes(SIGNED_INPUT_CLAIM_SCHEMA_V2);
    this.allIssuersSupportStateBoundInputClaims = [...this.issuers.values()].every(
      issuer => issuer.claimSchemas.has(SIGNED_INPUT_CLAIM_SCHEMA_V2));
  }

  get mode() {
    return this.ledger ? 'ed25519_host_durable' : 'ed25519_host_enforced';
  }

  get replayScope() {
    return this.ledger ? 'host_file_across_restarts' : 'process_lifetime';
  }

  verify(binding, context, proof) {
    if (!proof || typeof proof !== 'object' || !proof.claim || typeof proof.signature !== 'string') throw new Error('signed input claim required before bind');
    const {claim, signature} = proof;
    if (this.hostSubject && claim.subject !== this.hostSubject) throw new Error('signed claim subject differs from the host runtime subject');
    const issuer = this.issuers.get(JSON.stringify([claim.source, claim.key_id]));
    if (!issuer || !issuer.subjects.has(claim.subject)) throw new Error('untrusted source, key, or subject');
    if (!issuer.claimSchemas.has(claim.schema)) throw new Error('issuer policy does not allow this signed input claim schema');
    let expected;
    if (claim.schema === SIGNED_INPUT_CLAIM_SCHEMA_V1) {
      expected = canonicalInputClaim(binding, context, claim.key_id, claim.subject, claim.nonce);
    } else if (claim.schema === SIGNED_INPUT_CLAIM_SCHEMA_V2) {
      expected = canonicalStateBoundInputClaim(binding, context, claim.key_id, claim.subject, claim.nonce);
      if (claim.active_state_checkpoint_bytes_sha256 !== context.active_state_checkpoint_bytes_sha256) {
        throw new Error('signed claim checkpoint digest differs from the active program state');
      }
    } else throw new Error('unsupported signed input claim schema');
    if (JSON.stringify(claim) !== JSON.stringify(expected)) throw new Error('signed claim differs from the current input or manifest');
    if (!/^[A-Za-z0-9+/]{86}==$/.test(signature)) throw new Error('invalid Ed25519 signature encoding');
    const bytes = Buffer.from(signature, 'base64');
    if (bytes.length !== 64 || bytes.toString('base64') !== signature || !verifySignature(null, Buffer.from(JSON.stringify(expected)), issuer.key, bytes)) {
      throw new Error('invalid Ed25519 signature');
    }
    const nonceKey = JSON.stringify([claim.source, claim.key_id, claim.nonce]);
    const revisionKey = JSON.stringify([claim.source, claim.subject, claim.slot]);
    const ticket = {claim, nonceKey, revisionKey};
    if (this.ledger) this.ledger.check(ticket);
    else {
      if (this.usedNonces.has(nonceKey)) throw new Error('signed claim nonce replay');
      if (this.usedNonces.size >= MAX_ACCEPTED_CLAIMS) throw new Error('process replay window is full; external durable replay policy required');
      if (BigInt(claim.revision) <= (this.lastRevision.get(revisionKey) ?? -1n)) throw new Error('stale signed input revision');
    }
    return ticket;
  }

  commit(ticket, handoff) {
    const durableHandoff = this.ledger ? this.ledger.commit(ticket, handoff) : null;
    this.usedNonces.add(ticket.nonceKey);
    this.lastRevision.set(ticket.revisionKey, BigInt(ticket.claim.revision));
    return durableHandoff;
  }

  withCurrent(claims, execute) {
    return this.ledger ? this.ledger.withCurrent(claims, execute) : execute();
  }

  executeWithReceipt(claims, identity, execute, handoffs) {
    if (!this.ledger) throw new Error('durable host ledger required for execution receipts');
    return this.ledger.executeWithReceipt(claims, identity, execute, handoffs);
  }

  getReceipt(receiptId) {
    if (!this.ledger) throw new Error('durable host ledger required for execution receipts');
    return this.ledger.getReceipt(receiptId);
  }

  getCheckpoint(receiptId) {
    if (!this.ledger) throw new Error('durable host ledger required for checkpoint restore');
    return this.ledger.getCheckpoint(receiptId);
  }

  recordRestore(receiptId, identity, manifestSha256, checkpointBytesSha256) {
    if (!this.ledger) throw new Error('durable host ledger required for checkpoint restore');
    return this.ledger.recordRestore(receiptId, identity, manifestSha256, checkpointBytesSha256);
  }
}
