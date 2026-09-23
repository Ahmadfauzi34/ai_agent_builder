import {createHash} from 'node:crypto';

export const EXECUTION_RECEIPT_SCHEMA = 'burn-research.host-execution-receipt.v1';

export function sha256Json(value) {
  return `sha256:${createHash('sha256').update(JSON.stringify(value)).digest('hex')}`;
}

export function f32ValueBytes(values) {
  if (!Array.isArray(values)) throw new Error('values must be an array');
  const bytes = Buffer.alloc(values.length * 4);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < values.length; i++) {
    if (typeof values[i] !== 'number' || !Number.isFinite(values[i])) throw new Error('values must be finite numbers');
    const f32 = Math.fround(values[i]);
    if (!Number.isFinite(f32)) throw new Error('values exceed the finite f32 range');
    view.setFloat32(i * 4, f32, true);
  }
  return bytes;
}

export function f32ValueDigest(values) {
  return `sha256:${createHash('sha256').update(f32ValueBytes(values)).digest('hex')}`;
}

export function encodedF32Matches(receipt, shape, base64) {
  if (JSON.stringify(shape) !== JSON.stringify(receipt.output.shape)
    || typeof base64 !== 'string' || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(base64)) return false;
  const bytes = Buffer.from(base64, 'base64');
  if (bytes.toString('base64') !== base64 || bytes.length !== receipt.output.shape.reduce((size, dim) => size * dim, 4)) return false;
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}` === receipt.output.value_sha256;
}

export function executionReceipt({subject, programIdentity, manifestSha256, claims, shape, values, sequence, claimCount}) {
  if (!Number.isSafeInteger(sequence) || sequence < 1 || !Number.isSafeInteger(claimCount) || claimCount < 1) {
    throw new Error('execution receipt requires a durable sequence and claim count');
  }
  const sorted = [...claims].sort((a, b) => a.slot - b.slot);
  if (sorted.length < 2 || sorted.length > 64 || sorted.some((claim, index) =>
    claim.subject !== subject || claim.manifest_sha256 !== manifestSha256
    || !Number.isInteger(claim.slot) || claim.slot < 0 || claim.slot > 63
    || (index > 0 && sorted[index - 1].slot === claim.slot))) {
    throw new Error('receipt input claims are incomplete, duplicated, or inconsistent');
  }
  if (!Array.isArray(shape) || shape.length !== 4 || shape.some(dim => !Number.isSafeInteger(dim) || dim < 1 || dim > 0xffffffff)) {
    throw new Error('execution output shape must have four positive u32 dimensions');
  }
  const record = {
    schema: EXECUTION_RECEIPT_SCHEMA,
    authority: 'node_host_observed_wasm_run',
    sequence,
    claim_count: claimCount,
    subject,
    program_identity: programIdentity,
    manifest_sha256: manifestSha256,
    input_claims: sorted.map(claim => ({slot: claim.slot, source: claim.source,
      revision: claim.revision, claim_sha256: sha256Json(claim)})),
    output: {shape: [...shape], value_sha256: f32ValueDigest(values)},
  };
  return {...record, receipt_id: sha256Json(record)};
}

export function receiptMatchesOutput(receipt, shape, values) {
  return JSON.stringify(shape) === JSON.stringify(receipt.output.shape)
    && f32ValueDigest(values) === receipt.output.value_sha256;
}
