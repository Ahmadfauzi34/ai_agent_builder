import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn} from 'node:child_process';
import {generateKeyPairSync, sign} from 'node:crypto';
import {canonicalInputClaim, manifestDigest} from './ingress_provenance.mjs';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'signed-ingress-'));
const {privateKey, publicKey} = generateKeyPairSync('ed25519');
const {privateKey: foreignKey} = generateKeyPairSync('ed25519');
const trustPath = path.join(temporary, 'trust.json');
fs.writeFileSync(trustPath, JSON.stringify({
  schema: 'burn-research.ingress-trust-policy.v1',
  issuers: [
    {source: 'sensor-a', key_id: 'lab-key', subjects: ['run-1'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})},
    {source: 'memory-b', key_id: 'lab-key', subjects: ['run-1'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})},
  ],
}));

const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs'), packageDir, trustPath], {stdio: ['pipe', 'pipe', 'pipe']});
let stderr = '';
child.stderr.on('data', data => { stderr += data; });
const lines = readline.createInterface({input: child.stdout});
const responses = [];
lines.on('line', line => responses.shift()?.(JSON.parse(line)));
let sequence = 0;
function ask(command) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('runner response timeout')), 5000);
    responses.push(response => { clearTimeout(timer); resolve(response); });
    child.stdin.write(JSON.stringify({...command, request_id: ++sequence}) + '\n');
  });
}
function check(condition, message) {
  if (!condition) throw new Error(message);
}
function signed(binding, manifest, nonce, subject = 'run-1', key = privateKey) {
  const mapped = manifest.ports.find(port => port.slot === binding.slot);
  const claim = canonicalInputClaim(binding, {
    plan_hex: manifest.plan_hex,
    manifest_fingerprint: manifest.manifest_fingerprint,
    manifest_sha256: manifestDigest(manifest),
    logical_port_id: mapped.logical_port_id,
  }, 'lab-key', subject, nonce);
  return {...binding, proof: {claim, signature: sign(null, Buffer.from(JSON.stringify(claim)), key).toString('base64')}};
}

try {
  check((await ask({op: 'capabilities'})).result.host_provenance.mode === 'ed25519_host_enforced', 'host trust policy did not activate');
  const created = await ask({
    op: 'create', numSlots: 3,
    layers: [{constructor: 'add', args: [31]}],
    steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}],
    outputSlot: 2,
    ports: [
      {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 2},
      {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 3},
    ],
    logicalPorts: [{id: 'observation', slot: 0, source: 'sensor-a'}, {id: 'memory', slot: 1, source: 'memory-b'}],
  });
  check(created.ok && !created.result.host_provenance.ready, 'initial proof state wrong');
  const manifest = created.result.manifest;
  const left = {op: 'bind', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 2, fingerprint: 'obs'};
  const right = {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'memory-b', revision: 3, fingerprint: 'state'};

  check(!(await ask(left)).ok, 'unsigned bind succeeded');
  // R-11: empty fingerprint in signed mode fails closed with the documented contract.
  const emptyFp = await ask({...left, fingerprint: ''});
  check(!emptyFp.ok && emptyFp.error_envelope?.code === 'invalid_argument'
    && emptyFp.error_envelope?.path === 'fingerprint'
    && emptyFp.error_envelope?.constraint === 'non_empty'
    && emptyFp.error_envelope?.authority_mode === 'ed25519_host_enforced',
    'empty fingerprint in signed mode did not fail with the documented contract');
  check(!(await ask(signed(left, manifest, 'foreign', 'run-1', foreignKey))).ok, 'foreign signature succeeded');
  check(!(await ask(signed(left, manifest, 'subject', 'other-run'))).ok, 'foreign subject succeeded');
  check(!(await ask(signed({...right, source: 'sensor-a'}, manifest, 'wrong-source'))).ok, 'signed source bypassed logical mapping');
  const leftSigned = signed(left, manifest, 'n-left');
  check(!(await ask({...leftSigned, values: [9, 2]})).ok, 'mutated tensor value succeeded');
  check((await ask({op: 'inspect'})).result.status.graph_preflight.inputs.bound_port_count === 0, 'rejected proof mutated the input bundle');
  check((await ask(leftSigned)).ok, 'valid signed observation rejected');
  check(!(await ask(leftSigned)).ok, 'replayed nonce succeeded');
  check(!(await ask({op: 'run'})).ok, 'incomplete signed coverage ran');
  check((await ask(signed(right, manifest, 'n-right'))).ok, 'valid signed state rejected');
  const firstRun = await ask({op: 'run'});
  check(firstRun.ok && firstRun.result.host_provenance.ready && JSON.stringify(firstRun.result.values) === '[4,6]', 'signed reference output mismatch');
  // R-10: ready is the execution gate; no FIELD named execution_authorized may
  // stay false on a legitimate successful run (the word may still appear inside
  // documentation strings explaining the removal).
  check(!JSON.stringify(firstRun.result).includes('"execution_authorized":'),
    'execution_authorized field leaked into a signed run result');
  const rhp = firstRun.result.host_provenance;
  check(rhp.authority_mode === 'ed25519_host_enforced' && rhp.execution_gate?.field === 'ready'
    && rhp.host_authority_attested === true && rhp.denial_reason === null
    && rhp.claim_verification?.all_verified === true
    && rhp.claim_verification?.verified_ports === rhp.claim_verification?.total_ports,
    'signed gate status malformed on ready');
  // R-10 negative: before claim coverage completes, denial_reason names the blocker.
  const chp = created.result.host_provenance;
  check(chp.ready === false && chp.host_authority_attested === false
    && typeof chp.denial_reason === 'string' && chp.denial_reason.length > 0
    && chp.enforcement_effect.includes('deny-by-default'), 'signed denial status malformed');
  const verification = await ask({op: 'verify', candidate: [4, 6]});
  check(verification.result.reference.verification.passed && verification.result.host_provenance.ready, 'signed reference verification failed');
  // R-10: the field is gone from trace and verify surfaces too.
  const signedTrace = await ask({op: 'trace'});
  check(signedTrace.ok && JSON.stringify(signedTrace.result.values) === '[4,6]', 'signed trace failed');
  for (const [name, res] of [['trace', signedTrace], ['verify', verification]]) {
    check(!JSON.stringify(res.result).includes('"execution_authorized":'),
      `execution_authorized field leaked into signed ${name} result`);
    check(res.result.host_provenance.host_authority_attested === true
      && res.result.host_provenance.denial_reason === null, `signed ${name} gate status malformed`);
  }
  check(signedTrace.result.execution_trace.steps[0].layer_type_name === 'binary',
    'signed trace step lacks canonical layer type name');

  const deferred = await ask({op: 'defer', id: 'optional-context', role: 'x-context', required: false});
  check(deferred.ok, 'optional manifest transition failed');
  check(!(await ask({op: 'run'})).ok, 'stale manifest signatures reached execution');
  check((await ask({op: 'clear', slot: 0})).result.cleared, 'left clear failed');
  check((await ask({op: 'clear', slot: 1})).result.cleared, 'right clear failed');
  const nextManifest = (await ask({op: 'inspect'})).result.manifest;
  check(!(await ask(signed({...right, revision: 3}, nextManifest, 'new-nonce-stale-revision'))).ok, 'stale revision succeeded after clear');
  check((await ask(signed({...left, revision: 3}, nextManifest, 'n-left-new'))).ok, 'updated observation rejected');
  check((await ask(signed({...right, revision: 4}, nextManifest, 'n-right-new'))).ok, 'updated state rejected');
  const secondRun = await ask({op: 'run'});
  check(secondRun.ok && secondRun.result.host_provenance.ready && JSON.stringify(secondRun.result.values) === '[4,6]', 'manifest transition did not recover');
  check((await ask({op: 'close'})).result.closed, 'session close failed');
  const exitCode = await new Promise(resolve => child.on('close', resolve));
  check(exitCode === 0, `runner did not exit after close: ${stderr}`);
  console.log(JSON.stringify({verdict: 'PASS', mode: 'ed25519_host_enforced', unsigned_rejected: true,
    forged_rejected: true, value_tamper_rejected: true, replay_rejected: true, stale_revision_rejected: true,
    stale_manifest_rejected: true, reference: secondRun.result.values}));
} finally {
  if (!child.killed) child.kill();
  fs.rmSync(temporary, {recursive: true, force: true});
}
