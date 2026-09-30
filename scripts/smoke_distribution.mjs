// Distribution smoke test for the packaged Node host (complaint #13, R-17).
// Runs the usage guide's quick start against a clean package directory with
// no manual file hunting:
//   1. caller-declared mode: `node interactive_multi_input_ingress.mjs`
//      (no packageDir argument; the runner defaults to its own directory)
//      then create -> bind -> bind -> run -> verify, expecting [4, 6].
//   2. signed mode: ephemeral Ed25519 key + trust policy, then a signed
//      bind/run session, expecting [4, 6] and ed25519_host_enforced.
//
// Usage: node scripts/smoke_distribution.mjs [packageDir]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import {spawn} from 'node:child_process';
import {generateKeyPairSync, sign} from 'node:crypto';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const runner = path.join(packageDir, 'interactive_multi_input_ingress.mjs');
const {canonicalInputClaim, manifestDigest} = await import(pathToFileUrl(path.join(packageDir, 'ingress_provenance.mjs')));
function pathToFileUrl(p) { return new URL(`file://${p}`).href; }

function check(condition, message) {
  if (!condition) throw new Error(`smoke test failed: ${message}`);
}

function spawnRunner(args) {
  const child = spawn(process.execPath, [runner, ...args], {cwd: packageDir, stdio: ['pipe', 'pipe', 'pipe']});
  let stderr = '';
  child.stderr.on('data', data => { stderr += data; });
  const lines = readline.createInterface({input: child.stdout});
  const pending = [];
  lines.on('line', line => {
    const respond = pending.shift();
    if (respond) respond(JSON.parse(line));
  });
  let sequence = 0;
  const ask = command => new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`runner response timeout (${JSON.stringify(command.op)})`)), 15000);
    pending.push(response => { clearTimeout(timer); resolve(response); });
    child.stdin.write(`${JSON.stringify({...command, request_id: ++sequence})}\n`);
  });
  return {child, ask, stderr: () => stderr};
}

function createCommand() {
  return {
    op: 'create', numSlots: 3,
    layers: [{constructor: 'add', args: [31]}],
    steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}],
    outputSlot: 2,
    ports: [
      {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 2},
      {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 3},
    ],
    logicalPorts: [
      {id: 'observation', slot: 0, source: 'sensor-a'},
      {id: 'memory', slot: 1, source: 'memory-b'},
    ],
  };
}

async function quickStartSession(ask, signBind) {
  const created = await ask(createCommand());
  check(created.ok, `create failed: ${JSON.stringify(created).slice(0, 300)}`);
  const manifest = created.result.manifest;
  const bindSlot = async (slot, values, revision, fingerprint, source) => {
    const binding = {slot, values, shape: [1, 2, 1, 1], role: slot === 0 ? 'observation' : 'state', layout: 'feature_axis1_singleton', source, revision, fingerprint};
    const bound = await ask({op: 'bind', ...(signBind ? signBind(binding, manifest) : binding)});
    check(bound.ok, `bind slot ${slot} failed: ${JSON.stringify(bound).slice(0, 300)}`);
  };
  await bindSlot(0, [1, 2], 2, 'obs', 'sensor-a');
  await bindSlot(1, [3, 4], 3, 'state', 'memory-b');
  const run = await ask({op: 'run'});
  check(run.ok && JSON.stringify(run.result.values) === '[4,6]', `run values wrong: ${JSON.stringify(run).slice(0, 300)}`);
  const verify = await ask({op: 'verify', candidate: [4, 6]});
  check(verify.ok, `verify failed: ${JSON.stringify(verify).slice(0, 300)}`);
}

const results = [];
// Mode 1: caller-declared, no packageDir argument (usage guide quick start).
{
  const {child, ask} = spawnRunner([]);
  try {
    await quickStartSession(ask, null);
    results.push({mode: 'caller_declared', ok: true});
  } finally {
    child.kill();
  }
}
// Mode 2: signed ingress with an ephemeral key and trust policy.
{
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'dist-smoke-'));
  try {
    const {privateKey, publicKey} = generateKeyPairSync('ed25519');
    const trustPath = path.join(temporary, 'trust.json');
    fs.writeFileSync(trustPath, JSON.stringify({
      schema: 'burn-research.ingress-trust-policy.v1',
      issuers: [
        {source: 'sensor-a', key_id: 'lab-key', subjects: ['smoke-1'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})},
        {source: 'memory-b', key_id: 'lab-key', subjects: ['smoke-1'], public_key_pem: publicKey.export({type: 'spki', format: 'pem'})},
      ],
    }));
    const {child, ask} = spawnRunner(['.', trustPath]);
    try {
      const caps = await ask({op: 'capabilities'});
      check(caps.result.host_provenance?.mode === 'ed25519_host_enforced', 'signed mode did not activate');
      let nonce = 0;
      await quickStartSession(ask, (binding, manifest) => {
        const mapped = manifest.ports.find(port => port.slot === binding.slot);
        const claim = canonicalInputClaim(binding, {
          plan_hex: manifest.plan_hex,
          manifest_fingerprint: manifest.manifest_fingerprint,
          manifest_sha256: manifestDigest(manifest),
          logical_port_id: mapped.logical_port_id,
        }, 'lab-key', 'smoke-1', `smoke-${++nonce}`);
        return {...binding, proof: {claim, signature: sign(null, Buffer.from(JSON.stringify(claim)), privateKey).toString('base64')}};
      });
      results.push({mode: 'ed25519_host_enforced', ok: true});
    } finally {
      child.kill();
    }
  } finally {
    fs.rmSync(temporary, {recursive: true, force: true});
  }
}

console.log(JSON.stringify({verdict: 'PASS', packageDir, results}, null, 2));
