import path from 'node:path';
import {spawnSync} from 'node:child_process';

const packageDir = path.resolve(process.argv[2] ?? 'pkg');
const requests = [
  {
    op: 'create', numSlots: 3,
    layers: [{constructor: 'add', args: [31]}],
    steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}],
    outputSlot: 2,
    ports: [
      {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 2},
      {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton', requireFingerprint: true, minimumRevision: 3},
    ],
    logicalPorts: [{id: 'observation', slot: 0, source: 'sensor-a'}, {id: 'memory', slot: 1, source: 'memory-b'}],
  },
  {op: 'bind', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 2, fingerprint: 'obs'},
  {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'wrong-source', revision: 3, fingerprint: 'state'},
  {op: 'inspect'},
  {op: 'run'},
  {op: 'clear', slot: 1},
  {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'memory-b', revision: 3, fingerprint: 'state'},
  {op: 'consumer', slot: 1, id: 'state-reader', acceptedRoles: ['state'], requireFingerprint: true, minimumRevision: 3},
  {op: 'run'},
  {op: 'verify', candidate: [4, 6]},
  {op: 'close'},
];

const run = spawnSync(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {
  input: `${requests.map((request, index) => JSON.stringify({...request, request_id: index})).join('\n')}\n`,
  encoding: 'utf8',
  maxBuffer: 2_000_000,
  timeout: 30_000,
});
if (run.error || run.status !== 0) {
  throw new Error(`interactive runner failed: ${String(run.error ?? run.stderr)} (${run.status})`);
}
const responses = run.stdout.trim().split('\n').map(line => JSON.parse(line));
function assert(condition, message) {
  if (!condition) throw new Error(message);
}
assert(responses.length === requests.length, `response cardinality ${responses.length} != ${requests.length}`);
assert(responses.every((response, index) => response.request_id === index), 'request IDs changed');
assert(responses[0].result.status.ready === false, 'empty inputs reported ready');
assert(responses[3].result.status.graph_preflight.ready === true, 'valid graph inputs failed graph preflight');
assert(responses[3].result.status.ports[1].status === 'source_mismatch', 'source mismatch was not reported');
assert(responses[4].ok === false && responses[4].error.includes('preflight failed'), 'bridge run accepted a mismatched source');
assert(responses[7].result.status === 'compatible', 'consumer compatibility was not reported');
assert(responses[8].ok && JSON.stringify(responses[8].result.values) === '[4,6]', 'reference output mismatch');
assert(responses[8].result.ingress.ready === true, 'ready ingress report missing');
assert(responses[9].result.reference.verification.passed === true, 'reference verification failed');
assert(responses[10].result.closed === true, 'session did not close');
console.log(JSON.stringify({verdict:'PASS',protocol:'JSON Lines',rejected_source:responses[3].result.status.ports[1].status,reference:responses[8].result.values,verified:responses[9].result.reference.verification.passed}));
