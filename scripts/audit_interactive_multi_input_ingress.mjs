import path from 'node:path';
import {spawn, spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';

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
  {op: 'explain'},
  {op: 'bind', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 2, fingerprint: 'obs'},
  {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'wrong-source', revision: 3, fingerprint: 'state'},
  {op: 'inspect'},
  {op: 'run'},
  {op: 'clear', slot: 1},
  {op: 'bind', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'memory-b', revision: 3, fingerprint: 'state'},
  {op: 'consumer', slot: 1, id: 'state-reader', acceptedRoles: ['state'], requireFingerprint: true, minimumRevision: 3},
  {op: 'run'},
  {op: 'verify', candidate: [4, 6]},
  {op: 'trace', startStep: 0, maxSteps: 1, maxTensorBytes: 64},
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
assert(responses[1].ok && responses[1].result.schema_id === 'burn-research.multi-input-plan-explain.v1', 'plan explanation missing');
assert(responses[1].result.registry_binding_current && responses[1].result.static_shape_status === 'complete', 'plan explanation was not structurally current');
assert(JSON.stringify(responses[1].result.output_shape) === '[1,2,1,1]' && responses[1].result.burn_executed === false,
  'plan explanation did not project the output without execution');
assert(responses[4].result.status.graph_preflight.ready === true, 'valid graph inputs failed graph preflight');
assert(responses[4].result.status.ports[1].status === 'source_mismatch', 'source mismatch was not reported');
assert(responses[5].ok === false && responses[5].error.includes('preflight failed'), 'bridge run accepted a mismatched source');
assert(responses[8].result.status === 'compatible', 'consumer compatibility was not reported');
assert(responses[9].ok && JSON.stringify(responses[9].result.values) === '[4,6]', 'reference output mismatch');
assert(responses[9].result.ingress.ready === true, 'ready ingress report missing');
assert(responses[10].result.reference.verification.passed === true, 'reference verification failed');
const trace = responses[11].result?.execution_trace;
const expectedBytes = Buffer.allocUnsafe(8);
expectedBytes.writeFloatLE(4, 0);
expectedBytes.writeFloatLE(6, 4);
const expectedDigest = `sha256:${createHash('sha256').update(expectedBytes).digest('hex')}`;
assert(responses[11].ok && JSON.stringify(responses[11].result.values) === '[4,6]', 'trace output mismatch');
assert(trace?.schema_id === 'burn-research.multi-input-execution-trace.v1' && trace?.trace_complete === true,
  'complete bounded trace missing');
assert(trace.program_identity.input_plan_hex === responses[0].result.program_identity.input_plan_hex,
  'trace program identity mismatch');
assert(trace.steps[0].observation.value_sha256 === expectedDigest
  && trace.terminal_output.value_sha256 === expectedDigest, 'trace f32 digest mismatch');
assert(!('execution_authorized' in trace) && trace.fault_step_index === null, 'trace authority field not removed or fault mismatch');
// Complaint #10: ready is the documented execution gate; the misleading
// execution_authorized field is gone from every surface.
assert(trace.steps[0].layer_type === 19 && trace.steps[0].layer_type_name === 'binary'
  && trace.steps[0].layer_type_code === 19 && trace.steps[0].layer_type_encoding === 'decimal',
  'trace step layer identity mismatch');
// Complaint #06: hex fingerprint spellings resolve to the same canonical name.
const traceFp = trace.program_identity?.layer_type_identities?.[0];
assert(traceFp && traceFp.layer_type_name === 'binary' && traceFp.layer_type_code === 19
  && traceFp.layer_type_encoding === 'hex', 'trace fingerprint identity mismatch');
assert(responses[12].result.closed === true, 'session did not close');
const closeWithoutEof = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  let stdout = '';
  let stderr = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.stderr.on('data', chunk => { stderr += chunk; });
  child.on('error', reject);
  const timeout = setTimeout(() => { child.kill(); reject(new Error('close waited for stdin EOF')); }, 5000);
  child.on('close', code => { clearTimeout(timeout); resolve({code, stdout, stderr}); });
  child.stdin.write('{"op":"close","request_id":"without-eof"}\n');
});
assert(closeWithoutEof.code === 0 && JSON.parse(closeWithoutEof.stdout.trim()).result.closed, `close without EOF failed: ${closeWithoutEof.stderr}`);
// ---- Fase 1 (#02): structured error envelope regression ----
// Every failure carries a machine-readable envelope; the human `error` string
// stays for backward compatibility; pre-execution failures report
// execution_started=false.
assert(typeof responses[5].error === 'string', 'error string field missing (compat break)');
assert(responses[5].error_envelope?.code === 'preflight_failed', 'preflight failure not classified');
assert(responses[5].error_envelope?.phase === 'preflight', 'preflight failure phase wrong');
assert(responses[5].error_envelope?.execution_started === false, 'pre-execution failure mislabeled as started');
const envelopeCases = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const cases = [
    {op: 'frobnicate', request_id: 'unknown-op'},
    {op: 'bind', request_id: 'no-session', slot: 0, values: [1], shape: [1]},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.write('{"op":"close","request_id":"done"}\n');
  child.stdin.end();
});
const byId = Object.fromEntries(envelopeCases.map(r => [r.request_id, r]));
assert(byId['unknown-op'].ok === false && byId['unknown-op'].error_envelope?.code === 'unknown_operation',
  'unknown op not classified');
assert(byId['no-session'].ok === false && byId['no-session'].error_envelope?.code === 'session_required'
  && byId['no-session'].error_envelope?.execution_started === false, 'session gate not classified');
assert(['unknown-op', 'no-session'].every(id => typeof byId[id].error === 'string'
  && typeof byId[id].error_envelope?.code === 'string'
  && typeof byId[id].error_envelope?.execution_started === 'boolean'), 'envelope shape incomplete');
// ---- Fase 3 (#03): binding lifecycle + validate regression ----
// bind slot 0 (correct): accepted + resolvable, but session not executable yet.
assert(responses[2].result.binding?.accepted === true, 'bind not reported accepted');
assert(responses[2].result.binding?.resolvable === true, 'good bind not reported resolvable');
assert(responses[2].result.binding?.executable === false, 'incomplete session reported executable');
assert(responses[2].result.blockers?.some(b => b.slot === 1), 'bind hid the remaining blocker');
// bind slot 1 (wrong source): accepted but NOT resolvable, with blocker named.
assert(responses[3].result.binding?.accepted === true, 'wrong-source bind not accepted');
assert(responses[3].result.binding?.resolvable === false, 'source-mismatched bind reported resolvable');
assert(responses[3].result.binding?.executable === false, 'mismatched session reported executable');
assert(responses[3].result.blockers?.some(b => b.slot === 1 && b.status === 'source_mismatch'),
  'bind did not name the source_mismatch blocker');
// rebind slot 1 (correct source): fully executable, no blockers.
assert(responses[7].result.binding?.resolvable === true, 'fixed bind not reported resolvable');
assert(responses[7].result.binding?.executable === true, 'ready session not reported executable');
assert(Array.isArray(responses[7].result.blockers) && responses[7].result.blockers.length === 0,
  'ready session still lists blockers');
// validate op: honest pre-run assessment without execution.
const validateCase = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const cases = [
    {op: 'create', request_id: 'c', numSlots: 3,
      layers: [{constructor: 'add', args: [31]}],
      steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2,
      ports: [
        {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'},
        {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'},
      ],
      logicalPorts: [{id: 'observation', slot: 0, source: 'sensor-a'}]},
    {op: 'validate', request_id: 'v0'},
    {op: 'bind', request_id: 'b0', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 1},
    {op: 'validate', request_id: 'v1'},
    {op: 'map', request_id: 'm1', id: 'memory', slot: 1, source: 'memory-b'},
    {op: 'bind', request_id: 'b1', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'memory-b', revision: 1},
    {op: 'validate', request_id: 'v2'},
    {op: 'run', request_id: 'r'},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const vById = Object.fromEntries(validateCase.map(r => [r.request_id, r]));
assert(vById['v0'].ok === true && vById['v0'].result.executable === false,
  'validate must assess (not fail) an incomplete session');
assert(vById['v0'].result.blockers.length >= 1, 'validate hid blockers on empty session');
assert(vById['v1'].result.executable === false && vById['v1'].result.blockers.some(b => b.slot === 1),
  'validate after partial bind must still block on slot 1');
assert(vById['v2'].ok === true && vById['v2'].result.executable === true
  && vById['v2'].result.blockers.length === 0, 'validate did not clear after honest completion');
assert(vById['r'].ok === true && JSON.stringify(vById['r'].result.values) === '[4,6]',
  'validate-then-run changed numerics');
// ---- Fase 2 (#01): explicit logicalPorts preflight regression ----
// create without logicalPorts must name the field explicitly; run must fail
// with the documented missing_required_field envelope (no execution); mapping
// the ports afterwards must recover honestly.
const logicalPortCase = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const cases = [
    {op: 'create', request_id: 'c', numSlots: 3,
      layers: [{constructor: 'add', args: [31]}],
      steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2,
      ports: [
        {slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'},
        {slot: 1, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'},
      ]},
    {op: 'run', request_id: 'r'},
    {op: 'map', request_id: 'm0', id: 'observation', slot: 0, source: 'sensor-a'},
    {op: 'map', request_id: 'm1', id: 'memory', slot: 1, source: 'memory-b'},
    {op: 'bind', request_id: 'b0', slot: 0, values: [1, 2], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 'sensor-a', revision: 1},
    {op: 'bind', request_id: 'b1', slot: 1, values: [3, 4], shape: [1, 2, 1, 1], role: 'state', layout: 'feature_axis1_singleton', source: 'memory-b', revision: 1},
    {op: 'run', request_id: 'r2'},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const lpById = Object.fromEntries(logicalPortCase.map(r => [r.request_id, r]));
const createPreflight = lpById['c'].result.preflight;
assert(lpById['c'].ok === true, 'create without logicalPorts should still succeed (deferred mapping is valid)');
assert(createPreflight && createPreflight.missing_required_fields.includes('logicalPorts'),
  'create preflight did not name the logicalPorts field explicitly');
assert(createPreflight.unmapped_required_slots.length === 2, 'create preflight did not list the unmapped slots');
const lpRun = lpById['r'];
assert(lpRun.ok === false, 'run without logical-port mappings must not succeed');
assert(lpRun.error_envelope?.code === 'missing_required_field', 'run failure not classified as missing_required_field');
assert(lpRun.error_envelope?.phase === 'preflight', 'run failure phase is not preflight');
assert(lpRun.error_envelope?.path === 'logicalPorts', 'run failure does not name path=logicalPorts');
assert(lpRun.error_envelope?.execution_started === false, 'run executed despite missing logicalPorts');
assert(lpRun.error_envelope?.required === true && lpRun.error_envelope?.missing_ports?.length === 2,
  'run failure did not list the required ports');
assert(lpById['r2'].ok === true && JSON.stringify(lpById['r2'].result.values) === '[4,6]',
  'honest recovery after op=map failed or changed numerics');
// ---- Fase 4 (#04, #05): weight revision + state digest; live_registry semantics ----
// Two runs around a weight mutation must show different revision/digest even
// when the numeric outputs coincide (ReLU/zero-input clamping hid the change
// in the original complaint). capabilities, explain, and every run/trace
// result report the same execution_state_semantics ('live_registry'); snapshot
// mode does not exist, declared explicitly via supported_execution_state_semantics.
const weightCase = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const layers = [
    {constructor: 'linear', args: [0, 2, 2, false]},
    {constructor: 'linear', args: [1, 2, 2, false]},
    {constructor: 'add', args: [2]},
    {constructor: 'relu', args: [3]},
  ];
  const steps = [
    {kind: 'unary', layer: 0, slots: [0, 2]},
    {kind: 'unary', layer: 1, slots: [1, 3]},
    {kind: 'binary', layer: 2, slots: [2, 3, 4]},
    {kind: 'unary', layer: 3, slots: [4, 5]},
  ];
  const ports = [0, 1].map(slot => ({slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}));
  const cases = [
    {op: 'capabilities', request_id: 'cap'},
    {op: 'create', request_id: 'c', numSlots: 6, layers, steps, outputSlot: 5, ports,
      logicalPorts: [{id: 'a', slot: 0, source: 't'}, {id: 'b', slot: 1, source: 't'}]},
    {op: 'explain', request_id: 'e'},
    {op: 'bind', request_id: 'b0', slot: 0, values: [0, 0], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 1},
    {op: 'bind', request_id: 'b1', slot: 1, values: [0, 0], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 1},
    {op: 'run', request_id: 'r1'},
    {op: 'setWeights', request_id: 'sw', layer: 0, values: [1, 1, 1, 1]},
    {op: 'run', request_id: 'r2'},
    {op: 'setWeights', request_id: 'sw1', layer: 1, values: [0, 0, 0, 0]},
    {op: 'clear', request_id: 'cl0', slot: 0},
    {op: 'bind', request_id: 'b2', slot: 0, values: [2, 3], shape: [1, 2, 1, 1], role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 2},
    {op: 'run', request_id: 'r3'},
    {op: 'trace', request_id: 'tr'},
    {op: 'setWeights', request_id: 'badcount', layer: 0, values: [1, 2, 3]},
    {op: 'setWeights', request_id: 'weightless', layer: 2, values: [1]},
    {op: 'setWeights', request_id: 'badlayer', layer: 9, values: [1, 1, 1, 1]},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const wById = Object.fromEntries(weightCase.map(r => [r.request_id, r]));
// #05: the same semantics on capabilities, explain, and run/trace results.
assert(wById['cap'].result.execution_state_semantics === 'live_registry', 'capabilities hides state semantics');
assert(JSON.stringify(wById['cap'].result.supported_execution_state_semantics) === '["live_registry"]',
  'snapshot mode implied but unsupported');
assert(wById['e'].result.execution_state_semantics === 'live_registry', 'explain hides state semantics');
// #04: run carries registry_revision, per-layer weight_revision, state_digest.
const es1 = wById['r1'].result.execution_state;
assert(wById['r1'].ok === true && JSON.stringify(wById['r1'].result.values) === '[0,0]', 'baseline run failed');
assert(es1.execution_state_semantics === 'live_registry', 'run hides state semantics');
assert(es1.registry_revision === 0 && JSON.stringify(es1.layer_weight_revisions) === '[0,0,0,0]',
  'initial revisions not zero');
assert(/^sha256:[0-9a-f]{64}$/.test(es1.state_digest), 'state_digest malformed');
// setWeights bumps the global and per-layer revision and re-digests.
const swState = wById['sw'].result.execution_state;
assert(wById['sw'].ok === true, 'setWeights failed');
assert(swState.registry_revision === 1 && JSON.stringify(swState.layer_weight_revisions) === '[1,0,0,0]',
  'setWeights did not bump revisions');
assert(swState.state_digest !== es1.state_digest, 'setWeights did not change state_digest');
// #04 done-criteria: outputs coincide ([0,0] clamped) yet revision/digest differ.
const es2 = wById['r2'].result.execution_state;
assert(wById['r2'].ok === true && JSON.stringify(wById['r2'].result.values) === '[0,0]',
  'post-mutation run changed numerics unexpectedly');
assert(es2.registry_revision === 1 && es2.state_digest === swState.state_digest
  && es2.state_digest !== es1.state_digest,
  'run after mutation did not expose the new revision/digest despite identical output');
// #05 live proof: the compiled graph reads the CURRENT registry weights,
// not a compile-time snapshot (ones on layer 0, zeros on layer 1, in=[2,3]).
assert(wById['r3'].ok === true && JSON.stringify(wById['r3'].result.values) === '[5,5]',
  'compiled graph did not read live mutated weights');
assert(wById['tr'].result.execution_state?.execution_state_semantics === 'live_registry',
  'trace hides state semantics');
// #05: requesting snapshot at create is rejected explicitly, never silent.
const snapshotCase = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const cases = [
    {op: 'create', request_id: 'snap', numSlots: 1, layers: [{constructor: 'relu', args: [7]}],
      steps: [], outputSlot: 0, ports: [], execution_state_semantics: 'snapshot'},
    {op: 'create', request_id: 'live', numSlots: 3, layers: [{constructor: 'add', args: [31]}],
      steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2,
      ports: [0, 1].map(slot => ({slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'})),
      execution_state_semantics: 'live_registry'},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const sById = Object.fromEntries(snapshotCase.map(r => [r.request_id, r]));
assert(sById['snap'].ok === false && sById['snap'].error_envelope?.code === 'invalid_argument'
  && sById['snap'].error_envelope?.phase === 'create'
  && sById['snap'].error_envelope?.execution_started === false,
  'snapshot semantics not rejected explicitly');
assert(sById['live'].ok === true, 'explicit live_registry create rejected');
// setWeights rejects honestly: wrong count, weightless layer, unknown index.
for (const id of ['badcount', 'weightless', 'badlayer']) {
  assert(wById[id].ok === false && wById[id].error_envelope?.code === 'invalid_argument'
    && wById[id].error_envelope?.phase === 'setWeights'
    && wById[id].error_envelope?.execution_started === false, `setWeights misclassified ${id}`);
}
// ---- Fase 5 (#07, #08, #09): vocabulary, plan constraints, uniform verification summary ----
// #07: roles/layouts are discoverable from capabilities without extracting
// strings from the binary; invalid values fail as invalid_enum naming the
// actual value, the valid choices, and a capability_ref. The x- extension
// namespace keeps working. #08: one-port and unused-port plans fail with
// DIFFERENT codes; validatePlan reports declared/consumed/unused/minimum
// without creating a session. #09: every verify outcome carries a uniform
// top-level verification_summary; the WASM detail structure is preserved.
const vocabCase = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const goodPorts = [0, 1].map(slot => ({slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}));
  const badRolePorts = [{slot: 0, role: 'input', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}, goodPorts[1]];
  const badLayoutPorts = [{slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'nchw'}, goodPorts[1]];
  const relationalPorts = [{slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'preserve_input'}, goodPorts[1]];
  const extRolePorts = [{slot: 0, role: 'x-custom-feed', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}, goodPorts[1]];
  const onePort = [{slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}];
  const unusedPort = [goodPorts[0], {slot: 3, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}];
  const binarySpec = {numSlots: 3, layers: [{constructor: 'add', args: [31]}],
    steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2};
  const unarySpec = {numSlots: 4, layers: [{constructor: 'relu', args: [31]}],
    steps: [{kind: 'unary', layer: 0, slots: [0, 2]}], outputSlot: 2};
  const cases = [
    {op: 'capabilities', request_id: 'cap'},
    {op: 'create', request_id: 'badrole', ...binarySpec, ports: badRolePorts},
    {op: 'create', request_id: 'badlayout', ...binarySpec, ports: badLayoutPorts},
    {op: 'create', request_id: 'relational', ...binarySpec, ports: relationalPorts},
    {op: 'create', request_id: 'extrole', ...binarySpec, ports: extRolePorts},
    {op: 'validatePlan', request_id: 'vp_ok', ...binarySpec, ports: [{slot: 0}, {slot: 1}]},
    {op: 'validatePlan', request_id: 'vp_one', numSlots: 2, layers: [], steps: [{kind: 'unary', layer: 0, slots: [0, 1]}],
      outputSlot: 1, ports: [{slot: 0}]},
    {op: 'validatePlan', request_id: 'vp_unused', ...unarySpec, ports: [{slot: 0}, {slot: 3}]},
    {op: 'create', request_id: 'cone', numSlots: 2, layers: [{constructor: 'relu', args: [31]}],
      steps: [{kind: 'unary', layer: 0, slots: [0, 1]}], outputSlot: 1, ports: onePort},
    {op: 'create', request_id: 'cunused', ...unarySpec, ports: unusedPort},
    // Read-after-write is NOT consumption: slot 2 is written by step 0, then read.
    {op: 'validatePlan', request_id: 'vp_raw', numSlots: 4,
      steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}, {kind: 'unary', layer: 0, slots: [2, 3]}],
      outputSlot: 3, ports: [{slot: 0}, {slot: 1}, {slot: 2}]},
    {op: 'create', request_id: 'c', ...binarySpec, ports: goodPorts,
      logicalPorts: [{id: 'a', slot: 0, source: 't'}, {id: 'b', slot: 1, source: 't'}]},
    {op: 'bind', request_id: 'b0', slot: 0, values: [1, 2], shape: [1, 2, 1, 1],
      role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 1},
    {op: 'bind', request_id: 'b1', slot: 1, values: [3, 4], shape: [1, 2, 1, 1],
      role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 1},
    {op: 'verify', request_id: 'vpass', candidate: [4, 6]},
    {op: 'verify', request_id: 'vfail', candidate: [4, 9]},
    {op: 'verify', request_id: 'vshape', candidate: [4, 6, 8]},
    {op: 'verify', request_id: 'vempty', candidate: []},
    {op: 'clear', request_id: 'cl', slot: 0},
    {op: 'bind', request_id: 'bbadrole', slot: 0, values: [5, 6], shape: [1, 2, 1, 1],
      role: 'bogus', layout: 'feature_axis1_singleton', source: 't', revision: 2},
    {op: 'defer', request_id: 'dbadrole', id: 'x', role: 'nope', required: false},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const f5ById = Object.fromEntries(vocabCase.map(r => [r.request_id, r]));
// #07: vocabulary is documented in capabilities.
const vocab = f5ById['cap'].result.vocabulary;
assert(vocab?.schema === 'burn-research.host-vocabulary.v1', 'vocabulary document missing from capabilities');
const roleNames = vocab.input_roles.map(r => r.canonical_name);
for (const name of ['observation', 'state', 'feature'])
  assert(roleNames.includes(name), `canonical role ${name} not documented`);
assert(vocab.input_roles.every(r => Array.isArray(r.aliases) && typeof r.constraints === 'string' && typeof r.since_version === 'string'),
  'role entries lack aliases/constraints/since_version');
const layoutNames = vocab.layouts.map(l => l.canonical_name);
for (const name of ['feature_axis1_singleton', 'channel_first', 'unknown'])
  assert(layoutNames.includes(name), `layout ${name} not documented`);
// #07: invalid_enum names the actual value, the valid choices, and a capability_ref.
const badRole = f5ById['badrole'];
assert(badRole.ok === false && badRole.error_envelope?.code === 'invalid_enum'
  && badRole.error_envelope?.phase === 'create'
  && badRole.error_envelope?.actual_value === 'input'
  && badRole.error_envelope?.allowed_values?.includes('observation')
  && badRole.error_envelope?.capability_ref === 'capabilities.vocabulary'
  && badRole.error_envelope?.execution_started === false,
  'invalid role not classified as invalid_enum with actual value and choices');
const badLayout = f5ById['badlayout'];
assert(badLayout.ok === false && badLayout.error_envelope?.code === 'invalid_enum'
  && badLayout.error_envelope?.actual_value === 'nchw'
  && badLayout.error_envelope?.allowed_values?.includes('feature_axis1_singleton'),
  'invalid layout not classified as invalid_enum with actual value and choices');
assert(f5ById['relational'].error_envelope?.code === 'invalid_enum'
  && typeof f5ById['relational'].error_envelope?.rejection_reason === 'string',
  'relational layout rejection lacks a reason');
assert(f5ById['extrole'].ok === true, 'x- extension role rejected');
// bind and defer enforce the same vocabulary.
assert(f5ById['bbadrole'].ok === false && f5ById['bbadrole'].error_envelope?.code === 'invalid_enum'
  && f5ById['bbadrole'].error_envelope?.actual_value === 'bogus', 'bind skipped vocabulary validation');
assert(f5ById['dbadrole'].ok === false && f5ById['dbadrole'].error_envelope?.code === 'invalid_enum',
  'defer skipped vocabulary validation');
// #08: plan_constraints disclosed; validatePlan reports the port analysis.
const pc = f5ById['cap'].result.plan_constraints;
assert(pc?.minimum_ports === 2 && pc?.minimum_ports_code === 'plan_minimum_ports'
  && pc?.unused_port_code === 'plan_unused_ports', 'plan_constraints missing from capabilities');
const vpOk = f5ById['vp_ok'].result;
assert(vpOk.valid === true && vpOk.code === 'ok' && vpOk.minimum_ports === 2
  && JSON.stringify(vpOk.declared_ports) === '[0,1]'
  && JSON.stringify(vpOk.consumed_ports) === '[0,1]'
  && JSON.stringify(vpOk.unused_ports) === '[]',
  'validatePlan did not report all ports consumed on a valid graph');
const vpOne = f5ById['vp_one'].result;
assert(vpOne.valid === false && vpOne.code === 'plan_minimum_ports', 'validatePlan missed the minimum-ports case');
const vpUnused = f5ById['vp_unused'].result;
assert(vpUnused.valid === false && vpUnused.code === 'plan_unused_ports'
  && JSON.stringify(vpUnused.unused_ports) === '[3]'
  && JSON.stringify(vpUnused.consumed_ports) === '[0]', 'validatePlan misreported unused ports');
// Read-after-write does not count as consumption (mirrors the WASM rule).
const vpRaw = f5ById['vp_raw'].result;
assert(vpRaw.valid === false && vpRaw.code === 'plan_unused_ports'
  && JSON.stringify(vpRaw.unused_ports) === '[2]', 'validatePlan misclassified read-after-write');
// One-port and unused-port creates fail with DIFFERENT codes.
assert(f5ById['cone'].ok === false && f5ById['cone'].error_envelope?.code === 'plan_minimum_ports'
  && f5ById['cone'].error_envelope?.execution_started === false, 'one-port create misclassified');
const cUnused = f5ById['cunused'];
assert(cUnused.ok === false && cUnused.error_envelope?.code === 'plan_unused_ports'
  && cUnused.error_envelope?.code !== f5ById['cone'].error_envelope?.code
  && JSON.stringify(cUnused.error_envelope?.unused_ports) === '[3]',
  'unused-port create did not fail with a distinct code');
// #09: uniform verification_summary on every outcome; WASM detail preserved.
const sPass = f5ById['vpass'].result.verification_summary;
assert(f5ById['vpass'].ok === true && sPass.verdict === 'pass' && sPass.reason === null
  && sPass.shape_matches === true && sPass.compared_elements === 2
  && sPass.max_abs_error === 0 && sPass.first_failure === null,
  'verify pass summary malformed');
assert(f5ById['vpass'].result.reference?.verification?.passed === true, 'verify detail structure changed');
const sFail = f5ById['vfail'].result.verification_summary;
assert(sFail.verdict === 'fail' && sFail.reason === 'tolerance_exceeded' && sFail.shape_matches === true
  && sFail.compared_elements === 2 && sFail.first_failure === 1, 'verify mismatch summary malformed');
for (const id of ['vshape', 'vempty']) {
  const s = f5ById[id].result.verification_summary;
  assert(f5ById[id].ok === true, `${id}: shape mismatch must be a verdict, not an op failure`);
  assert(s.verdict === 'fail' && s.reason === 'shape_mismatch' && s.shape_matches === false
    && s.compared_elements === null && s.max_abs_error === null && s.max_rel_error === null
    && s.first_failure === null, `${id}: shape_mismatch summary malformed`);
}
console.log(JSON.stringify({verdict:'PASS',protocol:'JSON Lines',explain:responses[1].result.static_shape_status,rejected_source:responses[4].result.status.ports[1].status,reference:responses[9].result.values,verified:responses[10].result.reference.verification.passed}));

// ---- Fase 6 (#06, #10, #11, #12) regression: identity, gate, fingerprint, integrity ----
const f6Case = await new Promise((resolve, reject) => {
  const child = spawn(process.execPath, [path.join(packageDir, 'interactive_multi_input_ingress.mjs')], {stdio: ['pipe', 'pipe', 'pipe']});
  const goodPorts = [0, 1].map(slot => ({slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton'}));
  const cases = [
    {op: 'capabilities', request_id: 'cap'},
    // #10: no field named authorized may exist on any output surface.
    {op: 'create', request_id: 'c', numSlots: 3, layers: [{constructor: 'add', args: [31]}],
      steps: [{kind: 'binary', layer: 0, slots: [0, 1, 2]}], outputSlot: 2, ports: goodPorts,
      logicalPorts: [{id: 'a', slot: 0, source: 't'}, {id: 'b', slot: 1, source: 't'}]},
    // #11: caller-declared mode accepts an empty fingerprint.
    {op: 'bind', request_id: 'bempty', slot: 0, values: [1, 2], shape: [1, 2, 1, 1],
      role: 'observation', layout: 'feature_axis1_singleton', source: 't', revision: 1, fingerprint: ''},
    // #12: the integrity helper is session-free.
    {op: 'verifyCheckpointIntegrity', request_id: 'vok',
      bundle_f32le_base64: Buffer.from('burn-research-synthetic-bundle').toString('base64'),
      expected_digest: `sha256:${createHash('sha256').update('burn-research-synthetic-bundle').digest('hex')}`},
    {op: 'verifyCheckpointIntegrity', request_id: 'vbad',
      bundle_f32le_base64: Buffer.from('burn-research-synthetic-bundle').toString('base64'),
      expected_digest: `sha256:${'0'.repeat(64)}`},
    {op: 'verifyCheckpointIntegrity', request_id: 'vfmt',
      bundle_f32le_base64: Buffer.from('x').toString('base64'), expected_digest: 'not-a-digest'},
    // #12: canonical base64 is enforced — non-canonical spellings are
    // rejected even when they would decode to bytes.
    {op: 'verifyCheckpointIntegrity', request_id: 'vpad',
      bundle_f32le_base64: Buffer.from('burn-research-synthetic-bundle!').toString('base64').replace(/=+$/, ''),
      expected_digest: `sha256:${createHash('sha256').update('burn-research-synthetic-bundle!').digest('hex')}`},
    {op: 'verifyCheckpointIntegrity', request_id: 'vws',
      bundle_f32le_base64: Buffer.from('white space').toString('base64').replace(/(.{4})/, '$1\n'),
      expected_digest: `sha256:${createHash('sha256').update('white space').digest('hex')}`},
    {op: 'verifyCheckpointIntegrity', request_id: 'vjunk',
      bundle_f32le_base64: '!!!not-base64!!!',
      expected_digest: `sha256:${createHash('sha256').update('junk').digest('hex')}`},
    {op: 'close', request_id: 'done'},
  ];
  let stdout = '';
  child.stdout.on('data', chunk => { stdout += chunk; });
  child.on('error', reject);
  child.on('close', () => resolve(stdout.trim().split('\n').map(line => JSON.parse(line))));
  child.stdin.write(cases.map(c => JSON.stringify(c)).join('\n') + '\n');
  child.stdin.end();
});
const f6ById = Object.fromEntries(f6Case.map(r => [r.request_id, r]));
// #06: the layer-type catalog names every protocol type once; Linear, ReLU and
// Add resolve to the same canonical names the explain/trace surfaces report.
const catalog = f6ById['cap'].result.layer_types;
assert(catalog?.schema === 'burn-research.host-layer-type-catalog.v1' && catalog.types.length === 11,
  'layer type catalog missing or incomplete');
const byCode = Object.fromEntries(catalog.types.map(t => [t.code, t]));
for (const [code, name, hex] of [[1, 'linear', '01'], [4, 'activation', '04'], [19, 'binary', '13']])
  assert(byCode[code]?.name === name && byCode[code]?.code_hex === hex
    && byCode[code]?.constructors?.length > 0, `layer type ${code} misnamed in catalog`);
assert(byCode[1].constructors.includes('linear') && byCode[4].constructors.includes('relu')
  && byCode[19].constructors.includes('add'), 'canonical constructors missing from catalog');
// #06: explain on the weight graph resolves decimal step codes AND hex
// fingerprint spellings to the same names (dynamic, not table-trusting).
const wExplain = wById['e'].result;
const wNames = Object.fromEntries(wExplain.steps.map(s => [s.layer_type, s.layer_type_name]));
assert(wNames[1] === 'linear' && wNames[19] === 'binary' && wNames[4] === 'activation',
  'explain steps do not carry canonical layer type names');
assert(wExplain.steps.every(s => s.layer_type_code === s.layer_type && s.layer_type_encoding === 'decimal'),
  'explain step identity fields malformed');
const wFp = wExplain.program_identity.layer_type_identities;
assert(Array.isArray(wFp) && wFp.length === wExplain.program_identity.layer_init_fingerprints.length
  && wFp.every(f => f.layer_type_encoding === 'hex'
    && f.layer_type_name === wNames[f.layer_type_code]), 'explain fingerprint identities inconsistent with steps');
// #10: ready is the documented gate; execution_authorized is gone everywhere.
assert(!('execution_authorized' in f6ById['c'].result.status), 'status blob still carries execution_authorized');
const f6hp = f6ById['c'].result.host_provenance;
assert(f6hp.authority_mode === 'caller_declared' && !('execution_authorized' in f6hp)
  && f6hp.execution_gate?.field === 'ready' && f6hp.host_authority_attested === false
  && f6hp.denial_reason === null && typeof f6hp.enforcement_effect === 'string',
  'caller_declared host_provenance gate fields malformed');
const rs = f6ById['cap'].result.runtime_semantics;
assert(rs?.execution_gate?.field === 'ready'
  && typeof rs.execution_gate.supersedes === 'string'
  && rs.execution_gate.supersedes.includes('execution_authorized'),
  'execution gate not documented in capabilities');
assert(JSON.stringify(rs.authority_modes) === '["caller_declared","ed25519_host_enforced","ed25519_host_durable"]',
  'authority modes not disclosed');
// R-20 (#16): session-free capabilities host_provenance carries the same canonical
// execution gate as the session surface (field, semantics, supersedes, enforcement_effect).
const capHp = f6ById['cap'].result.host_provenance;
assert(capHp?.execution_gate?.field === 'ready'
  && capHp.execution_gate.meaning === rs.execution_gate.meaning
  && capHp.execution_gate.supersedes === rs.execution_gate.supersedes
  && capHp.enforcement_effect === f6hp.enforcement_effect,
  'session-free capabilities host_provenance does not carry the canonical execution gate');
// #11: fingerprint contract is explicit per mode; caller-declared accepts empty.
const fpReq = rs.fingerprint_requirement?.by_authority_mode;
assert(fpReq?.caller_declared?.fingerprint_required === false
  && fpReq?.ed25519_host_enforced?.fingerprint_required === true
  && fpReq?.ed25519_host_enforced?.path === 'fingerprint'
  && fpReq?.ed25519_host_enforced?.constraint === 'non_empty'
  && fpReq?.ed25519_host_durable?.constraint === 'non_empty',
  'fingerprint requirement not disclosed per authority mode');
assert(f6ById['bempty'].ok === true, 'caller-declared bind rejected an empty fingerprint');
// #12: helper verdicts; corruption at EVERY byte position (incl. the last) is
// detected, unlike the structural bundle parser.
assert(f6ById['vok'].ok === true && f6ById['vok'].result.integrity_ok === true
  && typeof f6ById['vok'].result.actual_digest === 'string', 'integrity helper rejected a valid bundle');
assert(f6ById['vbad'].ok === true && f6ById['vbad'].result.integrity_ok === false
  && f6ById['vbad'].result.actual_digest !== f6ById['vbad'].result.expected_digest,
  'integrity helper accepted a tampered bundle');
assert(f6ById['vfmt'].ok === false && f6ById['vfmt'].error_envelope?.code === 'invalid_argument'
  && f6ById['vfmt'].error_envelope?.path === 'expected_digest', 'bad digest format not rejected');
// #12: non-canonical base64 spellings are rejected at the security boundary,
// even when a lenient decoder would still produce bytes.
{
  // Self-validating fixture: the vpad input must actually carry padding,
  // otherwise stripping it would test nothing.
  const paddedFixture = Buffer.from('burn-research-synthetic-bundle!').toString('base64');
  assert(paddedFixture.endsWith('=') && paddedFixture.replace(/=+$/, '').length % 4 !== 0,
    'vpad fixture lost its base64 padding');
}
for (const id of ['vpad', 'vws', 'vjunk']) {
  const envelope = f6ById[id].error_envelope;
  assert(f6ById[id].ok === false && envelope?.code === 'invalid_argument'
    && envelope?.path === 'bundle_f32le_base64' && envelope?.constraint === 'canonical_base64',
    `${id}: non-canonical base64 not rejected with the canonical_base64 constraint`);
}
{
  const raw = Buffer.from('burn-research-synthetic-bundle');
  const good = `sha256:${createHash('sha256').update(raw).digest('hex')}`;
  for (let i = 0; i < raw.length; i++) {
    const tampered = Buffer.from(raw);
    tampered[i] ^= 0x01;
    const digest = `sha256:${createHash('sha256').update(tampered).digest('hex')}`;
    assert(digest !== good, `single-bit flip at byte ${i} did not change the digest`);
  }
}
assert(rs.checkpoint_integrity?.helper === 'verifyCheckpointIntegrity'
  && rs.checkpoint_integrity?.byte_level_integrity === 'host_responsibility',
  'checkpoint integrity responsibility not documented');
console.log(JSON.stringify({verdict:'PASS',fase:6,checks:['layer-identity','execution-gate','fingerprint-contract','checkpoint-integrity']}));
