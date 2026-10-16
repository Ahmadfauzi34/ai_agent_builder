#!/usr/bin/env node
// burn-math run: validate -> create -> setWeights -> bind -> run, one shot.
// Usage: node run.mjs <spec.json>
// Prints {"ok":true,"outputs":{"shape":[...],"values":[...]},"execution_gate":{...},
// "host_provenance":{...}} on success. The plan is validated BEFORE any
// session is created, so spec errors surface as structured, self-correcting
// messages (codes plan_minimum_ports / plan_unused_ports / invalid_enum with
// remediation hints) instead of mid-run failures.
import { startRunner, unwrap, printJson, failCli } from '../lib/runner.mjs';
import { loadSpec, planFragment, portBySlot } from '../lib/spec.mjs';

const [specFile] = process.argv.slice(2);
if (!specFile) { console.error('usage: node run.mjs <spec.json>'); process.exit(2); }

let spec;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }

const { send, stop } = startRunner();
async function shutdown() { try { await stop(); } catch { /* ignore */ } }

try {
  // 1. Fail fast on plan problems (session-free).
  const plan = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
  if (!plan.valid) {
    await shutdown();
    printJson({ ok: false, stage: 'validatePlan', ...plan });
    process.exit(2);
  }

  // 2. Create the session (logicalPorts maps each declared slot to a runtime
  //    source; the runner requires this mapping before run).
  const logicalPorts = spec.ports.map((p, i) => ({
    id: `in${i}`, slot: p.slot, source: 'burn-math-skill',
  }));
  const created = unwrap(await send({
    op: 'create',
    layers: spec.layers,
    steps: spec.steps,
    numSlots: spec.numSlots,
    outputSlot: spec.outputSlot,
    ports: spec.ports,
    logicalPorts,
  }));
  void created;

  // 3. Weights (optional; unset layers run on deterministic zero-init).
  for (const [indexStr, values] of Object.entries(spec.weights)) {
    unwrap(await send({ op: 'setWeights', layer: Number(indexStr), values }));
  }

  // 4. Bind inputs using each port's declared shape/role/layout.
  const ports = portBySlot(spec);
  for (const [slotStr, values] of Object.entries(spec.inputs)) {
    const slot = Number(slotStr);
    const port = ports.get(slot);
    if (!port) throw new Error(`input for slot ${slot} has no declared port`);
    unwrap(await send({
      op: 'bind', slot, values,
      shape: port.shape, role: port.role, layout: port.layout,
      source: 'burn-math-skill', revision: 1, fingerprint: '',
    }));
  }

  // 5. Run.
  const ran = unwrap(await send({ op: 'run' }));
  await shutdown();
  printJson({
    ok: true,
    outputs: { shape: ran.shape, values: ran.values },
    execution_gate: ran.host_provenance?.execution_gate ?? null,
    ready: ran.host_provenance?.ready ?? null,
    host_provenance: ran.host_provenance ?? null,
  });
} catch (e) {
  await shutdown();
  // Structured engine errors already carry code/phase/remediation_hint.
  failCli(e.message, e.envelope ?? null);
}
