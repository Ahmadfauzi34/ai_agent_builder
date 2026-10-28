#!/usr/bin/env node
// burn-math run: validate -> create -> setWeights -> bind -> run, one shot.
// Usage: node run.mjs <spec.json> [<spec2.json> ...]
//   One spec  -> prints the single result object (legacy shape, unchanged).
//   N specs   -> ONE runner process handles all N (batch mode, #6 option B):
//                prints [{spec, ...result}, ...]. A failing spec records
//                {ok:false} and the batch continues with the rest.
//
// Prints {"ok":true,"outputs":{"shape":[...],"values":[...]},"execution_gate":{...},
// "host_provenance":{...}} on success. The plan is validated BEFORE any
// session is created, so spec errors surface as structured, self-correcting
// messages (codes plan_minimum_ports / plan_unused_ports / invalid_enum with
// remediation hints) instead of mid-run failures.
//
// ISOLATION (#6): every spec gets a FRESH `create` (new LayerRegistry,
// fresh layers/proofs/handoffs). The runner process holds no other mutable
// global state, so batch item N cannot observe item N-1. Proven by the
// batch-isolation check: batch [A, B] outputs == standalone A, B outputs.
import { startRunner, unwrap, printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';
import { loadSpec, planFragment, portBySlot, checkArgValues } from '../lib/spec.mjs';

const specFiles = process.argv.slice(2);
// REG-08: --help/-h in ANY position is usage, never a run (a spec file
// literally named --help does not exist; the old code tried to load it
// and exited 1, which looked like a tool failure, not usage).
if (!specFiles.length || specFiles.includes('--help') || specFiles.includes('-h')) {
  console.error('usage: node run.mjs <spec.json> [<spec2.json> ...]');
  process.exit(2);
}

// Run one spec against an already-running runner. Never throws: failures
// become {ok:false} records so the batch can continue.
async function runOne(send, specFile) {
  let spec;
  try {
    spec = loadSpec(specFile);
  } catch (e) {
    return { ok: false, stage: 'loadSpec', error: e.message, error_envelope: null };
  }
  // Ported from pack v2.4 (Audit_7 E4): strict arg/value typing before
  // any engine call. Without this, a string bias flag was silently
  // coerced and returned ok:true.
  const argIssues = checkArgValues(spec);
  if (argIssues.length) {
    return { ok: false, stage: 'argValues', code: 'invalid_argument', issues: argIssues, error_envelope: null };
  }
  try {
    // 1. Fail fast on plan problems (session-free).
    const plan = unwrap(await send({ op: 'validatePlan', ...planFragment(spec) }));
    if (!plan.valid) {
      return { ok: false, stage: 'validatePlan', ...plan, error_envelope: null };
    }

    // 2. Create the session (logicalPorts maps each declared slot to a runtime
    //    source; the runner requires this mapping before run). Fresh per spec.
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
    return {
      ok: true,
      outputs: { shape: ran.shape, values: ran.values },
      execution_gate: ran.host_provenance?.execution_gate ?? null,
      ready: ran.host_provenance?.ready ?? null,
      host_provenance: ran.host_provenance ?? null,
    };
  } catch (e) {
    // Structured engine errors already carry code/phase/remediation_hint.
    // Ported from pack v2.4 (Audit_7 A1): batch records used to call
    // this field `envelope` while single-spec mode prints
    // `error_envelope` — same field, same name, both modes.
    return { ok: false, stage: 'run', error: e.message, error_envelope: e.envelope ?? null };
  }
}

const { send, stop } = startRunner();
async function shutdown() { try { await stop(); } catch { /* ignore */ } }

const results = [];
try {
  for (const f of specFiles) {
    results.push(await runOne(send, f));
  }
} finally {
  await shutdown();
}

// Drain-safe prints use the shared looped-sync printJsonAndExit from
// lib/runner.mjs (stronger than the drain-wait version this file used to
// define locally: proven on 1.3MB piped batch output).
// Legacy: exactly one spec -> the historical single-object shape and the
// exit codes: 2 = plan/arg validation failure (validatePlan, argValues),
// 1 = tool-level load failure and run-phase failure. (Ported from pack
// v2.4: the old comment claimed load failures exit 2; they exit 1 via
// failCli's default — the comment was wrong, not the code.)
if (results.length === 1) {
  const r = results[0];
  if (!r.ok) {
    if (r.stage === 'loadSpec') failCli(r.error); // exit 1 (tool-level)
    if (r.stage === 'validatePlan' || r.stage === 'argValues') printJsonAndExit(r, 2); // exit 2, as before
    failCli(r.error, r.error_envelope ?? null); // exit 1, as before
  }
  printJson(r);
} else {
  const out = specFiles.map((f, i) => ({ spec: f, ...results[i] }));
  if (results.some((r) => !r.ok)) printJsonAndExit(out, 1);
  printJson(out);
}
