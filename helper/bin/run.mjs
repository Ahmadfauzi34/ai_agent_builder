#!/usr/bin/env node
// burn-math run: validate -> create -> setWeights -> bind -> run, one shot.
// Usage: node run.mjs [--direct] <spec.json> [<spec2.json> ...]
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
// MIXED-DTYPE SPECS (Option B): a spec with any int step runs as
// segments against one shared registry via the direct primitives
// (lib/mixed.mjs), NOT through an ingress session — one graph per
// session cannot express a split spec. Its success record is the third
// output shape: execution_gate/host_provenance null (like --direct)
// PLUS segments[] and boundary_casts[] (the host-side receipt).
// Build errors (dtype mismatch, i64 output slot) exit 2 like other
// spec-validation failures; value-dependent failures (boundary,
// engine) exit 1 like other run failures. --direct refuses mixed
// specs in v1 (named error, run without --direct instead).
//
// ISOLATION (#6): every spec gets a FRESH `create` (new LayerRegistry,
// fresh layers/proofs/handoffs). The runner process holds no other mutable
// global state, so batch item N cannot observe item N-1. Proven by the
// batch-isolation check: batch [A, B] outputs == standalone A, B outputs.
//
// --direct: same engine, same values, in-process — no child process, no
// IPC round-trips, no ingress session. loadWasm once, then one fresh
// buildGraph per spec inside its own withGraphScope arena (freed even
// on throw), so batch isolation holds in-process too. Measured
// (linear-relu, this machine): single ~3-4x faster than the runner path,
// batch of 10 ~8x; steady-state per-spec ~1ms once wasm is warm.
// TRADE-OFF (explicit, not hidden): --direct produces NO execution_gate
// or host_provenance — those are ingress-session artifacts (proofs,
// ledger, receipts). Its records keep the same keys with null plus
// mode:'direct', so consumers never receive a fake gate.
import { startRunner, unwrap, printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';
import { loadSpec, planFragment, portBySlot, checkArgValues, analyzePlanPorts } from '../lib/spec.mjs';
import { loadWasm, withGraphScope } from '../lib/wasm_direct.mjs';
import { specUsesInt, splitSpec, runMixed } from '../lib/mixed.mjs';

const rawArgs = process.argv.slice(2);
// REG-08: --help/-h in ANY position is usage, never a run (a spec file
// literally named --help does not exist; the old code tried to load it
// and exited 1, which looked like a tool failure, not usage).
if (!rawArgs.length || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node run.mjs [--direct] <spec.json> [<spec2.json> ...]');
  process.exit(2);
}
const direct = rawArgs.includes('--direct');
const specFiles = rawArgs.filter((a) => a !== '--direct');
if (!specFiles.length) {
  console.error('usage: node run.mjs [--direct] <spec.json> [<spec2.json> ...]');
  process.exit(2);
}

// Run one spec against an already-running runner. Never throws: failures
// become {ok:false} records so the batch can continue.
// Mixed-dtype specs (any step uses an int constructor) do NOT go through
// the runner: ingress sessions are one-graph-per-session and cannot
// express a split spec. They run in-process via the direct primitives
// (lib/mixed.mjs) with execution_gate/host_provenance null — the third
// output shape — plus the segment receipt. The wasm runtime is loaded
// lazily (getRt) so pure-f32 batches never pay for it.
async function runOne(send, specFile, getRt) {
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
  if (specUsesInt(spec)) {
    // Plan-shape gate first (same check the direct path runs
    // in-process); dtype/build errors follow, before any segment runs.
    const plan = analyzePlanPorts(spec);
    if (!plan.valid) {
      return { ok: false, stage: 'validatePlan', ...plan, error_envelope: null };
    }
    let segments;
    let slotDtype;
    try {
      ({ segments, slotDtype } = splitSpec(spec));
    } catch (e) {
      return { ok: false, stage: 'build', error: e.message, error_envelope: null };
    }
    if (slotDtype.get(spec.outputSlot) !== 'f32') {
      return {
        ok: false, stage: 'build',
        error: `output slot ${spec.outputSlot} dtype ${slotDtype.get(spec.outputSlot) ?? 'unknown'}; end with intToFloat or use int.mjs`,
        error_envelope: null,
      };
    }
    try {
      const rt = await getRt();
      const mixed = runMixed(rt, spec, segments, portBySlot(spec));
      return {
        ok: true,
        outputs: mixed.outputs,
        execution_gate: null,
        ready: null,
        host_provenance: null,
        segments: mixed.segments,
        boundary_casts: mixed.boundary_casts,
      };
    } catch (e) {
      // Engine throws can be bare strings (plan/inputContract errors),
      // not Errors — normalize so the record never carries "undefined".
      const msg = e instanceof Error ? e.message : String(e);
      return { ok: false, stage: 'run', error: msg, error_envelope: e?.envelope ?? null };
    }
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

// Direct run of one spec: same stages and record shapes as runOne,
// but in-process. Never throws: failures become {ok:false} records so
// the batch can continue. The scope frees reg/plan/g/bundle even when
// the run throws. buildGraph itself is exception-safe since 2026-10-30,
// so a failed construction leaks nothing either.
async function runDirectOne(rt, specFile) {
  let spec;
  try {
    spec = loadSpec(specFile);
  } catch (e) {
    return { ok: false, stage: 'loadSpec', error: e.message, error_envelope: null };
  }
  const argIssues = checkArgValues(spec);
  if (argIssues.length) {
    return { ok: false, stage: 'argValues', code: 'invalid_argument', issues: argIssues, error_envelope: null };
  }
  const plan = analyzePlanPorts(spec);
  if (!plan.valid) {
    return { ok: false, stage: 'validatePlan', ...plan, error_envelope: null };
  }
  // v1 scope (fork F2): --direct runs pure-f32 specs only. Mixed specs
  // need the segment receipt, which lives on the default path's
  // third output shape; refusing here beats a builder rejection with
  // no path forward.
  if (specUsesInt(spec)) {
    return {
      ok: false, stage: 'build',
      error: 'mixed-dtype spec: --direct does not support specs with int steps in v1; run without --direct (the default path runs mixed specs with a segment receipt)',
      error_envelope: null,
    };
  }
  // Phase is tracked by hand: bundle creation (bind) vs graph run.
  // Raw wasm throws carry neither code nor phase; the runner wraps
  // those as internal_error, so direct does the same — the phase is
  // the step we were actually in, never invented detail.
  let phase = 'bind';
  try {
    const outputs = await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
      const b = bundle(spec.inputs, portBySlot(spec));
      phase = 'run';
      const out = g.run(reg, b);
      const values = Array.from(out.to_array());
      const shape = Array.from(out.shape());
      out.free();
      return { shape, values };
    });
    return {
      ok: true,
      outputs,
      mode: 'direct',
      execution_gate: null,
      ready: null,
      host_provenance: null,
    };
  } catch (e) {
    const envelope = {
      code: e.code ?? 'internal_error',
      phase: e.phase ?? phase,
      message: e.message,
    };
    return { ok: false, stage: 'run', error: e.message, error_envelope: envelope };
  }
}

let results;
if (direct) {
  let rt;
  try {
    rt = await loadWasm();
  } catch (e) {
    failCli(e instanceof Error ? e.message : String(e));
  }
  results = [];
  for (const f of specFiles) {
    results.push(await runDirectOne(rt, f));
  }
} else {
  const { send, stop } = startRunner();
  async function shutdown() { try { await stop(); } catch { /* ignore */ } }
  // Lazy wasm runtime: only mixed specs need it on this path.
  let rt = null;
  const getRt = async () => {
    if (!rt) rt = await loadWasm();
    return rt;
  };
  results = [];
  try {
    for (const f of specFiles) {
      results.push(await runOne(send, f, getRt));
    }
  } finally {
    await shutdown();
  }
}

// Drain-safe prints use the shared looped-sync printJsonAndExit from
// lib/runner.mjs (stronger than the drain-wait version this file used to
// define locally: proven on 1.3MB piped batch output).
// Legacy: exactly one spec -> the historical single-object shape and the
// exit codes: 2 = plan/arg/build validation failure (validatePlan,
// argValues, build), 1 = tool-level load failure and run-phase failure.
// (Ported from pack v2.4: the old comment claimed load failures exit 2;
// they exit 1 via failCli's default — the comment was wrong, not the
// code.)
if (results.length === 1) {
  const r = results[0];
  if (!r.ok) {
    if (r.stage === 'loadSpec') failCli(r.error); // exit 1 (tool-level)
    if (r.stage === 'validatePlan' || r.stage === 'argValues' || r.stage === 'build') printJsonAndExit(r, 2); // exit 2, as before
    failCli(r.error, r.error_envelope ?? null); // exit 1, as before
  }
  printJson(r);
} else {
  const out = specFiles.map((f, i) => ({ spec: f, ...results[i] }));
  if (results.some((r) => !r.ok)) printJsonAndExit(out, 1);
  printJson(out);
}
