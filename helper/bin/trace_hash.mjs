#!/usr/bin/env node
// burn-math trace_hash: per-step tensor fingerprints for a whole run.
//
// The engine's trace observation already computes a sha256 over each
// step's output tensor (value_sha256) — but the helper's traceSteps used
// to drop it, keeping only shape/bytes/timing. This CLI surfaces the
// fingerprints, which turn "the outputs differ" into "the outputs first
// differ at step K": a divergence localizer that moves 64 hex chars per
// step instead of whole tensors.
//
//   node trace_hash.mjs <spec.json> [otherSpec.json] [maxTensorBytes] [--isolate]
// Prints {ok, max_tensor_bytes, output_slot, steps:[{index, layer, out_slot,
// out_shape, out_bytes, capture_status, finite_values, value_sha256}],
// comparison?}.
// With a second spec, comparison reports step_counts, output_slots,
// output_slot_equal, output_value_equal, the first step whose captured
// hashes differ (first_divergent_step; the shorter length when the chains
// differ in length with no earlier hash mismatch), divergent_step_detail,
// and whether the whole captured chain is equal (hash_chain_equal, which
// also requires equal output slots and equal output values). Steps whose
// tensor exceeds the byte budget report capture_status
// "tensor_budget_exceeded" and a null hash — they are skipped by the
// comparison, never treated as equal or divergent; raise maxTensorBytes
// to capture them.
// With --isolate, comparison also carries isolated_base_subspec and
// isolated_cand_subspec: the prefix subspecs up to the divergent step.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { extractPrefixSubspec } from '../lib/rollout.mjs';
import { gateSpec } from '../lib/gates.mjs';
import { printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';

const rawArgs = process.argv.slice(2);
const isolate = rawArgs.includes('--isolate');
const posArgs = rawArgs.filter((a) => a !== '--isolate');
const [specFile, otherFile, budgetArg] = posArgs;
if (!specFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node trace_hash.mjs <spec.json> [otherSpec.json] [maxTensorBytes] [--isolate]');
  process.exit(2);
}
const budget = budgetArg !== undefined ? Number(budgetArg)
  : (otherFile !== undefined && !otherFile.endsWith('.json') ? Number(otherFile) : 1024 * 1024);
const otherSpecFile = otherFile !== undefined && otherFile.endsWith('.json') ? otherFile : null;

let spec, otherSpec = null;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e); }
if (otherSpecFile) {
  try { otherSpec = loadSpec(otherSpecFile); }
  catch (e) { failCli(e); }
}

// Tahap 2a: fail-closed static gate BEFORE any WASM work — a defective
// spec used to run and report ok:true with a step_failed capture.
for (const [name, s] of [['spec', spec], ['other', otherSpec]]) {
  if (!s) continue;
  const gate = gateSpec(s);
  if (gate) printJsonAndExit({ ...gate, spec: name }, 1);
}

try {
  const rt = await loadWasm();
  const traceHashes = (s) => {
    const { reg, plan, g } = buildGraph(rt, s);
    let bundle = null, traced = null;
    try {
      bundle = makeBundle(rt, plan, s.inputs, portBySlot(s));
      traced = g.runWithTrace(reg, bundle, 0, s.steps.length, budget);
      const rep = JSON.parse(traced.report());
      const steps = (rep.steps ?? []).map((st) => {
        const decl = s.layers[s.steps[st.index]?.layer] ?? {};
        return {
          index: st.index,
          layer: decl.constructor ?? `type:${st.layer_type}`,
          out_slot: st.output_slot ?? null,
          out_shape: st.observation?.shape ?? null,
          out_bytes: st.observation?.byte_length ?? null,
          capture_status: st.observation?.capture_status ?? null,
          finite_values: st.observation?.finite_values ?? null,
          value_sha256: st.observation?.value_sha256 ?? null,
        };
      });
      return { steps, fault: rep.fault ?? null };
    } finally {
      // Tahap 2a: free every handle allocated after buildGraph, on
      // success AND on failure. Previously the try started AFTER
      // makeBundle/runWithTrace, so a throw there leaked bundle/g/
      // plan/reg. (buildGraph itself is exception-safe — its internal
      // catch frees plan/reg on failure.)
      try { if (traced?.free) traced.free(); } catch { /* ignore */ }
      try { if (bundle?.free) bundle.free(); } catch { /* ignore */ }
      try { if (g?.free) g.free(); } catch { /* ignore */ }
      try { if (plan?.free) plan.free(); } catch { /* ignore */ }
      try { if (reg?.free) reg.free(); } catch { /* ignore */ }
    }
  };

  const lastHashForSlot = (stepList, slot) => {
    for (let i = stepList.length - 1; i >= 0; i--) {
      if (stepList[i].out_slot === slot) return stepList[i].value_sha256 ?? null;
    }
    return null;
  };

  const { steps, fault } = traceHashes(spec);
  let otherResult = null;
  const result = { ok: true, max_tensor_bytes: budget, output_slot: spec.outputSlot, steps };
  if (otherSpec) {
    otherResult = traceHashes(otherSpec);
    const other = otherResult.steps;
    result.other_steps = other;
    let firstDivergent = null;
    const n = Math.min(steps.length, other.length);
    for (let i = 0; i < n; i++) {
      const a = steps[i].value_sha256, b = other[i].value_sha256;
      if (a && b && a !== b) { firstDivergent = i; break; }
    }
    if (firstDivergent === null && steps.length !== other.length) {
      firstDivergent = n;
    }
    const uncaptured = [];
    for (let i = 0; i < n; i++) {
      if (!steps[i].value_sha256 || !other[i].value_sha256) uncaptured.push(i);
    }
    const outputSlotEqual = spec.outputSlot === otherSpec.outputSlot;
    const outHashA = lastHashForSlot(steps, spec.outputSlot);
    const outHashB = lastHashForSlot(other, otherSpec.outputSlot);
    const outputValueEqual = Boolean(outHashA && outHashB && outHashA === outHashB);

    result.comparison = {
      step_counts: [steps.length, other.length],
      output_slots: [spec.outputSlot, otherSpec.outputSlot],
      output_slot_equal: outputSlotEqual,
      output_value_equal: outputValueEqual,
      first_divergent_step: firstDivergent,
      hash_chain_equal: firstDivergent === null && steps.length === other.length && uncaptured.length === 0 && outputSlotEqual && outputValueEqual,
      uncaptured_steps: uncaptured,
      ...(firstDivergent !== null ? {
        divergent_step_detail: {
          step_index: firstDivergent,
          baseline: steps[firstDivergent] ?? null,
          candidate: other[firstDivergent] ?? null,
        },
      } : {}),
      ...(isolate && firstDivergent !== null ? {
        isolated_base_subspec: firstDivergent < spec.steps.length ? extractPrefixSubspec(spec, firstDivergent) : spec,
        isolated_cand_subspec: firstDivergent < otherSpec.steps.length ? extractPrefixSubspec(otherSpec, firstDivergent) : otherSpec,
      } : {}),
    };
  }
  // Tahap 2a fail-closed (keputusan kontrak): a failed capture — or a
  // trace fault — fails the command instead of reporting ok:true with
  // a step_failed step. The record keeps the full steps array (and the
  // fault) so the failure stays attributed. `tensor_budget_exceeded`
  // is NOT a failure: those steps are skipped by the comparison, as
  // before.
  const failedStep = (arr) => arr.find((s) => s.capture_status === 'step_failed');
  const f1 = failedStep(steps);
  const f2 = otherResult ? failedStep(otherResult.steps) : null;
  const otherFault = otherResult?.fault ?? null;
  if (fault || otherFault || f1 || f2) {
    const bad = f1 ?? f2;
    const activeFault = fault ?? otherFault;
    printJsonAndExit({
      ok: false, code: 'step_failed',
      error: activeFault
        ? `trace fault: ${typeof activeFault === 'string' ? activeFault : JSON.stringify(activeFault)}`
        : `trace step ${bad.index} (${bad.layer}) failed to capture (step_failed)`,
      steps,
      ...(otherResult ? { other_steps: otherResult.steps } : {}),
      fault: activeFault,
    }, 1);
  }
  printJson(result);
} catch (e) {
  failCli(e, e?.envelope ?? null);
}
