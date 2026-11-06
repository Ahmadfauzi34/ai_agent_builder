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
import { printJson, failCli } from '../lib/runner.mjs';

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

try {
  const rt = await loadWasm();
  const traceHashes = (s) => {
    const { reg, plan, g } = buildGraph(rt, s);
    const bundle = makeBundle(rt, plan, s.inputs, portBySlot(s));
    const traced = g.runWithTrace(reg, bundle, 0, s.steps.length, budget);
    let steps;
    try {
      const rep = JSON.parse(traced.report());
      steps = (rep.steps ?? []).map((st) => {
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
    } finally {
      if (traced.free) traced.free();
      if (bundle?.free) bundle.free();
      if (g.free) g.free();
      if (plan.free) plan.free();
      if (reg.free) reg.free();
    }
    return steps;
  };

  const lastHashForSlot = (stepList, slot) => {
    for (let i = stepList.length - 1; i >= 0; i--) {
      if (stepList[i].out_slot === slot) return stepList[i].value_sha256 ?? null;
    }
    return null;
  };

  const steps = traceHashes(spec);
  const result = { ok: true, max_tensor_bytes: budget, output_slot: spec.outputSlot, steps };
  if (otherSpec) {
    const other = traceHashes(otherSpec);
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
  printJson(result);
} catch (e) {
  failCli(e, e?.envelope ?? null);
}
