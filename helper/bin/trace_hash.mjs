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
//   node trace_hash.mjs <spec.json> [otherSpec.json] [maxTensorBytes]
// Prints {ok, steps:[{index, layer, out_slot, out_shape, out_bytes,
// capture_status, finite_values, value_sha256}], comparison?}.
// With a second spec, comparison reports step_counts, the first step
// whose captured hashes differ (first_divergent_step), and whether the
// whole captured chain is equal. Steps whose tensor exceeds the byte
// budget report capture_status "tensor_budget_exceeded" and a null
// hash — they are skipped by the comparison, never treated as equal
// or divergent; raise maxTensorBytes to capture them.
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, failCli } from '../lib/runner.mjs';

const [specFile, otherFile, budgetArg] = process.argv.slice(2);
if (!specFile || process.argv.slice(2).includes('--help') || process.argv.slice(2).includes('-h')) { console.error('usage: node trace_hash.mjs <spec.json> [otherSpec.json] [maxTensorBytes]'); process.exit(2); }
const budget = budgetArg !== undefined ? Number(budgetArg)
  : (otherFile !== undefined && !otherFile.endsWith('.json') ? Number(otherFile) : 1024 * 1024);
const otherSpecFile = otherFile !== undefined && otherFile.endsWith('.json') ? otherFile : null;

let spec, otherSpec = null;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }
if (otherSpecFile) {
  try { otherSpec = loadSpec(otherSpecFile); }
  catch (e) { failCli(e.message); }
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
    }
    return steps;
  };

  const steps = traceHashes(spec);
  const result = { ok: true, max_tensor_bytes: budget, steps };
  if (otherSpec) {
    const other = traceHashes(otherSpec);
    result.other_steps = other;
    let firstDivergent = null;
    const n = Math.min(steps.length, other.length);
    for (let i = 0; i < n; i++) {
      const a = steps[i].value_sha256, b = other[i].value_sha256;
      if (a && b && a !== b) { firstDivergent = i; break; }
    }
    const uncaptured = [];
    for (let i = 0; i < n; i++) {
      if (!steps[i].value_sha256 || !other[i].value_sha256) uncaptured.push(i);
    }
    result.comparison = {
      step_counts: [steps.length, other.length],
      first_divergent_step: firstDivergent,
      hash_chain_equal: firstDivergent === null && steps.length === other.length && uncaptured.length === 0,
      uncaptured_steps: uncaptured,
    };
  }
  printJson(result);
} catch (e) {
  failCli(e.message, e.envelope ?? null);
}
