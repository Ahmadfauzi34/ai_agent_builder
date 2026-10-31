#!/usr/bin/env node
// burn-math graph_mutate: transactional, equivalence-gated graph mutation.
//
// Wraps the engine's GraphMutationTransaction. The transaction snapshots
// the baseline program bundle, applies edits to an ISOLATED candidate,
// stages it, and verifies the candidate against the baseline over input
// cases (MultiInputVerificationCases). Promotion (commitByReceipt) is only
// possible with the receipt of a PASSING verification and authorize=true;
// without commit the live graph is untouched — abandoning the transaction
// is the rollback.
//
// Engine semantics (corrected 2026-10-21; the 2026-10-20 claim below it was
// wrong): the verifier checks that the graphs inside the verification cases
// match what was staged — the baseline case side must be the live baseline,
// the candidate case side must be the STAGED candidate. It does NOT require
// candidate program identity == baseline identity. Structural edits
// (replace/insert/remove/reconnect/setOutput) verify and promote fine when
// output-equivalent over the cases. The old CLI built both case sides from
// the live graph, which made the engine reject every identity-changing edit
// with "baseline or candidate program identity changed" — an artifact of the
// CLI, not an engine limitation. The CLI now imports the staged candidate
// bundle into a fresh registry and builds the candidate case side from it.
//
// (Superseded 2026-10-20 note, kept for provenance: it claimed the gate
// required candidate identity == baseline identity, limiting promotion to
// state edits. Disproven by probe 2026-10-21: insertStep verified
// equivalent=true and committed via both the transaction and the branch
// path once the cases were built correctly.)
//
// Usage: node graph_mutate.mjs <spec.json> <mutation.json> [cases.json] [absTol=1e-6] [relTol=1e-6]
//   mutation.json: {"commit": bool, "edits": [
//     {"op":"setWeights","layer_index":0,"values":[...]},
//     {"op":"replaceStep","index":0,"layer":{"constructor":"relu","args":[1]},"slots":[a,b,c]},
//     {"op":"insertStep","index":0,"layer":{...},"slots":[a,b,c]},
//     {"op":"removeStep","index":0},
//     {"op":"reconnectStep","index":0,"slots":[a,b,c]},
//     {"op":"setOutput","slot":0} ]}
//   cases.json: [{"<slot>":[values...]}, ...] (default: the spec's inputs)
// Prints {ok, baseline_state_sha256, candidate_state_sha256, equivalent,
// max_abs_error, max_rel_error, receipt_digest, committed, outputs?}.
// Exit 0 iff (commit ? committed : equivalent); 2 otherwise.
import fs from 'node:fs';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { printJson, printJsonAndExit, failCli } from '../lib/runner.mjs';

// Layer-type codes (engine capabilities.layer_types; same map as
// lib/engine.mjs uses for wiring weights).
const LAYER_TYPE = { linear: 1, layerNorm: 2, batchNorm: 2, groupNorm: 2, instanceNorm: 2, rmsNorm: 2, conv: 3, conv1d: 3, conv2d: 3, convTranspose2d: 3, embedding: 5 };

const [specFile, mutFile, casesFile, absTolArg, relTolArg] = process.argv.slice(2);
if (!specFile || !mutFile || process.argv.slice(2).includes('--help') || process.argv.slice(2).includes('-h')) {
  console.error('usage: node graph_mutate.mjs <spec.json> <mutation.json> [cases.json] [absTol] [relTol]');
  process.exit(2);
}
const absTol = absTolArg !== undefined ? Number(absTolArg) : 1e-6;
const relTol = relTolArg !== undefined ? Number(relTolArg) : 1e-6;

let spec, mutation, cases;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e.message); }
try { mutation = JSON.parse(fs.readFileSync(mutFile, 'utf8')); }
catch (e) { failCli(`cannot read mutation: ${e.message}`); }
if (!Array.isArray(mutation.edits) || mutation.edits.length === 0) {
  failCli('mutation.json must hold a non-empty "edits" array');
}
try {
  cases = casesFile ? JSON.parse(fs.readFileSync(casesFile, 'utf8')) : [spec.inputs];
} catch (e) { failCli(`cannot read cases: ${e.message}`); }

const { createHash } = await import('node:crypto');

try {
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);
  const identity = g.programIdentity();
  const baselineBytes = rt.exportMultiInputProgramBundle(g, reg, true);
  const baselineSha = `sha256:${createHash('sha256').update(baselineBytes).digest('hex')}`;

  const tx = new rt.GraphMutationTransaction(reg, g, identity, baselineSha);
  const layerSpecOf = (decl) => {
    const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
    if (typeof s !== 'object' || s === null) throw new Error(`unknown layer constructor: ${decl.constructor}`);
    return s;
  };
  for (const edit of mutation.edits) {
    if (edit.op === 'setWeights') {
      const decl = spec.layers[edit.layer_index];
      if (!decl) throw new Error(`setWeights: no layer at index ${edit.layer_index}`);
      const type = LAYER_TYPE[decl.constructor];
      if (type === undefined) throw new Error(`setWeights: layer type ${decl.constructor} has no flat-weight code`);
      tx.setWeightsFlat(type, decl.args[0], Float32Array.from(edit.values));
    } else if (edit.op === 'replaceStep') {
      const [a, b, c] = edit.slots;
      tx.replaceStep(edit.index, layerSpecOf(edit.layer), a, b ?? a, c);
    } else if (edit.op === 'insertStep') {
      const [a, b, c] = edit.slots;
      tx.insertStep(edit.index, layerSpecOf(edit.layer), a, b ?? a, c);
    } else if (edit.op === 'removeStep') {
      tx.removeStep(edit.index);
    } else if (edit.op === 'reconnectStep') {
      const [a, b, c] = edit.slots;
      tx.reconnectStep(edit.index, a, b ?? a, c);
    } else if (edit.op === 'setOutput') {
      tx.setOutput(edit.slot);
    } else {
      throw new Error(`unknown edit op: ${edit.op}`);
    }
  }

  const stage = JSON.parse(tx.stageCandidate());
  // The candidate case side must be built from the STAGED candidate, not the
  // live graph: import the staged bundle into a fresh registry (see header).
  const regC = new rt.LayerRegistry();
  const gC = rt.importMultiInputProgramBundle(regC, Buffer.from(tx.candidateBundle()));
  const planC = rt.MultiInputGraphPlan.fromBytes(gC.inputPlanV1());
  const vc = new rt.MultiInputVerificationCases(g, gC);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, plan, c, ports), makeBundle(rt, planC, c, ports));
  }
  let verdict;
  try {
    verdict = JSON.parse(tx.verifyCases(vc, absTol, relTol));
  } catch (e) {
    printJsonAndExit({
      ok: false, stage: 'verifyCases', error: String(e.message ?? e),
      baseline_state_sha256: baselineSha,
      candidate_state_sha256: stage.candidate_state_checkpoint_bytes_sha256 ?? null,
      equivalent: null, committed: false,
    }, 2);
  }

  const base = {
    baseline_state_sha256: baselineSha,
    candidate_state_sha256: verdict.candidate_state_checkpoint_bytes_sha256,
    equivalent: verdict.equivalent,
    max_abs_error: verdict.max_abs_error,
    max_rel_error: verdict.max_rel_error,
    receipt_digest: verdict.receipt_digest,
  };

  if (!mutation.commit) {
    printJsonAndExit({ ok: verdict.equivalent, committed: false, ...base }, verdict.equivalent ? 0 : 2);
  }
  if (!verdict.equivalent) {
    printJsonAndExit({ ok: false, committed: false, error: 'candidate not equivalent; engine retains no passing receipt', ...base }, 2);
  }
  const newG = tx.commitByReceipt(reg, g, identity, baselineSha, verdict.receipt_digest, true);
  // Run the promoted graph with the CANDIDATE plan: for structural edits the
  // promoted graph's plan is planC (input ports are contract-fixed, so this
  // is also correct for weight-only edits).
  const bundle = makeBundle(rt, planC, cases[0], ports);
  const out = newG.run(reg, bundle);
  const values = Array.from(out.to_array());
  out.free();
  printJson({ ok: true, committed: true, ...base, outputs: { values } });
} catch (e) {
  failCli(e.message, e.envelope ?? null);
}
