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
// Proven engine semantics (audited 2026-10-20): the verification gate
// requires the candidate's PROGRAM IDENTITY to equal the baseline's, so
// the promotable edit class is state edits (setWeights). Structural edits
// (replace/insert/remove/reconnect/setOutput) can be staged, but the
// engine rejects them at verifyCases with an identity-gate error, which
// this CLI surfaces faithfully as {stage:"verifyCases", equivalent:null}.
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
import { printJson, failCli } from '../lib/runner.mjs';

// Layer-type codes (engine capabilities.layer_types; same map as
// lib/engine.mjs uses for wiring weights).
const LAYER_TYPE = { linear: 1, layerNorm: 2, batchNorm: 2, groupNorm: 2, instanceNorm: 2, rmsNorm: 2, conv: 3, conv1d: 3, conv2d: 3, convTranspose2d: 3, embedding: 5 };

const [specFile, mutFile, casesFile, absTolArg, relTolArg] = process.argv.slice(2);
if (!specFile || !mutFile) {
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
  const vc = new rt.MultiInputVerificationCases(g, g);
  for (const c of cases) {
    vc.addCase(makeBundle(rt, plan, c, ports), makeBundle(rt, plan, c, ports));
  }
  let verdict;
  try {
    verdict = JSON.parse(tx.verifyCases(vc, absTol, relTol));
  } catch (e) {
    printJson({
      ok: false, stage: 'verifyCases', error: String(e.message ?? e),
      baseline_state_sha256: baselineSha,
      candidate_state_sha256: stage.candidate_state_checkpoint_bytes_sha256 ?? null,
      equivalent: null, committed: false,
    });
    process.exit(2);
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
    printJson({ ok: verdict.equivalent, committed: false, ...base });
    process.exit(verdict.equivalent ? 0 : 2);
  }
  if (!verdict.equivalent) {
    printJson({ ok: false, committed: false, error: 'candidate not equivalent; engine retains no passing receipt', ...base });
    process.exit(2);
  }
  const newG = tx.commitByReceipt(reg, g, identity, baselineSha, verdict.receipt_digest, true);
  const bundle = makeBundle(rt, plan, cases[0], ports);
  const out = newG.run(reg, bundle);
  const values = Array.from(out.to_array());
  out.free();
  printJson({ ok: true, committed: true, ...base, outputs: { values } });
} catch (e) {
  failCli(e.message, e.envelope ?? null);
}
