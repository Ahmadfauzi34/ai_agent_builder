#!/usr/bin/env node
// burn-math graph_mutate: transactional & multi-branch equivalence-gated graph mutation.
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
// Usage: node graph_mutate.mjs <spec.json> <mutation.json> [cases.json] [absTol=1e-6] [relTol=1e-6] [--out <out.json>]
//   mutation.json: {"commit": bool, "edits": [
//     {"op":"setWeights","layer_index":0,"values":[...]},
//     {"op":"replaceStep","index":0,"layer":{"constructor":"relu","args":[1]},"slots":[a,b,c]},
//     {"op":"insertStep","index":0,"layer":{...},"slots":[a,b,c]},
//     {"op":"removeStep","index":0},
//     {"op":"reconnectStep","index":0,"slots":[a,b,c]},
//     {"op":"setOutput","slot":0} ]}
//   branches mode: {"commit": bool, "promote_branch": "<id>"?, "branches": [
//     {"branch_id": "<id>", "edits": [...]}, ... ]} — each branch is forked
//     into a CheckpointBranchSet, verified the same way, and the winner
//     (lowest max_abs_error among passing branches, or "promote_branch"
//     when given) is promoted via commitBranchByReceipt. Output adds
//     mode: "branch_set", branch reports with byte-level state_diff, and
//     winning_branch. --out writes the winning branch's mutated spec
//     (single-branch mode: the mutated spec when equivalent/committed).
//   cases.json: [{"<slot>":[values...]}, ...] (default: the spec's inputs)
// Prints {ok, baseline_state_sha256, candidate_state_sha256, equivalent,
// max_abs_error, max_rel_error, receipt_digest, committed, outputs?}.
// Exit 0 iff (commit ? committed : equivalent); 2 otherwise.
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { loadWasm, buildGraph, makeBundle } from '../lib/engine.mjs';
import { loadSpec, portBySlot } from '../lib/spec.mjs';
import { arityOf } from '../lib/shapes.mjs';
import { printJson, printJsonAndExit, failCli, assertKnownFlags , errText } from '../lib/runner.mjs';

// Layer-type codes (engine capabilities.layer_types; same map as
// lib/engine.mjs uses for wiring weights).
const LAYER_TYPE = {
  linear: 1, layerNorm: 2, batchNorm: 2, groupNorm: 2, instanceNorm: 2, rmsNorm: 2,
  conv: 3, conv1d: 3, conv2d: 3, convTranspose2d: 3, embedding: 5,
};

const rawArgs = process.argv.slice(2);
let outFile = null;
const posArgs = [];
for (let i = 0; i < rawArgs.length; i++) {
  if (rawArgs[i] === '--out' && i + 1 < rawArgs.length) {
    outFile = rawArgs[++i];
  } else {
    posArgs.push(rawArgs[i]);
  }
}

const [specFile, mutFile, casesFile, absTolArg, relTolArg] = posArgs;
if (!specFile || !mutFile || rawArgs.includes('--help') || rawArgs.includes('-h')) {
  console.error('usage: node graph_mutate.mjs <spec.json> <mutation.json> [cases.json] [absTol] [relTol] [--out <out.json>]');
  process.exit(2);
}
assertKnownFlags(rawArgs, { valued: ['--out'], prog: 'graph_mutate' });
const absTol = absTolArg !== undefined ? Number(absTolArg) : 1e-6;
const relTol = relTolArg !== undefined ? Number(relTolArg) : 1e-6;

let spec, mutation, cases;
try { spec = loadSpec(specFile); }
catch (e) { failCli(e); }
try { mutation = JSON.parse(fs.readFileSync(mutFile, 'utf8')); }
catch (e) { failCli(`cannot read mutation: ${errText(e)}`); }

const isMultiBranch = Array.isArray(mutation.branches) && mutation.branches.length > 0;
if (!isMultiBranch && (!Array.isArray(mutation.edits) || mutation.edits.length === 0)) {
  failCli('mutation.json must hold a non-empty "edits" array or "branches" array');
}
try {
  cases = casesFile ? JSON.parse(fs.readFileSync(casesFile, 'utf8')) : [spec.inputs];
} catch (e) { failCli(`cannot read cases: ${errText(e)}`); }

function applyEditsToSpecCopy(baseSpec, edits) {
  const s = JSON.parse(JSON.stringify(baseSpec));
  s.weights = s.weights ?? {};
  for (const edit of edits) {
    if (edit.op === 'setWeights') {
      s.weights[String(edit.layer_index)] = [...edit.values];
    } else if (edit.op === 'replaceStep') {
      const layerIdx = s.layers.length;
      s.layers.push(edit.layer);
      const arNum = arityOf(edit.layer.constructor);
      const isBinary = arNum === 2 || arNum === 'binary' || (arNum === null && edit.slots.length === 3 && edit.slots[0] !== edit.slots[1]);
      const [a, b, c] = edit.slots;
      s.steps[edit.index] = {
        kind: isBinary ? 'binary' : 'unary',
        layer: layerIdx,
        slots: isBinary ? [a, b, c] : [a, c ?? b],
      };
      s.numSlots = Math.max(s.numSlots, ...edit.slots.filter((x) => typeof x === 'number').map((x) => x + 1));
    } else if (edit.op === 'insertStep') {
      const layerIdx = s.layers.length;
      s.layers.push(edit.layer);
      const arNum = arityOf(edit.layer.constructor);
      const isBinary = arNum === 2 || arNum === 'binary' || (arNum === null && edit.slots.length === 3 && edit.slots[0] !== edit.slots[1]);
      const [a, b, c] = edit.slots;
      s.steps.splice(edit.index, 0, {
        kind: isBinary ? 'binary' : 'unary',
        layer: layerIdx,
        slots: isBinary ? [a, b, c] : [a, c ?? b],
      });
      s.numSlots = Math.max(s.numSlots, ...edit.slots.filter((x) => typeof x === 'number').map((x) => x + 1));
    } else if (edit.op === 'removeStep') {
      s.steps.splice(edit.index, 1);
    } else if (edit.op === 'reconnectStep') {
      const prev = s.steps[edit.index];
      const [a, b, c] = edit.slots;
      prev.slots = prev.kind === 'binary' ? [a, b, c] : [a, c ?? b];
    } else if (edit.op === 'setOutput') {
      s.outputSlot = edit.slot;
    }
  }
  return s;
}

try {
  const rt = await loadWasm();
  const { reg, plan, g } = buildGraph(rt, spec);
  const ports = portBySlot(spec);
  const identity = g.programIdentity();
  const baselineBytes = rt.exportMultiInputProgramBundle(g, reg, true);
  const baselineSha = `sha256:${createHash('sha256').update(baselineBytes).digest('hex')}`;

  const layerSpecOf = (decl) => {
    const s = rt.AgentLayerSpec[decl.constructor](...decl.args);
    if (typeof s !== 'object' || s === null) throw new Error(`unknown layer constructor: ${decl.constructor}`);
    return s;
  };

  if (isMultiBranch) {
    const bs = new rt.CheckpointBranchSet(reg, g, identity, baselineSha);
    const branchReports = [];
    const branchSpecById = new Map();
    const branchPlanById = new Map();

    try {
      for (const br of mutation.branches) {
        const bid = String(br.branch_id ?? br.id ?? '');
        if (!bid) throw new Error('each branch in mutation.branches must have a non-empty "branch_id"');
        if (!Array.isArray(br.edits) || br.edits.length === 0) {
          throw new Error(`branch '${bid}' must have a non-empty "edits" array`);
        }
        bs.fork(bid);
        for (const edit of br.edits) {
          if (edit.op === 'setWeights') {
            const decl = spec.layers[edit.layer_index];
            if (!decl) throw new Error(`setWeights: no layer at index ${edit.layer_index}`);
            const type = LAYER_TYPE[decl.constructor];
            if (type === undefined) throw new Error(`setWeights: layer type ${decl.constructor} has no flat-weight code`);
            bs.setWeightsFlat(bid, type, decl.args[0], Float32Array.from(edit.values));
          } else if (edit.op === 'replaceStep') {
            const [a, b, c] = edit.slots;
            const ls = layerSpecOf(edit.layer);
            try { bs.replaceStep(bid, edit.index, ls, a, b ?? a, c); } finally { if (ls?.free) ls.free(); }
          } else if (edit.op === 'insertStep') {
            const [a, b, c] = edit.slots;
            const ls = layerSpecOf(edit.layer);
            try { bs.insertStep(bid, edit.index, ls, a, b ?? a, c); } finally { if (ls?.free) ls.free(); }
          } else if (edit.op === 'removeStep') {
            bs.removeStep(bid, edit.index);
          } else if (edit.op === 'reconnectStep') {
            const [a, b, c] = edit.slots;
            bs.reconnectStep(bid, edit.index, a, b ?? a, c);
          } else if (edit.op === 'setOutput') {
            bs.setOutput(bid, edit.slot);
          } else {
            throw new Error(`unknown edit op '${edit.op}' in branch '${bid}'`);
          }
        }
        const stage = JSON.parse(bs.stageBranch(bid));
        const rawDiff = JSON.parse(bs.stateDiff(bid));
        const sd = rawDiff.state_diff ?? rawDiff;
        const regC = new rt.LayerRegistry();
        const gC = rt.importMultiInputProgramBundle(regC, Buffer.from(bs.candidateBundle(bid)));
        const planC = rt.MultiInputGraphPlan.fromBytes(gC.inputPlanV1());
        const vc = new rt.MultiInputVerificationCases(g, gC);
        for (const c of cases) {
          const bBase = makeBundle(rt, plan, c, ports);
          const bCand = makeBundle(rt, planC, c, ports);
          vc.addCase(bBase, bCand);
          if (bBase?.free) bBase.free();
          if (bCand?.free) bCand.free();
        }
        const verdict = JSON.parse(bs.verifyBranch(bid, vc, absTol, relTol));
        if (vc.free) vc.free();
        if (gC.free) gC.free();
        if (regC.free) regC.free();

        branchPlanById.set(bid, planC);
        branchSpecById.set(bid, applyEditsToSpecCopy(spec, br.edits));
        branchReports.push({
          branch_id: bid,
          candidate_state_sha256: verdict.candidate_state_checkpoint_bytes_sha256 ?? stage.candidate_state_checkpoint_bytes_sha256,
          equivalent: verdict.equivalent,
          max_abs_error: verdict.max_abs_error,
          max_rel_error: verdict.max_rel_error,
          receipt_digest: verdict.receipt_digest,
          state_diff: {
            changed_byte_count: sd.changed_byte_count ?? 0,
            byte_length_delta: sd.byte_length_delta ?? 0,
            ranges_count: (sd.ranges ?? []).length,
            ranges: sd.ranges ?? [],
          },
        });
      }

      const passing = branchReports.filter((r) => r.equivalent);
      passing.sort((a, b) => a.max_abs_error - b.max_abs_error);
      const preferredId = mutation.promote_branch ? String(mutation.promote_branch) : null;
      const winner = preferredId
        ? passing.find((r) => r.branch_id === preferredId) ?? null
        : (passing[0] ?? null);

      const baseEnv = {
        mode: 'branch_set',
        baseline_state_sha256: baselineSha,
        branch_count: bs.branchCount(),
        branches: branchReports,
        equivalent: winner !== null,
        winning_branch: winner ? winner.branch_id : null,
      };

      if (!mutation.commit) {
        if (outFile && winner) {
          fs.writeFileSync(outFile, JSON.stringify(branchSpecById.get(winner.branch_id), null, 2) + '\n');
          baseEnv.out_file = outFile;
        }
        printJsonAndExit({ ok: winner !== null, committed: false, ...baseEnv }, winner !== null ? 0 : 2);
      }

      if (!winner) {
        printJsonAndExit({
          ok: false,
          committed: false,
          error: 'no candidate branch passed verification within tolerance',
          ...baseEnv,
        }, 2);
      }

      const newG = bs.commitBranchByReceipt(winner.branch_id, reg, g, identity, baselineSha, winner.receipt_digest, true);
      const planW = branchPlanById.get(winner.branch_id);
      const bundle = makeBundle(rt, planW, cases[0], ports);
      const out = newG.run(reg, bundle);
      const values = Array.from(out.to_array());
      out.free();
      if (bundle?.free) bundle.free();
      if (newG.free) newG.free();
      const promotedBranch = bs.promotedBranch() ?? winner.branch_id;
      if (outFile) {
        fs.writeFileSync(outFile, JSON.stringify(branchSpecById.get(winner.branch_id), null, 2) + '\n');
      }
      printJsonAndExit({
        ok: true,
        committed: true,
        promoted_branch: promotedBranch,
        candidate_state_sha256: winner.candidate_state_sha256,
        max_abs_error: winner.max_abs_error,
        max_rel_error: winner.max_rel_error,
        receipt_digest: winner.receipt_digest,
        ...baseEnv,
        ...(outFile ? { out_file: outFile } : {}),
        outputs: { values },
      }, 0);
    } finally {
      for (const p of branchPlanById.values()) {
        if (p?.free) p.free();
      }
      if (bs.free) bs.free();
      if (g.free) g.free();
      if (plan.free) plan.free();
      if (reg.free) reg.free();
    }
  }

  const tx = new rt.GraphMutationTransaction(reg, g, identity, baselineSha);
  for (const edit of mutation.edits) {
    if (edit.op === 'setWeights') {
      const decl = spec.layers[edit.layer_index];
      if (!decl) throw new Error(`setWeights: no layer at index ${edit.layer_index}`);
      const type = LAYER_TYPE[decl.constructor];
      if (type === undefined) throw new Error(`setWeights: layer type ${decl.constructor} has no flat-weight code`);
      tx.setWeightsFlat(type, decl.args[0], Float32Array.from(edit.values));
    } else if (edit.op === 'replaceStep') {
      const [a, b, c] = edit.slots;
      const ls = layerSpecOf(edit.layer);
      try { tx.replaceStep(edit.index, ls, a, b ?? a, c); } finally { if (ls?.free) ls.free(); }
    } else if (edit.op === 'insertStep') {
      const [a, b, c] = edit.slots;
      const ls = layerSpecOf(edit.layer);
      try { tx.insertStep(edit.index, ls, a, b ?? a, c); } finally { if (ls?.free) ls.free(); }
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
    const bBase = makeBundle(rt, plan, c, ports);
    const bCand = makeBundle(rt, planC, c, ports);
    vc.addCase(bBase, bCand);
    if (bBase?.free) bBase.free();
    if (bCand?.free) bCand.free();
  }
  let verdict;
  try {
    verdict = JSON.parse(tx.verifyCases(vc, absTol, relTol));
  } catch (e) {
    if (planC.free) planC.free();
    if (tx.free) tx.free();
    if (g.free) g.free();
    if (plan.free) plan.free();
    if (reg.free) reg.free();
    printJsonAndExit({
      ok: false, stage: 'verifyCases', error: String(e.message ?? e),
      baseline_state_sha256: baselineSha,
      candidate_state_sha256: stage.candidate_state_checkpoint_bytes_sha256 ?? null,
      equivalent: null, committed: false,
    }, 2);
  } finally {
    if (vc.free) vc.free();
    if (gC.free) gC.free();
    if (regC.free) regC.free();
  }

  const mutatedSpec = applyEditsToSpecCopy(spec, mutation.edits);
  const base = {
    baseline_state_sha256: baselineSha,
    candidate_state_sha256: verdict.candidate_state_checkpoint_bytes_sha256,
    equivalent: verdict.equivalent,
    max_abs_error: verdict.max_abs_error,
    max_rel_error: verdict.max_rel_error,
    receipt_digest: verdict.receipt_digest,
  };

  if (!mutation.commit) {
    if (outFile && verdict.equivalent) {
      fs.writeFileSync(outFile, JSON.stringify(mutatedSpec, null, 2) + '\n');
      base.out_file = outFile;
    }
    if (planC.free) planC.free();
    if (tx.free) tx.free();
    if (g.free) g.free();
    if (plan.free) plan.free();
    if (reg.free) reg.free();
    printJsonAndExit({ ok: verdict.equivalent, committed: false, ...base }, verdict.equivalent ? 0 : 2);
  }
  if (!verdict.equivalent) {
    if (planC.free) planC.free();
    if (tx.free) tx.free();
    if (g.free) g.free();
    if (plan.free) plan.free();
    if (reg.free) reg.free();
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
  if (bundle?.free) bundle.free();
  if (newG.free) newG.free();
  if (planC.free) planC.free();
  if (tx.free) tx.free();
  if (g.free) g.free();
  if (plan.free) plan.free();
  if (reg.free) reg.free();
  if (outFile) {
    fs.writeFileSync(outFile, JSON.stringify(mutatedSpec, null, 2) + '\n');
    base.out_file = outFile;
  }
  printJson({ ok: true, committed: true, ...base, outputs: { values } });
} catch (e) {
  failCli(e, e?.envelope ?? null);
}
