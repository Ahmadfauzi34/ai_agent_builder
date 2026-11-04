#!/usr/bin/env node
// burn-math regression_recursive: Automated Regression & Post-PASS Verification Suite
// for the 4-Layer Recursive & Harness-Native Architecture (v3.0), Stage-2 Feature Recursion (v3.5),
// Stage-3 Existing-Feature Recursive Audit (v3.6), and Stage-4 Deep Observation Audit (v3.7).

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { loadEngineWithOracle, pages, profileSpec } from '../lib/opt.mjs';
import { compactSlotsByLiveness } from '../lib/explain.mjs';
import { compileRecursiveSpec } from '../lib/composer.mjs';
import { rolloutScan, rolloutFixpoint } from '../lib/rollout.mjs';
import { printJsonAndExit } from '../lib/runner.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const TMP = fs.mkdtempSync(path.join(os.tmpdir(), 'burn-rec-reg-'));

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function runCli(args, inputStr = null) {
  try {
    const out = execFileSync('node', args, {
      cwd: ROOT,
      input: inputStr ?? undefined,
      encoding: 'utf8',
      stdio: [inputStr !== null ? 'pipe' : 'ignore', 'pipe', 'pipe'],
    });
    return { code: 0, out, json: JSON.parse(out) };
  } catch (e) {
    const out = e.stdout ? e.stdout.toString() : '';
    const err = e.stderr ? e.stderr.toString() : '';
    let json = null;
    try { json = JSON.parse(out); } catch { /* ignore */ }
    return { code: e.status ?? 1, out, err, json };
  }
}

function writeTmpJson(name, obj) {
  const p = path.join(TMP, name);
  fs.writeFileSync(p, JSON.stringify(obj, null, 2) + '\n');
  return p;
}

// B5 fix helper: explicit zero targets preserving the old no-target
// behavior (minimize toward zero), now declared instead of silent.
function zeroTargetsFor(specFile, name) {
  const run = runCli(['bin/run.mjs', specFile]);
  assert(run.code === 0 && run.json?.ok === true, `run for zero-targets failed: ${run.out}`);
  const spec = JSON.parse(fs.readFileSync(specFile, 'utf8'));
  return writeTmpJson(name, { cases: [{ inputs: spec.inputs, expected: new Array(run.json.outputs.values.length).fill(0) }] });
}

const cases = [];
function test(name, fn) {
  cases.push({ name, fn });
}

// --- 1. LAYER 1: STRUCTURAL & COMPOSITIONAL RECURSION ---

test('L1.1: Recursive Spec v2 weight-tied repeat(depth=2) compiles & proves exact [1.5, 3]', async () => {
  const r = runCli(['bin/compose.mjs', 'prove', 'examples/recursive-gated-residual.v2.json']);
  assert(r.code === 0 && r.json?.ok === true && r.json?.verdict === 'ACCEPT', `prove failed: ${r.out}`);
  assert(JSON.stringify(r.json.outputs.values) === '[1.5,3]', `unexpected values: ${JSON.stringify(r.json.outputs.values)}`);
  assert(r.json.spec.layers.length === 2 && r.json.spec.steps.length === 4, `weight-tying broken: layers=${r.json.spec.layers.length}, steps=${r.json.spec.steps.length}`);
  assert(r.json.meta?.workspace_snapshot?.num_slots === 64, 'AgentWorkspace snapshot missing');
});

test('L1.2: Nested subgraphs + residual + parallel_concat combinators execute accurately', async () => {
  const v2 = {
    schema: 'burn-math.recursive-spec.v2',
    ports: {
      x: { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      y: { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    },
    subgraphs: {
      scale2: {
        in: ['u'],
        out: 'v',
        steps: [
          { op: 'add', in: ['u', 'u'], out: 'v' },
        ],
      },
      res_scale2: {
        in: ['a'],
        out: 'b',
        steps: [
          { op: 'residual', subgraph: 'scale2', in: ['a'], out: 'b' },
        ],
      },
    },
    steps: [
      {
        op: 'parallel_concat',
        dim: 1,
        in: ['x'],
        branches: [
          { op: 'subgraph:res_scale2', in: ['x'] },
          { op: 'add', in: ['x', 'y'] },
        ],
        out: 'joined',
      },
    ],
    output: 'joined',
    inputs: { x: [1, 2], y: [10, 20] },
  };
  const f = writeTmpJson('nested_comb.v2.json', v2);
  const r = runCli(['bin/compose.mjs', 'prove', f]);
  assert(r.code === 0 && r.json?.ok === true, `nested combinators failed: ${r.out}`);
  assert(JSON.stringify(r.json.outputs.values) === '[3,6,11,22]', `got ${JSON.stringify(r.json.outputs.values)}`);
});

test('L1.3: Guarded cycle detection rejects cyclic subgraph recursion with exit 2', async () => {
  const cyc = {
    schema: 'burn-math.recursive-spec.v2',
    ports: {
      x: { slot: 0, role: 'observation', shape: [1, 2, 1, 1] },
      y: { slot: 1, role: 'observation', shape: [1, 2, 1, 1] },
    },
    subgraphs: {
      A: { in: ['u'], out: 'v', steps: [{ op: 'subgraph:B', in: ['u'], out: 'v' }] },
      B: { in: ['u'], out: 'v', steps: [{ op: 'subgraph:A', in: ['u'], out: 'v' }] },
    },
    steps: [
      { op: 'add', in: ['x', 'y'], out: 's' },
      { op: 'subgraph:A', in: ['s'], out: 'out' },
    ],
    output: 'out',
    inputs: { x: [1, 2], y: [3, 4] },
  };
  const f = writeTmpJson('cycle.v2.json', cyc);
  const r = runCli(['bin/compose.mjs', 'compile', f]);
  assert(r.code === 2 && r.json?.ok === false && r.json?.code === 'recursive_cycle_detected', `cycle not rejected: ${r.out}`);
});

test('L1.4: Auto-repair (CEGIS) repairs missing intFromFloat/intToFloat casts, unused ports, and >u16 linear sharding', async () => {
  const brokenMixed = {
    numSlots: 4,
    layers: [
      { constructor: 'intAdd', args: [0] },
    ],
    steps: [
      { kind: 'binary', layer: 0, slots: [0, 1, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 3, role: 'context', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [5, 7], '1': [10, 20], '3': [1, 1] },
  };
  const f = writeTmpJson('broken_mixed.json', brokenMixed);
  const outF = path.join(TMP, 'repaired_mixed.json');
  const c = runCli(['bin/compose.mjs', 'compile', f, outF, '--auto-repair']);
  assert(c.code === 0 && c.json?.ok === true, `auto-repair compile failed: ${c.out}`);
  const receipts = c.json.meta?.remediation_receipt ?? [];
  assert(receipts.length === 4, `expected 4 remediation receipts (2 intFromFloat + 1 intToFloat + 1 unused port drain), got ${receipts.length}`);
  const runRes = runCli(['bin/run.mjs', outF]);
  assert(runRes.code === 0 && JSON.stringify(runRes.json?.outputs?.values) === '[15,27]', `repaired mixed spec failed to run: ${runRes.out}`);
});

test('L1.5: Symbolic Autodiff Graph Compiler (1st-order VJP & 2nd-order HVP) matches exact derivatives', async () => {
  const quadSpec = {
    numSlots: 4,
    layers: [
      { constructor: 'mul', args: [0] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'binary', layer: 0, slots: [0, 0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [3, -4], '1': [1, 2] },
  };
  const qFile = writeTmpJson('quad.json', quadSpec);
  const g1 = runCli(['bin/compose.mjs', 'grad-spec', qFile, '--wrt', '0', '--order', '1']);
  assert(g1.code === 0 && JSON.stringify(g1.json?.grad_values) === '[6,-8]', `order-1 grad mismatch: ${g1.out}`);
  const g2 = runCli(['bin/compose.mjs', 'grad-spec', qFile, '--wrt', '0', '--order', '2']);
  assert(g2.code === 0 && JSON.stringify(g2.json?.grad_values) === '[2,2]', `order-2 Hessian diag mismatch: ${g2.out}`);

  const linG2 = runCli(['bin/compose.mjs', 'grad-spec', 'examples/recursive-gated-residual.v2.json', '--wrt', '0', '--order', '2']);
  assert(linG2.code === 0 && JSON.stringify(linG2.json?.grad_values) === '[0,0]', `order-2 linear grad mismatch: ${linG2.out}`);
});

// --- 2. LAYER 2: TEMPORAL & EXECUTION-TIME RECURSION ---

test('L2.1: rollout fixpoint converges on contractive system and fails closed on explosive system', async () => {
  const conv = runCli(['bin/rollout.mjs', 'fixpoint', 'examples/recursive-gated-residual.v2.json', '--tol', '1e-6', '--max-steps', '50']);
  assert(conv.code === 0 && conv.json?.converged === true && conv.json?.status === 'CONVERGED', `fixpoint failed: ${conv.out}`);
  assert(Math.abs(conv.json.fixed_point.values[0] - 2.0) < 1e-5 && Math.abs(conv.json.fixed_point.values[1] - 4.0) < 1e-5, `wrong fixed point: ${JSON.stringify(conv.json.fixed_point.values)}`);
  assert(conv.json.empirical_lipschitz === 0.25, `expected empirical_lipschitz 0.25, got ${conv.json.empirical_lipschitz}`);

  const divSpec = {
    schema: 'burn-math.recursive-spec.v2',
    ports: {
      s: { slot: 0, role: 'state', shape: [1, 2, 1, 1] },
      b: { slot: 1, role: 'context', shape: [1, 2, 1, 1] },
    },
    steps: [
      { op: 'linear', args: [2, 2, false], weights: [4, 0, 0, 4], in: ['s'], out: 's4' },
      { op: 'add', in: ['s4', 'b'], out: 'next' },
    ],
    output: 'next',
    inputs: { s: [10, 10], b: [1, 1] },
  };
  const df = writeTmpJson('divergent.v2.json', divSpec);
  const divRes = runCli(['bin/rollout.mjs', 'fixpoint', df, '--div-threshold', '500', '--max-steps', '20']);
  assert(divRes.code === 2 && divRes.json?.ok === false && divRes.json?.status === 'DIVERGED_EXPLOSIVE', `divergent fixpoint did not fail closed: ${divRes.out}`);
});

test('L2.2: rollout bisect localizes first divergent step and extracts runnable prefix subspecs', async () => {
  const bRes = runCli(['bin/rollout.mjs', 'bisect', 'examples/linear-relu.json', 'examples/linear-relu-mutated.json']);
  assert(bRes.code === 0 && bRes.json?.ok === true, `bisect failed: ${bRes.out}`);
  assert(bRes.json.identical_trace === false && bRes.json.first_divergent_step === 0, `expected divergence at step 0, got ${bRes.json.first_divergent_step}`);
  const sub = bRes.json.isolated_base_subspec;
  assert(sub.steps.length === 2 && sub.meta.remediation_receipt[0].repair === 'drain_unused_port', `unexpected isolated subspec structure: ${JSON.stringify(sub)}`);
  const subFile = writeTmpJson('isolated_prefix.json', sub);
  const runSub = runCli(['bin/run.mjs', '--direct', subFile]);
  assert(runSub.code === 0 && JSON.stringify(runSub.json?.outputs?.values) === '[1.5,-3.5]', `isolated prefix subspec failed to run: ${runSub.out}`);
});

test('L2.3: Fase A Memory Oracle proves zero steady-state WASM memory page leak over 50-step rollout', async () => {
  const { rt } = await loadEngineWithOracle();
  const raw = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/recursive-gated-residual.v2.json'), 'utf8'));
  const compiled = compileRecursiveSpec(raw, { rt, useWorkspace: true });
  await rolloutScan(rt, compiled, { steps: 5, recordTrajectory: false });
  const pagesBefore = pages();
  for (let i = 0; i < 10; i++) {
    await rolloutScan(rt, compiled, { steps: 20, recordTrajectory: false });
    await rolloutFixpoint(rt, compiled, { maxSteps: 20, tol: 1e-6, recordTrajectory: false });
  }
  const pagesAfter = pages();
  assert(pagesAfter - pagesBefore === 0, `WASM linear memory leaked ${pagesAfter - pagesBefore} pages across 400 rollout ticks`);
});

// --- 3. LAYER 3: RECURSIVE TREE SEARCH & MULTI-LAYER OPTIMIZATION ---

test('L3.1: recurse search chains linear-branch-fusion + relu-idempotent + dead-step-elimination over CheckpointBranchSet', async () => {
  const redundantSpec = {
    numSlots: 8,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'linear', args: [1, 2, 2, false] },
      { constructor: 'add', args: [4] },
      { constructor: 'relu', args: [3] },
      { constructor: 'relu', args: [4] },
      { constructor: 'add', args: [5] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [0, 3] },
      { kind: 'binary', layer: 2, slots: [2, 3, 4] },
      { kind: 'unary', layer: 3, slots: [4, 5] },
      { kind: 'unary', layer: 4, slots: [5, 6] },
      { kind: 'binary', layer: 5, slots: [6, 1, 7] },
    ],
    outputSlot: 7,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [1, 0, 0, 1],
      '1': [2, 0, 0, 2],
    },
    inputs: { '0': [2, -3], '1': [10, 20] },
  };
  const rf = writeTmpJson('redundant.json', redundantSpec);
  const sRes = runCli(['bin/recurse.mjs', 'search', rf, '--max-depth', '3']);
  assert(sRes.code === 0 && sRes.json?.ok === true, `recurse search failed: ${sRes.out}`);
  assert(sRes.json.best_candidate.steps_count === 3, `expected 6 steps reduced to 3, got ${sRes.json.best_candidate.steps_count}`);
  assert(sRes.json.best_candidate.max_abs_error === 0, `expected 0 error, got ${sRes.json.best_candidate.max_abs_error}`);
});

test('L3.2: Multi-layer es_optimize fits a 2-linear-layer spec and improves fitness', async () => {
  const twoLayerSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'linear', args: [1, 2, 2, false] },
      { constructor: 'add', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'binary', layer: 2, slots: [3, 1, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [0.1, 0, 0, 0.1],
      '1': [0.1, 0, 0, 0.1],
    },
    inputs: { '0': [1, 2], '1': [0, 0] },
  };
  const targets = {
    cases: [
      { inputs: { '0': [1, 2], '1': [0, 0] }, expected: [1, 2] },
      { inputs: { '0': [2, -1], '1': [0, 0] }, expected: [2, -1] },
    ],
  };
  const sf = writeTmpJson('two_layer.json', twoLayerSpec);
  const tf = writeTmpJson('targets.json', targets);
  const fitRes = runCli(['bin/es_optimize.mjs', sf, tf, '25', '--sensitivity-guided']);
  assert(fitRes.code === 0 && fitRes.json?.ok === true, `multi-layer es_optimize failed: ${fitRes.out}`);
  assert(fitRes.json.active_layers.length === 2, `expected 2 active weighted layers, got ${fitRes.json.active_layers.length}`);
  assert(fitRes.json.final_best_fitness > fitRes.json.start_fitness, `fitness did not improve: ${fitRes.json.start_fitness} -> ${fitRes.json.final_best_fitness}`);
});

// --- 4. LAYER 4: AGENT HARNESS INTEGRATION ---

test('L4.1: Artifact Store round-trip, hydration in spec.inputs, and fail-closed corruption detection', async () => {
  const wsDir = path.join(TMP, 'ws_artifacts');
  const dRes = runCli(['bin/harness.mjs', 'digest', 'examples/linear-relu.json', '--workspace', wsDir]);
  assert(dRes.code === 0 && dRes.json?.digest?.artifact_ref?.startsWith('burn://sha256:'), `digest failed: ${dRes.out}`);
  const uri = dRes.json.digest.artifact_ref;

  const rRes = runCli(['bin/harness.mjs', 'resolve-artifact', uri, '--workspace', wsDir]);
  assert(rRes.code === 0 && JSON.stringify(rRes.json?.values) === '[4.5,10.5]', `resolve-artifact mismatch: ${rRes.out}`);

  const consumerSpec = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  consumerSpec.inputs['0'] = uri;
  const cf = writeTmpJson('consumer_artifact.json', consumerSpec);
  const cDigest = runCli(['bin/harness.mjs', 'digest', cf, '--workspace', wsDir]);
  assert(cDigest.code === 0 && cDigest.json?.ok === true, `artifact input hydration failed: ${cDigest.out}`);

  const hex = uri.replace('burn://sha256:', '');
  const artFile = path.join(wsDir, 'artifacts', `${hex}.json`);
  fs.writeFileSync(artFile, '{"kind":"tensor_f32","shape":[1,2,1,1],"values":[999,999]}\n');
  const badRes = runCli(['bin/harness.mjs', 'resolve-artifact', uri, '--workspace', wsDir]);
  assert(badRes.code === 2 && badRes.json?.ok === false && badRes.json?.code === 'artifact_integrity_failed', `corrupted artifact was not rejected: ${badRes.out}`);
});

test('L4.2: Declarative multi-stage harness workflow & subagent branch-export / branch-merge gate', async () => {
  const wsDir = path.join(TMP, 'ws_workflow');
  const wf = {
    workspace: wsDir,
    spec: path.join(ROOT, 'examples/recursive-gated-residual.v2.json'),
    pipeline: [
      { stage: 'compile_v2', auto_repair: true },
      { stage: 'validate' },
      { stage: 'search_rewrites', max_depth: 2 },
      { stage: 'grad_spec', wrt_slot: 0, order: 1 },
      { stage: 'rollout_fixpoint', tol: 1e-5, max_steps: 40 },
      { stage: 'checkpoint_promote', tag: 'verified-v1' },
    ],
  };
  const wfRes = runCli(['bin/harness.mjs', 'run-workflow', '-'], JSON.stringify(wf));
  assert(wfRes.code === 0 && wfRes.json?.ok === true && wfRes.json?.stages_completed === 6, `workflow failed: ${wfRes.out}`);

  const expRes = runCli(['bin/harness.mjs', 'branch-export', 'examples/linear-relu.json', '--workspace', wsDir]);
  assert(expRes.code === 0 && expRes.json?.branch_packets?.length === 2, `branch-export failed: ${expRes.out}`);

  const mergeRes = runCli([
    'bin/harness.mjs',
    'branch-merge',
    'examples/linear-relu.json',
    'examples/linear-relu.json',
    'examples/linear-relu-mutated.json',
  ]);
  assert(mergeRes.code === 0 && mergeRes.json?.ok === true, `branch-merge failed: ${mergeRes.out}`);
  assert(mergeRes.json.evaluations[0].accepted === true && mergeRes.json.evaluations[1].accepted === false, `branch-merge did not reject mutated candidate: ${mergeRes.out}`);
});

// --- 5. STAGE 2: RECURSIVE UPGRADES ACROSS ALL EXISTING FEATURES ---

test('S2.1: linear-chain-fusion (rectangular 2->3->2 with bias) & transpose-involution rewrite exactly', async () => {
  const chainSpec = {
    numSlots: 7,
    layers: [
      { constructor: 'linear', args: [0, 2, 3, true] },
      { constructor: 'linear', args: [1, 3, 2, true] },
      { constructor: 'transpose', args: [2, 1, 2] },
      { constructor: 'transpose', args: [3, 1, 2] },
      { constructor: 'add', args: [4] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'unary', layer: 2, slots: [3, 4] },
      { kind: 'unary', layer: 3, slots: [4, 5] },
      { kind: 'binary', layer: 4, slots: [5, 1, 6] },
    ],
    outputSlot: 6,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [1, 0, 1, 0, 1, -1, 0.5, -0.5, 1.0],
      '1': [2, 1, 0, -1, 1, 2, 1.0, -2.0],
    },
    inputs: { '0': [2, -1], '1': [0.5, 1.5] },
  };
  const cf = writeTmpJson('rect_chain_transpose.json', chainSpec);
  const sRes = runCli(['bin/recurse.mjs', 'search', cf, '--max-depth', '4']);
  assert(sRes.code === 0 && sRes.json?.ok === true, `search failed: ${sRes.out}`);
  assert(sRes.json.best_candidate.steps_count === 2, `expected 5 steps reduced to 2, got ${sRes.json.best_candidate.steps_count}`);
  assert(sRes.json.best_candidate.max_abs_error <= 1e-6, `expected <=1e-6 error, got ${sRes.json.best_candidate.max_abs_error}`);
});

test('S2.2: Liveness-based slot compaction (--compact-slots) recycles dead intermediate slots while preserving outputs', async () => {
  const deepChain = {
    schema: 'burn-math.recursive-spec.v2',
    ports: {
      x: { slot: 0, role: 'observation', shape: [1, 2, 1, 1] },
      b: { slot: 1, role: 'context', shape: [1, 2, 1, 1] },
    },
    subgraphs: {
      step_block: {
        in: ['s', 'bias'],
        out: 'next',
        steps: [
          { op: 'relu', in: ['s'], out: 'r' },
          { op: 'add', in: ['r', 'bias'], out: 'next' },
        ],
      },
    },
    steps: [
      { op: 'repeat', depth: 12, subgraph: 'step_block', in: ['x', 'b'], out: 'y' },
    ],
    output: 'y',
    inputs: { x: [1, 2], b: [0.5, 0.25] },
  };
  const df = writeTmpJson('deep_chain.v2.json', deepChain);
  const uncompF = path.join(TMP, 'uncomp.json');
  const compF = path.join(TMP, 'comp.json');
  const c1 = runCli(['bin/compose.mjs', 'compile', df, uncompF]);
  const c2 = runCli(['bin/compose.mjs', 'compile', df, compF, '--compact-slots']);
  assert(c1.code === 0 && c2.code === 0, `compile failed: ${c1.out} / ${c2.out}`);
  assert(c2.json.num_slots < c1.json.num_slots && c2.json.num_slots <= 5, `slot compaction did not reduce slots: ${c1.json.num_slots} -> ${c2.json.num_slots}`);
  const vRes = runCli(['bin/verify.mjs', uncompF, compF]);
  assert(vRes.code === 0 && vRes.json?.verdict === 'ACCEPT', `compacted spec diverged: ${vRes.out}`);
});

test('S2.3: Layer-weight fit_hessian_diag (--layer 0) & closed-loop solve-gd with backtracking line search', async () => {
  const hd = runCli(['bin/fit_hessian_diag.mjs', 'examples/linear-relu.json', '--layer', '0']);
  assert(hd.code === 0 && hd.json?.ok === true && hd.json?.target_kind === 'layer_weights' && hd.json?.layer === 0, `fit_hessian_diag --layer 0 failed: ${hd.out}`);
  assert(hd.json.param_count === 4 && hd.json.diagonal.length === 2, `unexpected Hessian dim: ${hd.out}`);

  const sgdOut = path.join(TMP, 'solved_gd.json');
  const ztS23 = zeroTargetsFor('examples/linear-relu.json', 's23_zero_targets.json');
  const sgd = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', ztS23, '--layer', '0', '--steps', '6', '--lr', '0.1', '--precondition', '--out', sgdOut]);
  assert(sgd.code === 0 && sgd.json?.ok === true, `solve-gd failed: ${sgd.out}`);
  assert(sgd.json.targets_mode === 'explicit', `targets_mode not explicit: ${sgd.out}`);
  assert(sgd.json.final_objective < sgd.json.initial_objective, `objective did not decrease: ${sgd.json.initial_objective} -> ${sgd.json.final_objective}`);
  assert(sgd.json.trajectory_verification.ascent_count === 0 && sgd.json.trajectory_verification.mismatch_count === 0, `trajectory verification failed: ${JSON.stringify(sgd.json.trajectory_verification)}`);
});

test('S2.4: Non-linear symbolic differentiation (sigmoid & tanh order-1 VJP and order-2 HVP) + 31 proven autodiff rules', async () => {
  const proveRes = runCli(['bin/autodiff.mjs', 'prove-rules']);
  assert(proveRes.code === 0 && proveRes.json?.ok === true && proveRes.json?.passed >= 32, `autodiff prove-rules failed: ${proveRes.out}`);

  const nlSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'sigmoid', args: [0] },
      { constructor: 'tanh', args: [1] },
      { constructor: 'add', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [1, 3] },
      { kind: 'binary', layer: 2, slots: [2, 3, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [0.5, -0.7], '1': [0.3, -0.4] },
  };
  const nlf = writeTmpJson('nl_sig_tanh.json', nlSpec);
  const g1 = runCli(['bin/compose.mjs', 'grad-spec', nlf, '--wrt', '0', '--order', '1']);
  const g2 = runCli(['bin/compose.mjs', 'grad-spec', nlf, '--wrt', '0', '--order', '2']);
  assert(g1.code === 0 && g2.code === 0, `non-linear grad-spec failed: ${g1.out} / ${g2.out}`);
  for (let i = 0; i < 2; i++) {
    const x = nlSpec.inputs['0'][i];
    const s = 1 / (1 + Math.exp(-x));
    const wantG1 = s * (1 - s);
    const wantG2 = s * (1 - s) * (1 - 2 * s);
    assert(Math.abs(g1.json.grad_values[i] - wantG1) < 1e-5, `sigmoid order-1 grad mismatch at ${i}: got ${g1.json.grad_values[i]}, want ${wantG1}`);
    assert(Math.abs(g2.json.grad_values[i] - wantG2) < 1e-5, `sigmoid order-2 HVP mismatch at ${i}: got ${g2.json.grad_values[i]}, want ${wantG2}`);
  }
});

test('S2.5: Extended DAG quant_budget & diff_ref match WASM within f32 ULP', async () => {
  const dagSpec = {
    numSlots: 7,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, true] },
      { constructor: 'gelu', args: [1] },
      { constructor: 'leakyRelu', args: [2, 0.1] },
      { constructor: 'sub', args: [3] },
      { constructor: 'hardSwish', args: [4] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'unary', layer: 2, slots: [1, 4] },
      { kind: 'binary', layer: 3, slots: [3, 4, 5] },
      { kind: 'unary', layer: 4, slots: [5, 6] },
    ],
    outputSlot: 6,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [1.0, -0.5, 0.25, 1.5, 0.1, -0.2],
    },
    inputs: { '0': [0.8, -1.2], '1': [0.4, -0.5] },
  };
  const df = writeTmpJson('dag_quant_ref.json', dagSpec);
  const casesF = writeTmpJson('dag_cases.json', [dagSpec.inputs]);
  const qb = runCli(['bin/quant_budget.mjs', df, casesF, '--tol', '0.1']);
  assert(qb.code === 0 && qb.json?.ok === true && qb.json?.mode === 'bound' && Array.isArray(qb.json?.layers), `quant_budget on DAG failed: ${qb.out}`);
  const dr = runCli(['bin/diff_ref.mjs', df]);
  assert(dr.code === 0 && dr.json?.ok === true && dr.json?.verdict?.startsWith('PASS'), `diff_ref on gelu/leakyRelu/sub/hardSwish failed: ${dr.out}`);
});

test('S2.6: Rectangular m x n weight_lab & recursive weight expression tree evaluation', async () => {
  const st = runCli(['bin/weight_lab.mjs', '--selftest']);
  assert(st.code === 0 && st.json?.ok === true, `weight_lab --selftest failed: ${st.out}`);
  const exprDoc = {
    repertoire: {
      up: { name: 'up', kind: 'map', W: [[0.5, 0], [0, 0.5], [0.25, -0.25]], b: [1, 0, -1], requires: {} },
      down: { name: 'down', kind: 'map', W: [[1, 0, 1], [0, 1, -1]], b: [0, 2], requires: {} },
    },
    expr: {
      op: 'repeat',
      steps: 2,
      expr: {
        op: 'compose',
        first: { op: 'ref', name: 'up' },
        second: { op: 'ref', name: 'down' },
      },
    },
  };
  const ef = writeTmpJson('weight_expr.json', exprDoc);
  const outW = path.join(TMP, 'composed_weight.json');
  const ev = runCli(['bin/weight_lab.mjs', 'eval', ef, '--out', outW]);
  assert(ev.code === 0 && ev.json?.ok === true && ev.json?.verdict === 'COMPOSED', `weight_lab eval failed: ${ev.out}`);
  const runW = runCli(['bin/weight_lab.mjs', 'run', outW, '--in', '4,8']);
  assert(runW.code === 0 && runW.json?.ok === true && runW.json?.max_abs_diff < 1e-5, `weight_lab run on composed expr failed: ${runW.out}`);
});

test('S2.7: Hierarchical 3D scene tree (--scene-tree) & inverse kinematics solver (--solve-ik)', async () => {
  const treeDoc = {
    nodes: [
      { id: 'shoulder', parent: null, chain: [{ op: 'translate', by: [1, 0, 0] }, { op: 'rotate_z', degrees: 90 }], point: [0, 0, 0] },
      { id: 'elbow', parent: 'shoulder', chain: [{ op: 'translate', by: [1, 0, 0] }], point: [0, 0, 0] },
    ],
  };
  const tf = writeTmpJson('scene_tree.json', treeDoc);
  const stRes = runCli(['bin/geom_build.mjs', '--scene-tree', tf]);
  assert(stRes.code === 0 && stRes.json?.ok === true && stRes.json?.nodes?.length === 2, `scene-tree failed: ${stRes.out}`);
  assert(Math.abs(stRes.json.nodes[1].world_point[1] - 2.0) < 1e-5, `unexpected elbow world_point: ${JSON.stringify(stRes.json.nodes[1].world_point)}`);

  const ikDoc = {
    joints: [
      { id: 'j1', axis: 'z', length: 1.0, initial_deg: 10 },
      { id: 'j2', axis: 'z', length: 1.0, initial_deg: 20 },
    ],
    target: [1.0, 1.0, 0],
    max_iters: 150,
    tol: 1e-3,
  };
  const ikf = writeTmpJson('ik_spec.json', ikDoc);
  const ikRes = runCli(['bin/geom_build.mjs', '--solve-ik', ikf]);
  assert(ikRes.code === 0 && ikRes.json?.ok === true && ikRes.json?.converged === true, `solve-ik failed: ${ikRes.out}`);
  assert(ikRes.json.final_error <= 1e-3, `IK did not reach target within tol: final_error=${ikRes.json.final_error}`);
});

test('S2.8: Layout-aware cost_model on layerNorm & recursive zoom fuzzing (--zoom-vs)', async () => {
  const lnSpec = {
    numSlots: 3,
    layers: [
      { constructor: 'layerNorm', args: [0, 4, 1e-5] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, 1, 4], layout: 'feature_last' },
      { slot: 1, role: 'observation', shape: [1, 1, 1, 4], layout: 'feature_last' },
    ],
    weights: {
      '0': [1, 1, 1, 1, 0, 0, 0, 0],
    },
    inputs: { '0': [1, 2, 3, 4], '1': [0, 0, 0, 0] },
  };
  const lnf = writeTmpJson('ln_cost.json', lnSpec);
  const cm = runCli(['bin/cost_model.mjs', lnf, '--iters', '10']);
  assert(cm.code === 0 && cm.json?.ok === true && cm.json?.per_op?.some((op) => op.layer === 'layerNorm') && cm.json?.warnings?.length === 0, `layout-aware cost_model failed: ${cm.out}`);

  const zoom = runCli(['bin/fuzz_cases.mjs', 'examples/linear-relu.json', '--zoom-vs', 'examples/linear-relu-mutated.json', '--zoom-depth', '3', '--count', '6']);
  assert(zoom.code === 0 && zoom.json?.ok === true && zoom.json?.zoom_depth === 3, `zoom fuzzing failed: ${zoom.out}`);
  assert(zoom.json.zoomed_max_abs_error >= zoom.json.initial_max_abs_error, `zoom fuzzing did not monotonically maintain/increase worst-case diff: ${zoom.out}`);
});

// --- 6. STAGE 3: EXISTING-FEATURE RECURSIVE HARMONIZATION AUDIT (v3.6) ---

test('S3.1: Universal loadSpec accepts Recursive Spec v2 & burn://sha256:... artifacts across validate, run, verify, diff_ref, soak, and trace_hash', async () => {
  const valV2 = runCli(['bin/validate.mjs', 'examples/recursive-gated-residual.v2.json']);
  assert(valV2.code === 0 && valV2.json?.valid === true, `validate on v2 failed: ${valV2.out}`);

  const runV2 = runCli(['bin/run.mjs', '--direct', 'examples/recursive-gated-residual.v2.json']);
  assert(runV2.code === 0 && JSON.stringify(runV2.json?.outputs?.values) === '[1.5,3]', `run on v2 failed: ${runV2.out}`);

  const drV2 = runCli(['bin/diff_ref.mjs', 'examples/recursive-gated-residual.v2.json']);
  assert(drV2.code === 0 && drV2.json?.ok === true && drV2.json?.max_abs_error === 0, `diff_ref on v2 failed: ${drV2.out}`);

  const soakV2 = runCli(['bin/soak.mjs', 'examples/recursive-gated-residual.v2.json', '5']);
  assert(soakV2.code === 0 && soakV2.json?.identical === true, `soak on v2 failed: ${soakV2.out}`);

  const thV2 = runCli(['bin/trace_hash.mjs', 'examples/recursive-gated-residual.v2.json']);
  assert(thV2.code === 0 && thV2.json?.steps?.length === 4, `trace_hash on v2 failed: ${thV2.out}`);

  const wsDir = path.join(TMP, 'ws_universal_art');
  const dig = runCli(['bin/harness.mjs', 'digest', 'examples/linear-relu.json', '--workspace', wsDir]);
  const uri = dig.json.digest.artifact_ref;
  const specWithArt = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  specWithArt.workspace = wsDir;
  specWithArt.inputs['0'] = uri;
  const artSpecFile = writeTmpJson('universal_art_spec.json', specWithArt);
  const runArt = runCli(['bin/run.mjs', '--direct', artSpecFile]);
  assert(runArt.code === 0 && JSON.stringify(runArt.json?.outputs?.values) === '[4.875,33.75]', `run.mjs on burn://sha256:... artifact failed: ${runArt.out}`);
});

test('S3.2: Weight-tied recursive BPTT & slot-compacted (--compact-slots) autodiff verify-fd and solve-gd match exact finite differences', async () => {
  const fdV2 = runCli(['bin/autodiff.mjs', 'verify-fd', 'examples/recursive-gated-residual.v2.json']);
  assert(fdV2.code === 0 && fdV2.json?.ok === true && fdV2.json?.mismatches?.length === 0, `autodiff verify-fd on weight-tied v2 failed: ${fdV2.out}`);

  const compFile = path.join(TMP, 'rec_compacted.json');
  runCli(['bin/compose.mjs', 'compile', 'examples/recursive-gated-residual.v2.json', compFile, '--compact-slots']);
  const fdComp = runCli(['bin/autodiff.mjs', 'verify-fd', compFile]);
  assert(fdComp.code === 0 && fdComp.json?.ok === true && fdComp.json?.mismatches?.length === 0, `autodiff verify-fd on --compact-slots failed: ${fdComp.out}`);

  const gsComp = runCli(['bin/compose.mjs', 'grad-spec', compFile, '--wrt', '0', '--order', '1']);
  assert(gsComp.code === 0 && JSON.stringify(gsComp.json?.grad_values) === '[0.25,0.25]', `grad-spec on --compact-slots failed: ${gsComp.out}`);

  const ztS32 = zeroTargetsFor(compFile, 's32_zero_targets.json');
  const gdComp = runCli(['bin/recurse.mjs', 'solve-gd', compFile, ztS32, '--layer', '0', '--steps', '4', '--lr', '0.05']);
  assert(gdComp.code === 0 && gdComp.json?.ok === true && gdComp.json?.final_loss < gdComp.json?.start_loss, `solve-gd on weight-tied compacted spec failed: ${gdComp.out}`);
});

test('S3.3: gradcheck --weights and fit_sensitivity --weights rank weight-tied recursive layer sensitivities', async () => {
  const gc = runCli(['bin/gradcheck.mjs', 'examples/recursive-gated-residual.v2.json', '--weights']);
  assert(gc.code === 0 && gc.json?.ok === true && gc.json?.weight_jacobians?.['0']?.cols === 4, `gradcheck --weights failed: ${gc.out}`);

  const fsens = runCli(['bin/fit_sensitivity.mjs', 'examples/recursive-gated-residual.v2.json', '4', '0', '--weights']);
  assert(fsens.code === 0 && fsens.json?.ok === true && fsens.json?.global_weight_top?.length === 4, `fit_sensitivity --weights failed: ${fsens.out}`);
  assert(Math.abs(fsens.json.global_weight_top[0].abs - 2.0) < 1e-3, `expected top weight sensitivity ~2.0, got ${fsens.json.global_weight_top[0].abs}`);
});

test('S3.4: linear_equiv proves exact algebraic equivalence across branched, residual, and unrolled recursive affine graphs', async () => {
  const eqSelf = runCli(['bin/linear_equiv.mjs', 'examples/recursive-gated-residual.v2.json', 'examples/recursive-gated-residual.v2.json']);
  assert(eqSelf.code === 0 && eqSelf.json?.verdict === 'ACCEPT', `linear_equiv on v2 recursive spec failed: ${eqSelf.out}`);

  const closedForm = {
    numSlots: 3,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, true] },
      { constructor: 'relu', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [1, 1] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'context', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [0.25, 0, 0, 0.25, 1.5, 3.0] },
    inputs: { '0': [0, 0], '1': [1, 2] },
  };
  const cfFile = writeTmpJson('closed_form_affine.json', closedForm);
  const eqClosed = runCli(['bin/linear_equiv.mjs', 'examples/recursive-gated-residual.v2.json', cfFile]);
  assert(eqClosed.code === 0 && eqClosed.json?.verdict === 'ACCEPT', `linear_equiv unrolled vs closed-form failed: ${eqClosed.out}`);
});

test('S3.5: macro_expand supports recurrent_cell and recursive depth unrolling with tie_weights', async () => {
  const recMacro = {
    macro: 'recurrent_cell',
    depth: 3,
    tie_weights: true,
    params: {
      state_dim: 2,
      in_dim: 2,
      state_slot: 0,
      in_slot: 1,
      out_slot: 2,
      bias: false,
    },
    weights: {
      '0': [0.5, 0, 0, 0.5],
      '1': [0.25, 0, 0, 0.25],
    },
    ports: [
      { slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [0.2, -0.2], '1': [0.4, 0.8] },
  };
  const mf = writeTmpJson('rec_cell_macro.json', recMacro);
  const exp = runCli(['bin/macro_expand.mjs', 'expand', mf]);
  assert(exp.code === 0 && exp.json?.ok === true, `macro_expand recurrent_cell failed: ${exp.out}`);
  assert(exp.json.spec.layers.length === 5 && exp.json.spec.steps.length === 15, `expected 5 shared layers and 15 steps for depth=3 tied recurrent_cell, got layers=${exp.json.spec.layers.length}, steps=${exp.json.spec.steps.length}`);

  const expandedFile = writeTmpJson('rec_cell_expanded.json', exp.json.spec);
  const prove = runCli(['bin/macro_expand.mjs', 'prove', mf, expandedFile]);
  assert(prove.code === 0 && prove.json?.equivalent === true, `macro_expand prove failed: ${prove.out}`);
});

test('S3.6: rewrite.mjs apply all (--recursive --compact-slots --verify) reaches multi-rule fixpoint in one call', async () => {
  const multiRwSpec = {
    numSlots: 9,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'linear', args: [1, 2, 2, false] },
      { constructor: 'add', args: [2] },
      { constructor: 'relu', args: [3] },
      { constructor: 'relu', args: [4] },
      { constructor: 'add', args: [5] },
      { constructor: 'relu', args: [6] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [0, 3] },
      { kind: 'binary', layer: 2, slots: [2, 3, 4] },
      { kind: 'unary', layer: 3, slots: [4, 5] },
      { kind: 'unary', layer: 4, slots: [5, 6] },
      { kind: 'binary', layer: 5, slots: [6, 1, 7] },
      { kind: 'unary', layer: 6, slots: [0, 8] },
    ],
    outputSlot: 7,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [1, 0, 0, 1], '1': [2, 0, 0, 2] },
    inputs: { '0': [1.5, -2.0], '1': [0.5, 1.0] },
  };
  const mrf = writeTmpJson('multi_rw.json', multiRwSpec);
  const rwRes = runCli(['bin/rewrite.mjs', 'apply', 'all', mrf, '--recursive', '--compact-slots', '--verify']);
  assert(rwRes.code === 0 && rwRes.json?.ok === true && rwRes.json?.verify?.equivalent === true, `rewrite apply all failed: ${rwRes.out}`);
  assert(rwRes.json.steps_before === 7 && rwRes.json.steps_after === 3, `expected 7 steps reduced to 3, got ${rwRes.json.steps_after}`);
});

test('S3.7: quant_budget bound mode supports gated recurrent mul & norm DAGs', async () => {
  const gatedSpec = {
    numSlots: 6,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, true] },
      { constructor: 'sigmoid', args: [1] },
      { constructor: 'linear', args: [2, 2, 2, true] },
      { constructor: 'mul', args: [3] },
      { constructor: 'add', args: [4] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'unary', layer: 2, slots: [0, 4] },
      { kind: 'binary', layer: 3, slots: [3, 4, 5] },
      { kind: 'binary', layer: 4, slots: [5, 1, 5] },
    ],
    outputSlot: 5,
    ports: [
      { slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'context', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [0.8, -0.2, 0.1, 0.9, 0.0, 0.0],
      '2': [0.5, 0.3, -0.4, 0.6, 0.1, -0.1],
    },
    inputs: { '0': [1.0, -0.5], '1': [0.2, 0.3] },
  };
  const gf = writeTmpJson('gated_quant.json', gatedSpec);
  const gCases = writeTmpJson('gated_cases.json', [gatedSpec.inputs]);
  const qb = runCli(['bin/quant_budget.mjs', gf, gCases, '--tol', '0.05', '--mode', 'bound']);
  assert(qb.code === 0 && qb.json?.ok === true && qb.json?.layers?.length === 2, `quant_budget on gated mul spec failed: ${qb.out}`);
});

test('S3.8: Sequential register chaining ($prev, $<idx>) in arith.mjs, int.mjs, and kernel.mjs --batch', async () => {
  const arithBatch = writeTmpJson('arith_chain.json', {
    ops: [
      { op: 'add', a: [1, 2], b: [3, 4] },
      { op: 'mul', a: '$prev', b: [2, 0.5] },
      { op: 'add', a: '$0', b: '$prev' },
    ],
  });
  const abRes = runCli(['bin/arith.mjs', '--batch', arithBatch]);
  assert(abRes.code === 0 && JSON.stringify(abRes.json?.results?.[2]?.values) === '[12,9]', `arith --batch chaining failed: ${abRes.out}`);

  const intBatch = writeTmpJson('int_chain.json', {
    ops: [
      { op: 'add', a: ['10000000000', '1'], b: ['20000000000', '1'] },
      { op: 'add', a: '$prev', b: ['5000000000', '3'] },
      { op: 'add', a: '$0', b: '$1' },
    ],
  });
  const ibRes = runCli(['bin/int.mjs', '--batch', intBatch]);
  assert(ibRes.code === 0 && JSON.stringify(ibRes.json?.results?.[2]?.values) === '["65000000000","7"]', `int --batch chaining failed: ${ibRes.out}`);

  const kernelBatch = writeTmpJson('kernel_chain.json', {
    ops: [
      { op: 'hypot', a: [3, 5], b: [4, 12] },
      { op: 'fma', a: '$prev', b: [2, 2], c: [0.5, 1.5] },
      { op: 'round', a: '$prev' },
    ],
  });
  const kbRes = runCli(['bin/kernel.mjs', '--batch', kernelBatch]);
  assert(kbRes.code === 0 && JSON.stringify(kbRes.json?.results?.[2]?.values) === '[10,28]', `kernel --batch chaining failed: ${kbRes.out}`);
});

// --- 7. STAGE 4: SECOND-PASS DEEP OBSERVATION AUDIT (v3.7) ---

test('S4.1: CheckpointBranchSet multi-branch competition in graph_mutate.mjs (fork -> stageBranch -> stateDiff -> verifyBranch -> commitBranchByReceipt)', async () => {
  const mutMulti = writeTmpJson('multi_branch_mut.json', {
    commit: true,
    branches: [
      {
        branch_id: 'bad_weight',
        edits: [{ op: 'setWeights', layer_index: 0, values: [9, 9, 9, 9] }],
      },
      {
        branch_id: 'good_relu_idemp',
        edits: [{ op: 'insertStep', index: 2, layer: { constructor: 'relu', args: [77] }, slots: [3, 3, 3] }],
      },
    ],
  });
  const outMut = path.join(TMP, 'mutated_winner.json');
  const gm = runCli(['bin/graph_mutate.mjs', 'examples/linear-relu.json', mutMulti, '--out', outMut]);
  assert(gm.code === 0 && gm.json?.ok === true && gm.json?.committed === true, `multi-branch graph_mutate failed: ${gm.out}`);
  assert(gm.json.mode === 'branch_set' && gm.json.promoted_branch === 'good_relu_idemp', `expected good_relu_idemp promoted, got ${gm.json.promoted_branch}`);
  assert(gm.json.branches[0].equivalent === false && gm.json.branches[0].state_diff.changed_byte_count > 0, `expected bad_weight state_diff changed_byte_count>0: ${JSON.stringify(gm.json.branches[0])}`);
  assert(gm.json.branches[1].equivalent === true && JSON.stringify(gm.json.outputs.values) === '[4.5,10.5]', `unexpected promoted output: ${JSON.stringify(gm.json.outputs)}`);
});

test('S4.2: Symbolic in-graph 1st-order VJP and 2nd-order HVP for div and sqrt match analytic calculus', async () => {
  const divSqrtSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'sqrt', args: [0] },
      { constructor: 'div', args: [1] },
      { constructor: 'add', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [0, 1, 3] },
      { kind: 'binary', layer: 2, slots: [2, 3, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [4, 9], '1': [2, 3] },
  };
  const dsf = writeTmpJson('div_sqrt.json', divSqrtSpec);
  const g1 = runCli(['bin/compose.mjs', 'grad-spec', dsf, '--wrt', '0', '--order', '1']);
  const g2 = runCli(['bin/compose.mjs', 'grad-spec', dsf, '--wrt', '0', '--order', '2']);
  assert(g1.code === 0 && g2.code === 0, `div/sqrt grad-spec failed: ${g1.out} / ${g2.out}`);
  assert(Math.abs(g1.json.grad_values[0] - 0.75) < 1e-5 && Math.abs(g1.json.grad_values[1] - 0.5) < 1e-5, `div/sqrt order-1 mismatch: ${JSON.stringify(g1.json.grad_values)}`);
  assert(Math.abs(g2.json.grad_values[0] - (-0.03125)) < 1e-5 && Math.abs(g2.json.grad_values[1] - (-1 / 108)) < 1e-5, `div/sqrt order-2 HVP mismatch: ${JSON.stringify(g2.json.grad_values)}`);
});

test('S4.3: 34-constructor diff_ref f64 oracle matches WASM on mish, glu, rmsNorm, layerNorm, maxPool1d, avgPool1d, and embedding', async () => {
  const normGluSpec = {
    numSlots: 6,
    layers: [
      { constructor: 'mish', args: [0] },
      { constructor: 'glu', args: [1, 3] },
      { constructor: 'rmsNorm', args: [2, 2, 1e-5] },
      { constructor: 'layerNorm', args: [3, 2, 1e-5] },
      { constructor: 'add', args: [4] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'unary', layer: 2, slots: [3, 4] },
      { kind: 'unary', layer: 3, slots: [4, 5] },
      { kind: 'binary', layer: 4, slots: [5, 1, 5] },
    ],
    outputSlot: 5,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, 1, 4], layout: 'any_rank4' },
      { slot: 1, role: 'observation', shape: [1, 1, 1, 2], layout: 'feature_last' },
    ],
    weights: {
      '2': [1.2, 0.8],
      '3': [1.0, 0.5, 0.1, -0.1],
    },
    inputs: { '0': [0.5, -1.0, 1.5, 0.2], '1': [0.1, 0.2] },
  };
  const ngf = writeTmpJson('norm_glu_ref.json', normGluSpec);
  const dr1 = runCli(['bin/diff_ref.mjs', ngf]);
  assert(dr1.code === 0 && dr1.json?.ok === true && dr1.json?.verdict?.startsWith('PASS'), `diff_ref on mish/glu/rmsNorm/layerNorm failed: ${dr1.out}`);

  const embPoolSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'embedding', args: [0, 4, 3] },
      { constructor: 'maxPool1d', args: [1, 2, 1, 0] },
      { constructor: 'avgPool1d', args: [2, 2, 1, 0] },
      { constructor: 'add', args: [3] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'unary', layer: 2, slots: [3, 4] },
      { kind: 'binary', layer: 3, slots: [4, 1, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'context', shape: [1, 2, 1, 1], layout: 'any_rank4' },
    ],
    weights: {
      '0': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    inputs: { '0': [0, 3], '1': [0, 0] },
  };
  const epf = writeTmpJson('emb_pool_ref.json', embPoolSpec);
  const dr2 = runCli(['bin/diff_ref.mjs', epf]);
  assert(dr2.code === 0 && dr2.json?.ok === true && dr2.json?.verdict?.startsWith('PASS'), `diff_ref on embedding/maxPool1d/avgPool1d failed: ${dr2.out}`);
});

test('S4.4: Temporal rollout unroller (rollout.mjs unroll --steps 3 --compact-slots) matches rollout scan bit-identically and supports BPTT autodiff', async () => {
  const unrolledFile = path.join(TMP, 'unrolled_3step.json');
  const un = runCli(['bin/rollout.mjs', 'unroll', 'examples/recursive-gated-residual.v2.json', unrolledFile, '--steps', '3', '--compact-slots']);
  assert(un.code === 0 && un.json?.ok === true && un.json?.layers_count === 2, `rollout unroll failed: ${un.out}`);

  const scanRes = runCli(['bin/rollout.mjs', 'scan', 'examples/recursive-gated-residual.v2.json', '--steps', '3']);
  const runUnrolled = runCli(['bin/run.mjs', '--direct', unrolledFile]);
  assert(scanRes.code === 0 && runUnrolled.code === 0, `scan or run failed: ${scanRes.out} / ${runUnrolled.out}`);
  assert(JSON.stringify(scanRes.json.final_output.values) === JSON.stringify(runUnrolled.json.outputs.values), `unrolled output ${JSON.stringify(runUnrolled.json.outputs.values)} != scan output ${JSON.stringify(scanRes.json.final_output.values)}`);

  const fdUnrolled = runCli(['bin/autodiff.mjs', 'verify-fd', unrolledFile]);
  assert(fdUnrolled.code === 0 && fdUnrolled.json?.ok === true && fdUnrolled.json?.mismatches?.length === 0, `BPTT verify-fd on 3-step unrolled rollout failed: ${fdUnrolled.out}`);
});

test('S4.5: Joint multi-layer reflective GD (solve-gd --multi-layer) and multi-layer trajectory verification (fit_gd_verify --layer all)', async () => {
  const twoLayerSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'linear', args: [1, 2, 2, false] },
      { constructor: 'add', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 3] },
      { kind: 'binary', layer: 2, slots: [3, 1, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [0.8, 0.1, -0.1, 0.8],
      '1': [0.7, -0.2, 0.1, 0.9],
    },
    inputs: { '0': [1.0, 2.0], '1': [0.0, 0.0] },
  };
  const f = writeTmpJson('multi_gd_spec.json', twoLayerSpec);
  const ztS45 = zeroTargetsFor(f, 's45_zero_targets.json');
  const gd = runCli(['bin/recurse.mjs', 'solve-gd', f, ztS45, '--multi-layer', '--steps', '8', '--lr', '0.02']);
  assert(gd.code === 0 && gd.json?.ok === true && gd.json?.layer_index === 'all', `multi-layer solve-gd failed: ${gd.out}`);
  assert(gd.json.active_layers.length === 2 && gd.json.final_loss < gd.json.start_loss * 0.1, `multi-layer GD did not converge: ${gd.json.start_loss} -> ${gd.json.final_loss}`);

  const trajFile = writeTmpJson('multi_gd_traj.json', gd.json.trajectory);
  const fv = runCli(['bin/fit_gd_verify.mjs', f, trajFile, '1e-4', '--layer', 'all']);
  assert(fv.code === 0 && fv.json?.ok === true && fv.json?.target_kind === 'multi_layer_weights' && fv.json?.mismatch_count === 0, `fit_gd_verify --layer all failed: ${fv.out}`);
});

test('S4.6: Recursive 3D transform op (repeat) and direct scene-tree projection in geom_project.mjs and geom_batch.mjs', async () => {
  const repeatChain = writeTmpJson('geom_repeat.json', {
    chain: [
      { op: 'repeat', depth: 4, chain: [{ op: 'rotate_z', degrees: 90 }] },
    ],
    point: [2, 3, 4],
  });
  const gb = runCli(['bin/geom_build.mjs', repeatChain]);
  assert(gb.code === 0 && Math.abs(gb.json.reference_point_f64[0] - 2) < 1e-6 && Math.abs(gb.json.reference_point_f64[1] - 3) < 1e-6, `geom_build repeat op failed: ${gb.out}`);

  const sceneProj = writeTmpJson('scene_proj.json', {
    nodes: [
      { id: 'base', parent: null, chain: [{ op: 'translate', by: [0, 0, -5] }], point: [0, 0, 0] },
      { id: 'tip', parent: 'base', chain: [{ op: 'repeat', depth: 2, chain: [{ op: 'translate', by: [0.5, 0, 0] }] }], point: [0, 0, 0] },
    ],
    camera: { eye: [0, 0, 0], target: [0, 0, -1], up: [0, 1, 0], fov_deg: 60, aspect: 1, near: 0.1, far: 100 },
    viewport: { w: 800, h: 600 },
  });
  const gp = runCli(['bin/geom_project.mjs', sceneProj]);
  const gbat = runCli(['bin/geom_batch.mjs', sceneProj]);
  assert(gp.code === 0 && gp.json?.ok === true && gp.json?.scene_nodes?.length === 2, `geom_project scene_nodes failed: ${gp.out}`);
  assert(gbat.code === 0 && gbat.json?.ok === true && gbat.json?.scene_nodes?.length === 2, `geom_batch scene_nodes failed: ${gbat.out}`);
  assert(Math.abs(gp.json.scene_nodes[1].world_point[0] - 1.0) < 1e-6 && Math.abs(gp.json.scene_nodes[1].world_point[2] - (-5.0)) < 1e-6, `unexpected tip world_point: ${JSON.stringify(gp.json.scene_nodes[1])}`);
});

test('S4.7: Custom tolerance in opt_compare.mjs (--tol), weight-tying detection in layer_state.mjs inspect, and divergence subspec isolation in trace_hash.mjs (--isolate)', async () => {
  const ls = runCli(['bin/layer_state.mjs', 'inspect', 'examples/recursive-gated-residual.v2.json']);
  assert(ls.code === 0 && ls.json?.ok === true, `layer_state inspect failed: ${ls.out}`);
  assert(ls.json.layers[0].is_weight_tied === true && JSON.stringify(ls.json.layers[0].used_by_steps) === '[0,2]', `expected layer 0 weight-tied across steps [0,2], got ${JSON.stringify(ls.json.layers[0])}`);

  const th = runCli(['bin/trace_hash.mjs', 'examples/linear-relu.json', 'examples/linear-relu-mutated.json', '--isolate']);
  assert(th.code === 0 && th.json?.comparison?.first_divergent_step === 0 && th.json?.comparison?.isolated_base_subspec?.steps?.length === 2, `trace_hash --isolate failed: ${th.out}`);

  const approxCand = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  approxCand.weights['0'][0] += 1e-4;
  const acFile = writeTmpJson('approx_cand.json', approxCand);
  const ocStrict = runCli(['bin/opt_compare.mjs', 'examples/linear-relu.json', acFile, '--iters', '3']);
  const ocRelaxed = runCli(['bin/opt_compare.mjs', 'examples/linear-relu.json', acFile, '--tol', '1e-2', '--iters', '3']);
  assert(ocStrict.code === 2 && ocStrict.json?.equivalent === false, `expected strict opt_compare to reject 1e-4 weight diff: ${ocStrict.out}`);
  assert(ocRelaxed.code === 0 && ocRelaxed.json?.equivalent === true, `expected opt_compare --tol 1e-2 to accept 1e-4 weight diff: ${ocRelaxed.out}`);
});

// --- 8. STAGE 5: v3.8 DEEP AUDIT REMEDIATION & HARDENING REGRESSIONS ---

test('REG-v3.8-01: compactSlotsByLiveness SSA-normalizes overwritten slots without silent corruption', async () => {
  const spec = {
    numSlots: 6,
    layers: [
      { constructor: 'relu', args: [0] },
      { constructor: 'add', args: [1] },
      { constructor: 'relu', args: [2] },
      { constructor: 'add', args: [3] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
      { kind: 'unary', layer: 2, slots: [1, 2] },
      { kind: 'binary', layer: 3, slots: [3, 2, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [10, 20], '1': [99, 99] },
  };
  const { spec: comp } = compactSlotsByLiveness(spec);
  const compFile = writeTmpJson('reg_v38_01_comp.json', comp);
  const runComp = runCli(['bin/run.mjs', '--direct', compFile]);
  assert(runComp.code === 0 && JSON.stringify(runComp.json?.outputs?.values) === '[208,218]', `expected [208,218], got ${runComp.out}`);
});

test('REG-v3.8-02: quant_budget bound mode handles zero-valued ports and norm Lipschitz amplification soundly', async () => {
  const normSpec = {
    numSlots: 4,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'transpose', args: [1, 1, 3] },
      { constructor: 'rmsNorm', args: [2, 2, 1e-5] },
      { constructor: 'transpose', args: [3, 1, 3] },
      { constructor: 'add', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [2, 2] },
      { kind: 'unary', layer: 2, slots: [2, 3] },
      { kind: 'unary', layer: 3, slots: [3, 3] },
      { kind: 'binary', layer: 4, slots: [3, 1, 3] },
    ],
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [1e-4, 0, 0, 1e-4], '2': [10.0, 10.0] },
    inputs: { '0': [1.0, -1.0], '1': [0.0, 0.0] },
  };
  const nf = writeTmpJson('reg_v38_02_norm.json', normSpec);
  const nCases = writeTmpJson('reg_v38_02_cases.json', [normSpec.inputs]);
  const qbBound = runCli(['bin/quant_budget.mjs', nf, nCases, '--tol', '1e9', '--mode', 'bound']);
  const qbMeas = runCli(['bin/quant_budget.mjs', nf, nCases, '--tol', '1e9', '--mode', 'measured']);
  assert(qbBound.code === 0 && qbBound.json?.ok === true && qbMeas.code === 0 && qbMeas.json?.ok === true, `quant_budget failed on zero port + rmsNorm: ${qbBound.out} / ${qbMeas.out}`);
  assert(qbBound.json.totals.bound_total_all >= qbMeas.json.totals.measured_delta_plan, `unsound bound_total_all ${qbBound.json.totals.bound_total_all} < measured_delta_plan ${qbMeas.json.totals.measured_delta_plan}`);
});

test('REG-v3.8-03: reflectiveGdSolve preserves shape metadata on compacted concat graphs and evaluates genuine 2-step hessianTrust', async () => {
  const concatSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'concat', args: [1, 1] },
      { constructor: 'linear', args: [2, 4, 2, false] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
      { kind: 'unary', layer: 2, slots: [3, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [0.5, 0.1, -0.2, 0.4],
      '2': [0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.5, -0.3],
    },
    inputs: { '0': [1.0, 2.0], '1': [0.5, -0.5] },
  };
  const cf = writeTmpJson('reg_v38_03_concat.json', concatSpec);
  const zt3803 = zeroTargetsFor(cf, 'reg_v38_03_zero_targets.json');
  const gd = runCli(['bin/recurse.mjs', 'solve-gd', cf, zt3803, '--layer', '0', '--steps', '3', '--lr', '0.05', '--precondition']);
  assert(gd.code === 0 && gd.json?.ok === true && Number.isFinite(gd.json.final_loss), `solve-gd on compacted concat graph failed: ${gd.out}`);
  assert(typeof gd.json.gd_verification?.prediction_off_count === 'number', `missing gd_verification.prediction_off_count: ${gd.out}`);
  assert(gd.json.preconditioned === true && gd.json.final_loss < gd.json.start_loss, `expected preconditioned solve-gd loss reduction: ${gd.out}`);
});

test('REG-v3.8-04: compose.mjs grad-spec supports intermediate slots, symbolic Spec v2 names, and default zero linear weights', async () => {
  const interSpec = {
    numSlots: 4,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
    ],
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [2, 0, 0, 3] },
    inputs: { '0': [1, 2], '1': [0.5, 0.5] },
  };
  const isf = writeTmpJson('reg_v38_04_inter.json', interSpec);
  const gsInter = runCli(['bin/compose.mjs', 'grad-spec', isf, '--wrt', '2', '--order', '1']);
  assert(gsInter.code === 0 && gsInter.json?.ok === true && JSON.stringify(gsInter.json?.grad_values) === '[1,1]', `grad-spec wrt intermediate slot 2 failed: ${gsInter.out}`);

  const gsSym = runCli(['bin/compose.mjs', 'grad-spec', 'examples/recursive-gated-residual.v2.json', '--wrt', 'state', '--order', '1']);
  assert(gsSym.code === 0 && gsSym.json?.ok === true && JSON.stringify(gsSym.json?.grad_values) === '[0.25,0.25]', `grad-spec wrt symbolic state failed: ${gsSym.out}`);

  const noWeightSpec = {
    numSlots: 2,
    layers: [{ constructor: 'linear', args: [0, 2, 2, true] }],
    steps: [{ kind: 'unary', layer: 0, slots: [0, 1] }],
    outputSlot: 1,
    ports: [{ slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' }],
    inputs: { '0': [3, 4] },
  };
  const nwf = writeTmpJson('reg_v38_04_noweight.json', noWeightSpec);
  const gsNw = runCli(['bin/compose.mjs', 'grad-spec', nwf, '--wrt', '0']);
  assert(gsNw.code === 0 && JSON.stringify(gsNw.json?.grad_values) === '[0,0]', `grad-spec on zero-default linear failed: ${gsNw.out}`);
});

test('REG-v3.8-05: weight_lab detects nullspace-of-ones spectral radius, 3x3 det_positive reflections, and eval accepts CAUTION', async () => {
  const nullspaceSet = writeTmpJson('reg_v38_05_nullspace.json', {
    name: 'nullspace_sym',
    kind: 'dynamics',
    W: [[1, -1], [-1, 1]],
    b: [0, 0],
    requires: {},
  });
  const cert1 = runCli(['bin/weight_lab.mjs', 'certify', nullspaceSet]);
  assert(cert1.code !== 0 && cert1.json?.ok === false && cert1.json?.verdict === 'REFUSED', `expected REFUSED on nullspace_sym, got ${cert1.out}`);
  assert(Math.abs(cert1.json.properties[0].value - 2.0) < 1e-4, `expected rho=2.0, got ${JSON.stringify(cert1.json.properties)}`);

  const refl3Set = writeTmpJson('reg_v38_05_refl3.json', {
    name: 'refl_3x3',
    kind: 'calibration',
    W: [[-1, 0, 0], [0, 1, 0], [0, 0, 1]],
    b: [0, 0, 0],
    requires: { det_positive: true },
  });
  const cert2 = runCli(['bin/weight_lab.mjs', 'certify', refl3Set]);
  assert(cert2.code !== 0 && cert2.json?.ok === false && cert2.json?.verdict === 'REFUSED', `expected REFUSED on 3x3 reflection, got ${cert2.out}`);
  const detProp = cert2.json.properties.find((p) => p.name === 'det_positive');
  assert(detProp && detProp.ok === false && Math.abs(detProp.value - (-1)) < 1e-6, `expected det_positive failure with det=-1, got ${JSON.stringify(cert2.json.properties)}`);

  const exprSet = writeTmpJson('reg_v38_05_expr.json', {
    repertoire: {
      a: { name: 'a', kind: 'map', W: [[1, 0], [0, 1]], b: [0, 0], requires: {} },
      b: { name: 'b', kind: 'map', W: [[0, -1], [1, 0]], b: [0, 0], requires: {} },
    },
    expr: {
      op: 'blend',
      alpha: 0.5,
      a: { op: 'ref', name: 'a' },
      b: { op: 'ref', name: 'b' },
    },
  });
  const outEval = path.join(TMP, 'reg_v38_05_eval_out.json');
  const ev = runCli(['bin/weight_lab.mjs', 'eval', exprSet, '--out', outEval]);
  assert(ev.code === 0 && ev.json?.ok === true && ev.json?.verdict === 'BLENDED' && fs.existsSync(outEval), `weight_lab eval failed on BLENDED: ${ev.out}`);
});

test('REG-v3.8-06: profileSpec and es_optimize leak 0 WASM linear memory pages across repeated runs', async () => {
  const eng = await loadEngineWithOracle();
  const spec = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  profileSpec(eng.rt, spec, { iters: 2, trace: true });
  const p0 = pages(eng);
  for (let i = 0; i < 40; i++) {
    profileSpec(eng.rt, spec, { iters: 2, trace: true });
  }
  const p1 = pages(eng);
  assert(p1 === p0, `profileSpec leaked WASM memory pages: ${p0} -> ${p1}`);
});

test('REG-v3.8-07: fit_gd_verify verifies trajectories from recurse.mjs solve-gd with targets.json', async () => {
  const targetsFile = writeTmpJson('reg_v38_07_targets.json', { cases: [{ inputs: { '0': [1, -2], '1': [3, 10.5] }, expected: [1.0, 2.0] }] });
  const gd = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', '--targets', targetsFile, '--layer', '0', '--steps', '4', '--lr', '0.02']);
  assert(gd.code === 0 && gd.json?.ok === true, `solve-gd with targets failed: ${gd.out}`);
  const trajFile = writeTmpJson('reg_v38_07_traj.json', gd.json.trajectory);
  const fv = runCli(['bin/fit_gd_verify.mjs', 'examples/linear-relu.json', trajFile, '1e-4', '--layer', '0']);
  assert(fv.code === 0 && fv.json?.ok === true && fv.json?.mismatch_count === 0, `fit_gd_verify failed on targeted trajectory: ${fv.out}`);
});

test('REG-fix-B5: solve-gd rejects malformed targets and declares default modes', async () => {
  const bare = writeTmpJson('fix_b5_bare.json', [1.0, 2.0]);
  const bad = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', '--targets', bare, '--layer', '0', '--steps', '2']);
  assert(bad.code === 2 && /non-empty "cases" array/.test(bad.json?.error ?? bad.out ?? ''), `bare-array targets not rejected: ${bad.out}`);
  const empty = writeTmpJson('fix_b5_empty.json', { cases: [] });
  const emp = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', '--targets', empty, '--layer', '0', '--steps', '2']);
  assert(emp.code === 2 && /non-empty "cases" array/.test(emp.json?.error ?? emp.out ?? ''), `empty cases not rejected: ${emp.out}`);
  const multi = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', '--layer', '0', '--steps', '2']);
  assert(multi.code === 0 && multi.json?.targets_mode === 'zero_default', `multi-output default mode wrong: ${multi.out}`);
  const singleSpec = writeTmpJson('fix_b5_single.json', { numSlots: 2, layers: [{ constructor: 'linear', args: [0, 1, 1, false] }], steps: [{ kind: 'unary', layer: 0, slots: [0, 1] }], outputSlot: 1, ports: [{ slot: 0, role: 'observation', shape: [1, 1, 1, 1], layout: 'feature_axis1_singleton' }], weights: { '0': [2.0] }, inputs: { '0': [3.0] } });
  const single = runCli(['bin/recurse.mjs', 'solve-gd', singleSpec, '--layer', '0', '--steps', '2']);
  assert(single.code === 0 && single.json?.targets_mode === 'scalar_objective', `single-output default mode wrong: ${single.out}`);
});

test('REG-fix-B1: branch-merge equivalence gate uses multiple deterministic cases', async () => {
  const base = writeTmpJson('fix_b1_base.json', { numSlots: 2, layers: [{ constructor: 'linear', args: [0, 1, 1, false] }], steps: [{ kind: 'unary', layer: 0, slots: [0, 1] }], outputSlot: 1, ports: [{ slot: 0, role: 'observation', shape: [1, 1, 1, 1], layout: 'feature_axis1_singleton' }], weights: { '0': [2.0] }, inputs: { '0': [1.0] } });
  const coincide = writeTmpJson('fix_b1_coincide.json', { numSlots: 2, layers: [{ constructor: 'linear', args: [0, 1, 1, true] }], steps: [{ kind: 'unary', layer: 0, slots: [0, 1] }], outputSlot: 1, ports: [{ slot: 0, role: 'observation', shape: [1, 1, 1, 1], layout: 'feature_axis1_singleton' }], weights: { '0': [1.0, 1.0] }, inputs: { '0': [1.0] } });
  const rej = runCli(['bin/harness.mjs', 'branch-merge', base, coincide]);
  assert(rej.json?.ok === false && rej.json?.evaluations?.[0]?.accepted === false, `coinciding candidate promoted: ${rej.out}`);
  assert((rej.json?.cases_count ?? 0) >= 2 && (rej.json?.equivalence_cases ?? []).includes('baseline'), `gate did not run multiple cases: ${rej.out}`);
  const acc = runCli(['bin/harness.mjs', 'branch-merge', base, base]);
  assert(acc.code === 0 && acc.json?.ok === true && acc.json?.evaluations?.[0]?.accepted === true, `identical candidate rejected: ${acc.out}`);
});

test('REG-fix-B3: tied repeat with weights_per_depth fails closed with a named code', async () => {
  const v2 = writeTmpJson('fix_b3_tied_perdepth.json', { schema: 'burn-math.recursive-spec.v2', ports: { state: { slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' }, bias_ctx: { slot: 1, role: 'context', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' } }, subgraphs: { contract_step: { in: ['s', 'b'], out: 's_next', steps: [{ op: 'linear', args: [2, 2, false], weight_key: 'w', in: ['s'], out: 'scaled' }, { op: 'add', in: ['scaled', 'b'], out: 's_next' }] } }, steps: [{ op: 'repeat', subgraph: 'contract_step', depth: 2, tie_weights: true, in: ['state', 'bias_ctx'], out: 'final_state', weights_per_depth: [{ w: [0.5, 0, 0, 0.5] }, { w: [2, 0, 0, 2] }] }], output: 'final_state', inputs: { state: [0.0, 0.0], bias_ctx: [1.0, 2.0] } });
  const bad = runCli(['bin/compose.mjs', 'compile', v2, path.join(TMP, 'fix_b3_out.json')]);
  assert(bad.code === 2 && bad.json?.code === 'tied_weights_per_depth_conflict', `tied+perdepth not rejected named: ${bad.out}`);
});

test('REG-fix-B4: unknown symbolic state-port name fails closed with a named code', async () => {
  const bad = runCli(['bin/rollout.mjs', 'unroll', 'examples/linear-relu.json', path.join(TMP, 'fix_b4_out.json'), '--steps', '2', '--state-port', 'nosuchname']);
  assert(bad.code === 2 && bad.json?.code === 'rollout_unknown_state_port', `unknown state-port not rejected named: ${bad.out}`);
  const ok = runCli(['bin/rollout.mjs', 'unroll', 'examples/linear-relu.json', path.join(TMP, 'fix_b4_ok.json'), '--steps', '2', '--state-port', '0']);
  assert(ok.code === 0 && ok.json?.ok === true, `numeric state-port control failed: ${ok.out}`);
});

test('REG-fix-B2: mutual file references fail closed with recursive_cycle_detected', async () => {
  const fa = writeTmpJson('fix_b2_fa.json', { schema: 'burn-math.recursive-spec.v2', ports: { x: { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' } }, steps: [{ op: 'ref', $ref: 'fix_b2_fb.json', in: ['x'], out: 'r' }], output: 'r', inputs: { x: [1, 2] } });
  writeTmpJson('fix_b2_fb.json', { schema: 'burn-math.recursive-spec.v2', ports: { x: { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' } }, steps: [{ op: 'ref', $ref: 'fix_b2_fa.json', in: ['x'], out: 'r' }], output: 'r', inputs: { x: [1, 2] } });
  const bad = runCli(['bin/compose.mjs', 'compile', fa, path.join(TMP, 'fix_b2_out.json')]);
  assert(bad.code === 2 && bad.json?.code === 'recursive_cycle_detected', `file-ref cycle not detected named: ${bad.out}`);
});

test('REG-fix-B7: solve-gd reported losses match independent recomputation from weights', async () => {
  const targets = writeTmpJson('fix_b7_targets.json', { cases: [{ inputs: { '0': [1, -2], '1': [3, 10.5] }, expected: [1.0, 2.0] }] });
  const gd = runCli(['bin/recurse.mjs', 'solve-gd', 'examples/linear-relu.json', targets, '--layer', '0', '--steps', '4', '--lr', '0.05']);
  assert(gd.code === 0 && gd.json?.ok === true, `solve-gd failed: ${gd.out}`);
  const sse = (y, e) => y.reduce((a, v, i) => a + (v - e[i]) ** 2, 0);
  const run0 = runCli(['bin/run.mjs', 'examples/linear-relu.json']);
  assert(run0.code === 0, `run original failed: ${run0.out}`);
  const startSse = sse(run0.json.outputs.values, [1.0, 2.0]);
  assert(Math.abs(startSse - gd.json.start_loss) < 1e-5, `start_loss not reproducible: ${startSse} vs ${gd.json.start_loss}`);
  const spec = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  for (const [idx, w] of Object.entries(gd.json.best_weights_by_layer)) spec.weights[idx] = w;
  const finalFile = writeTmpJson('fix_b7_final.json', spec);
  const run1 = runCli(['bin/run.mjs', finalFile]);
  assert(run1.code === 0, `run final failed: ${run1.out}`);
  const finalSse = sse(run1.json.outputs.values, [1.0, 2.0]);
  assert(Math.abs(finalSse - gd.json.final_loss) < 1e-5, `final_loss not reproducible: ${finalSse} vs ${gd.json.final_loss}`);
  assert(gd.json.gd_verification?.mismatch_count === 0, `mismatch_count nonzero: ${gd.out}`);
});

test('REG-fix-B8: failing workflow returns partial reports and syncs partial state; fixpoint lipschitz independent of trajectory recording', async () => {
  const wfFail = writeTmpJson('fix_b8_fail.json', { spec: path.join(ROOT, 'examples/linear-relu.json'), pipeline: [{ stage: 'validate' }, { stage: 'bogus_stage' }] });
  const wsFail = path.join(TMP, 'fix_b8_ws');
  const bad = runCli(['bin/harness.mjs', 'run-workflow', wfFail, '--workspace', wsFail]);
  assert(bad.code === 2 && bad.json?.ok === false && bad.json?.failed_stage === 'bogus_stage', `failing workflow not reported partially: ${bad.out}`);
  assert(bad.json?.stage_reports?.length === 2 && bad.json.stage_reports[0]?.ok === true && bad.json.stage_reports[1]?.ok === false, `partial reports missing: ${bad.out}`);
  const st = JSON.parse(fs.readFileSync(path.join(wsFail, 'workspace-state.json'), 'utf8'));
  assert(st.wasm_workspace !== undefined, 'partial state not synced on workflow failure');
  const fpSpec = writeTmpJson('fix_b8_fp.json', { numSlots: 4, layers: [{ constructor: 'linear', args: [0, 2, 2, false] }, { constructor: 'add', args: [1] }], steps: [{ kind: 'unary', layer: 0, slots: [0, 2] }, { kind: 'binary', layer: 1, slots: [2, 1, 3] }], outputSlot: 3, ports: [{ slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' }, { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' }], weights: { '0': [0.5, 0, 0, 0.5] }, inputs: { '0': [2, 4], '1': [1, 0] } });
  const wfFp = writeTmpJson('fix_b8_fp_wf.json', { spec: fpSpec, pipeline: [{ stage: 'rollout_fixpoint' }] });
  const fp = runCli(['bin/harness.mjs', 'run-workflow', wfFp, '--workspace', path.join(TMP, 'fix_b8_fp_ws')]);
  assert(fp.code === 0 && fp.json?.ok === true, `fixpoint workflow failed: ${fp.out}`);
  const rep = fp.json?.stage_reports?.[0];
  assert(rep?.status === 'CONVERGED' && Math.abs((rep?.empirical_lipschitz ?? -1) - 0.5) < 1e-6, `lipschitz null or wrong in workflow fixpoint stage: ${fp.out}`);
});

test('REG-v3.8-08: Universal burn://sha256:... hydration in rollout/recurse/compose and 47-tool capability registry', async () => {
  const wsDir = path.join(TMP, 'ws_v38_08');
  const dig = runCli(['bin/harness.mjs', 'digest', 'examples/recursive-gated-residual.v2.json', '--workspace', wsDir]);
  const uri = dig.json.digest.artifact_ref;
  const v2Art = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/recursive-gated-residual.v2.json'), 'utf8'));
  v2Art.workspace = wsDir;
  v2Art.inputs.state = uri;
  const v2ArtFile = writeTmpJson('reg_v38_08_v2_art.json', v2Art);

  const compRes = runCli(['bin/compose.mjs', 'prove', v2ArtFile]);
  assert(compRes.code === 0 && compRes.json?.ok === true, `compose prove on v2 artifact failed: ${compRes.out}`);

  const scanRes = runCli(['bin/rollout.mjs', 'scan', v2ArtFile, '--steps', '2']);
  assert(scanRes.code === 0 && scanRes.json?.ok === true, `rollout scan on v2 artifact failed: ${scanRes.out}`);

  const recRes = runCli(['bin/recurse.mjs', 'search', v2ArtFile, '--max-depth', '1']);
  assert(recRes.code === 0 && recRes.json?.ok === true, `recurse search on v2 artifact failed: ${recRes.out}`);

  const capRes = runCli(['bin/harness.mjs', 'capabilities']);
  const uniqueBins = new Set(Object.values(capRes.json?.domains ?? {}).flatMap((d) => d.tools.map((t) => t.bin)));
  assert(capRes.code === 0 && uniqueBins.size === 47, `expected 47 unique tools in capabilities, got ${uniqueBins.size}`);
});

// --- 9. STAGE 6: v3.9 TOOL HONESTY & FAIL-CLOSED INTEGRITY REGRESSIONS (F-01..F-12) ---

test('REG-v3.9-01 (F-01): rewrites.v1.json catalog contains no placeholder sequential hex SHA-256 strings', async () => {
  const rwList = runCli(['bin/rewrite.mjs', 'list']);
  assert(rwList.code === 0 && rwList.json?.ok === true, `rewrite list failed: ${rwList.out}`);
  const badSequential = /0123456789abcdef|12349812abcd/i;
  for (const r of rwList.json.rewrites) {
    const bSha = r.proof?.before_sha256 ?? '';
    const aSha = r.proof?.after_sha256 ?? '';
    assert(/^sha256:[0-9a-f]{64}$/.test(bSha) && !badSequential.test(bSha), `invalid or placeholder before_sha256 in ${r.name}: ${bSha}`);
    assert(/^sha256:[0-9a-f]{64}$/.test(aSha) && !badSequential.test(aSha), `invalid or placeholder after_sha256 in ${r.name}: ${aSha}`);
  }
});

test('REG-v3.9-02 (F-02): recurse search produces distinct candidate_state_sha256 and non-zero state_diff on structural edits', async () => {
  const deadSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'relu', args: [0] },
      { constructor: 'add', args: [1] },
      { constructor: 'relu', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 3] },
      { kind: 'unary', layer: 2, slots: [0, 4] },
    ],
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [2, -3], '1': [4, 5] },
  };
  const df = writeTmpJson('reg_v39_02_dead.json', deadSpec);
  const sr = runCli(['bin/recurse.mjs', 'search', df, '--max-depth', '2']);
  assert(sr.code === 0 && sr.json?.ok === true, `recurse search failed: ${sr.out}`);
  assert(sr.json.best_candidate.steps_count === 2, `expected 2 steps, got ${sr.json.best_candidate.steps_count}`);
  assert(sr.json.best_candidate.candidate_state_sha256 !== sr.json.baseline.state_sha256, `structural candidate must have distinct state sha256 from baseline`);
  const candNode = sr.json.search_tree.find((n) => n.id === sr.json.best_candidate.id);
  assert(candNode?.state_diff && (candNode.state_diff.steps_delta !== 0 || candNode.state_diff.changed_byte_count > 0), `expected non-zero state_diff on structural edit: ${JSON.stringify(candNode?.state_diff)}`);
});

test('REG-v3.9-03 (F-03): linear_equiv REJECTS folding active non-context observation ports into constant bias', async () => {
  const specA = {
    numSlots: 3,
    layers: [{ constructor: 'add', args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [1, 2], '1': [10, 20] },
  };
  const specB = {
    numSlots: 4,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, true] },
      { constructor: 'relu', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [1, 3] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [1, 0, 0, 1, 10, 20] },
    inputs: { '0': [1, 2], '1': [10, 20] },
  };
  const fa = writeTmpJson('reg_v39_03_a.json', specA);
  const fb = writeTmpJson('reg_v39_03_b.json', specB);
  const le = runCli(['bin/linear_equiv.mjs', fa, fb]);
  assert(le.code === 2 && le.json?.verdict === 'REJECT', `linear_equiv must REJECT non-context port folding: ${le.out}`);
});

test('REG-v3.9-04 (F-04): graph_mutate writes valid binary steps to --out on insertStep/replaceStep', async () => {
  const mut = writeTmpJson('reg_v39_04_mut.json', {
    commit: true,
    edits: [
      { op: 'replaceStep', index: 2, layer: { constructor: 'add', args: [99] }, slots: [3, 1, 4] },
    ],
  });
  const outF = path.join(TMP, 'reg_v39_04_out.json');
  const gm = runCli(['bin/graph_mutate.mjs', 'examples/linear-relu.json', mut, '--out', outF]);
  assert(gm.code === 0 && gm.json?.ok === true, `graph_mutate failed: ${gm.out}`);
  const val = runCli(['bin/validate.mjs', outF]);
  assert(val.code === 0 && val.json?.valid === true, `mutated --out spec failed validate.mjs: ${val.out}`);
});

test('REG-v3.9-05 (F-05): validate.mjs and propagateLayouts seed port layouts and warn on incompatible port-to-layer layout', async () => {
  const badPortLayoutSpec = {
    numSlots: 3,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_last' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [1, 0, 0, 1] },
    inputs: { '0': [1, 2], '1': [0, 0] },
  };
  const bf = writeTmpJson('reg_v39_05_layout.json', badPortLayoutSpec);
  const v = runCli(['bin/validate.mjs', bf]);
  assert(v.code === 0 && (v.json?.static_warnings ?? []).some((w) => w.includes('incompatible with required input layout')), `expected layout incompatibility warning on port 0: ${v.out}`);
  const ex = runCli(['bin/validate.mjs', '--explain', bf]);
  assert(ex.code === 0 && ex.json?.slots?.[0]?.layout === 'feature_last', `expected port 0 layout in --explain to be feature_last, got ${JSON.stringify(ex.json?.slots?.[0])}`);
});

test('REG-v3.9-06 (F-06): trace_hash and rollout bisect detect outputSlot mismatch and step count difference', async () => {
  const base = JSON.parse(fs.readFileSync(path.join(ROOT, 'examples/linear-relu.json'), 'utf8'));
  const diffOut = JSON.parse(JSON.stringify(base));
  diffOut.outputSlot = 2;
  const dof = writeTmpJson('reg_v39_06_diffout.json', diffOut);

  const th = runCli(['bin/trace_hash.mjs', 'examples/linear-relu.json', dof]);
  assert(th.code === 0 && th.json?.comparison?.hash_chain_equal === false && th.json?.comparison?.output_slot_equal === false, `trace_hash must report hash_chain_equal:false when outputSlot differs: ${th.out}`);

  const bis = runCli(['bin/rollout.mjs', 'bisect', 'examples/linear-relu.json', dof]);
  assert(bis.code === 0 && bis.json?.identical_trace === false && bis.json?.output_slot_equal === false, `rollout bisect must report identical_trace:false when outputSlot differs: ${bis.out}`);
});

test('REG-v3.9-07 (F-07): quant_budget bound mode remains >= measured delta when int8 quantization rounds weights up', async () => {
  const mulUpSpec = {
    numSlots: 5,
    layers: [
      { constructor: 'linear', args: [0, 2, 1, false] },
      { constructor: 'linear', args: [1, 2, 1, false] },
      { constructor: 'mul', args: [2] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [1, 3] },
      { kind: 'binary', layer: 2, slots: [2, 3, 4] },
    ],
    outputSlot: 4,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [0.99606299, 0.99606299],
      '1': [0.99606299, 0.99606299],
    },
    inputs: { '0': [1, 1], '1': [1, 1] },
  };
  const mf = writeTmpJson('reg_v39_07_mulup.json', mulUpSpec);
  const cf = writeTmpJson('reg_v39_07_cases.json', [mulUpSpec.inputs]);
  const qbBound = runCli(['bin/quant_budget.mjs', mf, cf, '--tol', '1.0', '--mode', 'bound']);
  const qbMeas = runCli(['bin/quant_budget.mjs', mf, cf, '--tol', '1.0', '--mode', 'measured']);
  assert(qbBound.code === 0 && qbBound.json?.ok === true && qbMeas.code === 0 && qbMeas.json?.ok === true, `quant_budget failed: ${qbBound.out} / ${qbMeas.out}`);
  assert(qbBound.json.totals.bound_total_all >= qbMeas.json.totals.measured_delta_plan, `unsound bound: ${qbBound.json.totals.bound_total_all} < ${qbMeas.json.totals.measured_delta_plan}`);
});

test('REG-v3.9-08 (F-08): weight_lab certify refuses mismatched probe wants and spectralRadius handles oscillatory matrices', async () => {
  const badProbeSet = writeTmpJson('reg_v39_08_badprobe.json', {
    name: 'bad_probe',
    kind: 'map',
    W: [[1, 0], [0, 1]],
    b: [0, 0],
    probes: [{ in: [10, 20], want: [9999, -9999] }],
  });
  const repTmp = path.join(TMP, 'reg_v39_08_rep.json');
  const certBad = runCli(['bin/weight_lab.mjs', 'certify', badProbeSet, '--repertoire', repTmp]);
  assert(certBad.code === 2 && certBad.json?.ok === false && certBad.json?.verdict === 'REFUSED', `weight_lab certify must refuse wrong probe want: ${certBad.out}`);

  const oscSet = writeTmpJson('reg_v39_08_osc.json', {
    name: 'osc_contraction',
    kind: 'dynamics',
    W: [[0, 5], [0.05, 0]],
    b: [0, 0],
    requires: { spectral_radius_lt: 1.0 },
  });
  const certOsc = runCli(['bin/weight_lab.mjs', 'certify', oscSet, '--repertoire', repTmp]);
  assert(certOsc.code === 0 && certOsc.json?.ok === true && certOsc.json?.verdict === 'CERTIFIED', `oscillatory contraction with true rho=0.5 should be CERTIFIED: ${certOsc.out}`);
  assert(Math.abs(certOsc.json.entry.properties[0].value - 0.5) < 1e-4, `expected rho=0.5, got ${certOsc.json.entry.properties[0].value}`);
});

test('REG-v3.9-09 (F-09): rollout fixpoint rejects constant-drift and expanding systems even when initial step <= tol', async () => {
  const driftSpec = {
    numSlots: 3,
    layers: [{ constructor: 'add', args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'state', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'context', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    inputs: { '0': [1.0, 2.0], '1': [5e-7, 5e-7] },
  };
  const df = writeTmpJson('reg_v39_09_drift.json', driftSpec);
  const res = runCli(['bin/rollout.mjs', 'fixpoint', df, '--tol', '1e-6', '--max-steps', '10']);
  assert(res.code === 2 && res.json?.converged === false && res.json?.status === 'NON_CONTRACTIVE_DRIFT', `constant drift must not converge: ${res.out}`);
});

test('REG-v3.9-10 (F-10): compose grad-spec supports matmul/shift* spatial outputs and overwritten intermediate --wrt slots', async () => {
  const mmSpec = {
    numSlots: 3,
    layers: [{ constructor: 'matmul', args: [0] }],
    steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, 2, 2], layout: 'any_rank4' },
      { slot: 1, role: 'observation', shape: [1, 1, 2, 2], layout: 'any_rank4' },
    ],
    inputs: { '0': [1, 2, 3, 4], '1': [5, 6, 7, 8] },
  };
  const mmf = writeTmpJson('reg_v39_10_mm.json', mmSpec);
  const gsMm = runCli(['bin/compose.mjs', 'grad-spec', mmf, '--wrt', '0']);
  assert(gsMm.code === 0 && gsMm.json?.ok === true && JSON.stringify(gsMm.json?.grad_values) === '[11,15,11,15]', `matmul grad-spec failed: ${gsMm.out}`);

  const ovwSpec = {
    numSlots: 4,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'linear', args: [1, 2, 2, false] },
      { constructor: 'linear', args: [2, 2, 2, false] },
      { constructor: 'add', args: [3] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'unary', layer: 1, slots: [1, 2] },
      { kind: 'unary', layer: 2, slots: [2, 3] },
      { kind: 'binary', layer: 3, slots: [3, 0, 3] },
    ],
    outputSlot: 3,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: {
      '0': [10, 0, 0, 10],
      '1': [2, 0, 0, 2],
      '2': [5, 0, 0, 5],
    },
    inputs: { '0': [1, 1], '1': [1, 1] },
  };
  const ovwf = writeTmpJson('reg_v39_10_ovw.json', ovwSpec);
  const gsOvw = runCli(['bin/compose.mjs', 'grad-spec', ovwf, '--wrt', '2']);
  assert(gsOvw.code === 0 && JSON.stringify(gsOvw.json?.grad_values) === '[5,5]', `overwritten slot 2 grad must be [5,5], got ${gsOvw.out}`);
});

test('REG-v3.9-11 (F-11): 1-layer es_optimize centers on startWeights and never degrades optimal start_fitness', async () => {
  const optSpec = {
    numSlots: 3,
    layers: [
      { constructor: 'linear', args: [0, 2, 2, false] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
      { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    ],
    weights: { '0': [2, 0, 0, 3] },
    inputs: { '0': [5, 7], '1': [0, 0] },
  };
  const targets = { cases: [{ inputs: { '0': [5, 7], '1': [0, 0] }, expected: [10, 21] }] };
  const sf = writeTmpJson('reg_v39_11_spec.json', optSpec);
  const tf = writeTmpJson('reg_v39_11_targets.json', targets);
  const es = runCli(['bin/es_optimize.mjs', sf, tf, '3']);
  assert(es.code === 0 && es.json?.ok === true, `es_optimize failed: ${es.out}`);
  assert(es.json.final_best_fitness >= es.json.start_fitness, `es_optimize degraded fitness: ${es.json.start_fitness} -> ${es.json.final_best_fitness}`);
  assert(es.json.final_best_fitness === 0, `expected optimal fitness 0 preserved, got ${es.json.final_best_fitness}`);
});

test('REG-v3.9-12 (F-12): autodiff verify-fd emits structured JSON error on unsupported op and compose prove checks output shapes', async () => {
  const unsupSpec = {
    numSlots: 3,
    layers: [
      { constructor: 'maxPool2d', args: [0, 2, 2, 1, 1, 0, 0] },
      { constructor: 'add', args: [1] },
    ],
    steps: [
      { kind: 'unary', layer: 0, slots: [0, 2] },
      { kind: 'binary', layer: 1, slots: [2, 1, 2] },
    ],
    outputSlot: 2,
    ports: [
      { slot: 0, role: 'observation', shape: [1, 1, 2, 2], layout: 'channel_first' },
      { slot: 1, role: 'observation', shape: [1, 1, 1, 1], layout: 'channel_first' },
    ],
    inputs: { '0': [1, 2, 3, 4], '1': [0] },
  };
  const uf = writeTmpJson('reg_v39_12_unsup.json', unsupSpec);
  const fd = runCli(['bin/autodiff.mjs', 'verify-fd', uf]);
  assert(fd.code !== 0 && fd.json && fd.json.ok === false && typeof fd.json.error === 'string', `verify-fd must return JSON error on stdout, got: ${fd.out}`);
});

test('REG-fix-C1: attention_head is layout-explicit (seq_axis1 pinned, heads_axis1 true attention)', async () => {
  const close = (a, b, tol) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= tol);
  const runMacro = (inv, name) => {
    const f = writeTmpJson(name, inv);
    const e = runCli(['bin/macro_expand.mjs', 'expand', f]);
    assert(e.code === 0 && e.json?.spec, `attention_head expand failed: ${e.out}`);
    const s = writeTmpJson(name.replace('.json', '_spec.json'), e.json.spec);
    const r = runCli(['bin/run.mjs', s]);
    assert(r.code === 0 && r.json?.ok === true, `attention_head run failed: ${r.out}`);
    return r.json.outputs.values;
  };
  // seq_axis1 (default, pinned by incident test 2026-10-22): [B,S,D,*]
  // inputs, outer-product attention, ref ~[13.2082, 10.4743].
  const vA = runMacro({ macro: 'attention_head', params: { d_model: 1, q_slot: 0, k_slot: 1, v_slot: 2, out_slot: 3 }, weights: {}, ports: [0, 1, 2].map((slot) => ({ slot, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' })), inputs: { '0': [1, 4], '1': [1, 0.25], '2': [10, 20] } }, 'fix_c1_a.json');
  assert(close(vA, [13.2082, 10.4743], 1e-3), `seq_axis1 attention wrong: ${JSON.stringify(vA)}`);
  // heads_axis1 ([B,H,S,D]): H=1 discriminator — the old default chain
  // returned V exactly [9,10,11,12] on this fixture.
  const ports1 = [
    { slot: 0, role: 'observation', shape: [1, 1, 2, 2], layout: 'feature_last' },
    { slot: 1, role: 'observation', shape: [1, 1, 2, 2], layout: 'feature_last' },
    { slot: 2, role: 'observation', shape: [1, 1, 2, 2], layout: 'feature_last' },
  ];
  const vB = runMacro({ macro: 'attention_head', params: { d_model: 2, q_slot: 0, k_slot: 1, v_slot: 2, out_slot: 3, layout: 'heads_axis1' }, weights: {}, ports: ports1, inputs: { '0': [1, 2, 3, 4], '1': [5, 6, 7, 8], '2': [9, 10, 11, 12] } }, 'fix_c1_b.json');
  assert(close(vB, [10.995054753686732, 11.995054753686732, 10.999998336943943, 11.999998336943943], 1e-4), `heads_axis1 attention wrong: ${JSON.stringify(vB)}`);
  assert(!close(vB, [9, 10, 11, 12], 0.1), `heads_axis1 output equals V (old default-chain bug): ${JSON.stringify(vB)}`);
  // heads_axis1 cross-attention Sq=2, Skv=3.
  const vC = runMacro({ macro: 'attention_head', params: { d_model: 2, q_slot: 0, k_slot: 1, v_slot: 2, out_slot: 3, layout: 'heads_axis1' }, weights: {}, ports: [
    { slot: 0, role: 'observation', shape: [1, 1, 2, 2], layout: 'feature_last' },
    { slot: 1, role: 'observation', shape: [1, 1, 3, 2], layout: 'feature_last' },
    { slot: 2, role: 'observation', shape: [1, 1, 3, 2], layout: 'feature_last' },
  ], inputs: { '0': [1, 0, 0, 1], '1': [1, 2, 0, 1, 2, 0], '2': [10, 0, 0, 20, 30, 40] } }, 'fix_c1_cross.json');
  assert(close(vC, [22.40451338379263, 28.41024969440048, 9.353326752859632, 8.49579234791117], 1e-4), `heads_axis1 cross-attention wrong: ${JSON.stringify(vC)}`);
  // Unknown layout fails closed with a named error.
  const bad = writeTmpJson('fix_c1_badlayout.json', { macro: 'attention_head', params: { d_model: 2, q_slot: 0, k_slot: 1, v_slot: 2, out_slot: 3, layout: 'bhsd' }, weights: {}, ports: ports1, inputs: { '0': [1, 2, 3, 4], '1': [5, 6, 7, 8], '2': [9, 10, 11, 12] } });
  const rb = runCli(['bin/macro_expand.mjs', 'expand', bad]);
  assert(rb.code !== 0 && /layout must be/.test(rb.out + rb.err), `unknown layout not rejected: ${rb.out}`);
});

test('REG-fix-T1: workflow grad_spec rejects order other than 1 or 2', async () => {
  const spec = writeTmpJson('fix_t1_spec.json', { numSlots: 2, layers: [{ constructor: 'matmul', args: [0] }], steps: [{ kind: 'binary', layer: 0, slots: [0, 0, 1] }], outputSlot: 1, ports: [{ slot: 0, role: 'observation', shape: [1, 1, 2, 2], layout: 'any_rank4' }], inputs: { '0': [1, 2, 3, 4] } });
  const ws = path.join(TMP, 'ws_fix_t1');
  const bad3 = writeTmpJson('fix_t1_o3.json', { spec, pipeline: [{ stage: 'grad_spec', order: 3, wrt_slot: 0 }] });
  const r3 = runCli(['bin/harness.mjs', 'run-workflow', bad3, '--workspace', ws]);
  assert(r3.code === 2 && r3.json?.ok === false && r3.json?.failed_stage === 'grad_spec' && /order must be 1 or 2/.test(r3.json?.error ?? ''), `order 3 not rejected: ${r3.out}`);
  const badStr = writeTmpJson('fix_t1_ostr.json', { spec, pipeline: [{ stage: 'grad_spec', order: '2', wrt_slot: 0 }] });
  const rs = runCli(['bin/harness.mjs', 'run-workflow', badStr, '--workspace', ws]);
  assert(rs.code === 2 && rs.json?.ok === false && rs.json?.failed_stage === 'grad_spec', `string order not rejected: ${rs.out}`);
  const ok2 = writeTmpJson('fix_t1_o2.json', { spec, pipeline: [{ stage: 'grad_spec', order: 2, wrt_slot: 0 }] });
  const r2 = runCli(['bin/harness.mjs', 'run-workflow', ok2, '--workspace', ws]);
  assert(r2.code === 0 && r2.json?.ok === true, `order 2 rejected: ${r2.out}`);
});

test('REG-fix-Z1: zoom-vs reports output length mismatch instead of 0', async () => {
  const ports2 = [
    { slot: 0, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
    { slot: 1, role: 'observation', shape: [1, 2, 1, 1], layout: 'feature_axis1_singleton' },
  ];
  const base = writeTmpJson('fix_z1_base.json', { numSlots: 8, layers: [{ constructor: 'add', args: [0] }], steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }], outputSlot: 2, ports: ports2, weights: {}, inputs: { '0': [0, 0], '1': [0, 0] } });
  const cand = writeTmpJson('fix_z1_cand.json', { numSlots: 8, layers: [{ constructor: 'add', args: [0] }, { constructor: 'concat', args: [1, 1] }], steps: [{ kind: 'binary', layer: 0, slots: [0, 1, 2] }, { kind: 'binary', layer: 1, slots: [2, 2, 3] }], outputSlot: 3, ports: ports2, weights: {}, inputs: { '0': [0, 0], '1': [0, 0] } });
  const z = runCli(['bin/fuzz_cases.mjs', base, '--zoom-vs', cand, '--zoom-depth', '3', '--count', '6']);
  assert(z.code === 0 && z.json?.ok === true, `zoom-vs failed: ${z.out}`);
  assert(z.json?.length_mismatch === true, `length mismatch not flagged: ${z.out}`);
  assert(z.json?.output_lengths?.base === 2 && z.json?.output_lengths?.candidate === 4, `output lengths wrong: ${z.out}`);
  assert(z.json?.zoomed_max_abs_error === 'Infinity', `mismatch reported as finite delta (old bug reported 0): ${z.out}`);
  const same = runCli(['bin/fuzz_cases.mjs', base, '--zoom-vs', base, '--zoom-depth', '3', '--count', '6']);
  assert(same.code === 0 && same.json?.ok === true && same.json?.length_mismatch === false && same.json?.zoomed_max_abs_error === 0, `same-shape control broken: ${same.out}`);
});

test('REG-fix-F2: unknown flags fail closed with exit 2 and never become files', async () => {
  const noStray = (name) => assert(!fs.existsSync(path.join(ROOT, name)), `stray file '${name}' created in pack root`);
  const unknown = (args, flag) => {
    const r = runCli(args);
    assert(r.code === 2 && (r.err ?? '').includes(`unknown flag '${flag}'`), `unknown flag ${flag} not rejected: code=${r.code} err=${r.err} out=${r.out}`);
    return r;
  };
  noStray('2'); noStray('1'); noStray('out.json');
  // The two original repros (bedah-2 track D): values of unknown flags used to leak into positionals.
  unknown(['bin/recurse.mjs', 'search', 'examples/linear-relu.json', '--beam', '2'], '--beam');
  noStray('2');
  unknown(['bin/recurse.mjs', 'search', 'examples/linear-relu.json', '--depth', '1'], '--depth');
  noStray('1');
  // Same class in every other bin that derives a path or parameter from the token stream.
  unknown(['bin/harness.mjs', 'capabilities', '--bogus', '5'], '--bogus');
  const f2roll = path.join(TMP, 'fix_f2_roll.json');
  unknown(['bin/rollout.mjs', 'unroll', 'examples/linear-relu.json', f2roll, '--bogus', '2'], '--bogus');
  assert(!fs.existsSync(f2roll), 'rollout wrote output despite unknown flag');
  const f2comp = path.join(TMP, 'fix_f2_comp.json');
  unknown(['bin/compose.mjs', 'compile', 'examples/recursive-gated-residual.v2.json', f2comp, '--bogus', '2'], '--bogus');
  assert(!fs.existsSync(f2comp), 'compose wrote output despite unknown flag');
  unknown(['bin/es_optimize.mjs', 'examples/linear-relu.json', 'no-such-targets.json', '--bogus', '5'], '--bogus');
  unknown(['bin/quant_budget.mjs', 'examples/linear-relu.json', 'no-such-cases.json', '--bogus', '1'], '--bogus');
  unknown(['bin/fuzz_cases.mjs', 'examples/linear-relu.json', '--bogus', 'out.json'], '--bogus');
  noStray('out.json');
  unknown(['bin/graph_mutate.mjs', 'examples/linear-relu.json', 'no-such-mut.json', '--bogus', '5'], '--bogus');
  unknown(['bin/geom_batch.mjs', 'no-such-model.json', '--bogus', 'evil.json'], '--bogus');
  unknown(['bin/geom_project.mjs', 'no-such-model.json', '--bogus', 'evil.json'], '--bogus');
  unknown(['bin/geom_build.mjs', 'no-such-chain.json', '--bogus', 'out.json'], '--bogus');
  unknown(['bin/rewrite.mjs', 'apply', 'all', 'examples/linear-relu.json', '--bogus'], '--bogus');
  // Positive controls: the known flags these cases mistyped still work.
  const okSearch = runCli(['bin/recurse.mjs', 'search', 'examples/linear-relu.json', '--max-depth', '1']);
  assert(okSearch.code === 0 && okSearch.json?.ok === true, `known flag --max-depth rejected: ${okSearch.out}`);
  const okRoll = runCli(['bin/rollout.mjs', 'unroll', 'examples/linear-relu.json', path.join(TMP, 'fix_f2_ok.json'), '--steps', '2', '--state-port', '0']);
  assert(okRoll.code === 0 && okRoll.json?.ok === true, `known rollout flags rejected: ${okRoll.out}`);
});

const results = [];
for (const c of cases) {
  try {
    await c.fn();
    results.push({ name: c.name, ok: true });
  } catch (e) {
    results.push({ name: c.name, ok: false, detail: e instanceof Error ? e.message : String(e) });
  }
}

try { fs.rmSync(TMP, { recursive: true, force: true }); } catch { /* ignore */ }
const passed = results.filter((r) => r.ok).length;
printJsonAndExit({
  ok: passed === results.length,
  suite: 'regression_recursive.v3.9',
  passed,
  failed: results.length - passed,
  total: results.length,
  cases: results,
}, passed === results.length ? 0 : 1);
