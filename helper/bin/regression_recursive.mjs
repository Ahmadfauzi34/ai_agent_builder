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

// --- 5. STAGE 2: RECURSIVE UPGRADES ACROSS ALL EXISTING FEATURES ---

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

// --- 6. STAGE 3: EXISTING-FEATURE RECURSIVE HARMONIZATION AUDIT (v3.6) ---

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

// --- 7. STAGE 4: SECOND-PASS DEEP OBSERVATION AUDIT (v3.7) ---

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

// --- 9. STAGE 6: v3.9 TOOL HONESTY & FAIL-CLOSED INTEGRITY REGRESSIONS (F-01..F-12) ---

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
