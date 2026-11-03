// burn-math Layer 4: Agent Harness Integration, Semantic Digest, Artifact Store & Pipeline Engine (lib/harness.mjs)

import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { withGraphScope } from './wasm_direct.mjs';
import { portBySlot, hydrateSpecArtifactsInPlace, isRecursiveSpecV2 } from './spec.mjs';
import { explainSpec } from './explain.mjs';
import { compileRecursiveSpec, compileBackwardSpec, autoRepairSpec } from './composer.mjs';
import { rolloutScan, rolloutFixpoint, unrollRolloutSpec } from './rollout.mjs';
import { searchGraphTree, optimizeMultiLayerES, reflectiveGdSolve, verifyAndStageCandidate } from './recurse.mjs';
import { buildGraph, makeBundle } from './engine.mjs';

export const HARNESS_SCHEMA = 'burn-math.harness-session.v1';
export const ARTIFACT_URI_PREFIX = 'burn://sha256:';

const clone = (o) => JSON.parse(JSON.stringify(o));
const sha256Hex = (bufOrStr) => createHash('sha256').update(bufOrStr).digest('hex');

export function ensureWorkspaceDir(workspaceDir = '.burn-harness') {
  const abs = path.resolve(workspaceDir);
  const artifactsDir = path.join(abs, 'artifacts');
  const checkpointsDir = path.join(abs, 'checkpoints');
  const branchesDir = path.join(abs, 'branches');
  fs.mkdirSync(artifactsDir, { recursive: true });
  fs.mkdirSync(checkpointsDir, { recursive: true });
  fs.mkdirSync(branchesDir, { recursive: true });
  const stateFile = path.join(abs, 'workspace-state.json');
  if (!fs.existsSync(stateFile)) {
    const initial = {
      schema: HARNESS_SCHEMA,
      created_at: new Date().toISOString(),
      tables: {
        search_frontier: {},
        artifacts: {},
        proofs: [],
        events: [],
      },
    };
    fs.writeFileSync(stateFile, JSON.stringify(initial, null, 2) + '\n');
  }
  return { abs, artifactsDir, checkpointsDir, branchesDir, stateFile };
}

export function storeArtifact(values, shape, workspaceDir = '.burn-harness', kind = 'tensor_f32') {
  const { artifactsDir, stateFile } = ensureWorkspaceDir(workspaceDir);
  const payload = JSON.stringify({ kind, shape, values: Array.from(values) });
  const digest = sha256Hex(payload);
  const file = path.join(artifactsDir, `${digest}.json`);
  if (!fs.existsSync(file)) {
    fs.writeFileSync(file, payload + '\n');
  } else {
    // Minor fix: a tampered file used to survive re-store forever
    // (same digest name, write skipped) while reads kept failing
    // integrity. Verify and repair on mismatch.
    try {
      if (fs.readFileSync(file, 'utf8') !== payload + '\n') fs.writeFileSync(file, payload + '\n');
    } catch { fs.writeFileSync(file, payload + '\n'); }
  }
  const uri = `${ARTIFACT_URI_PREFIX}${digest}`;
  try {
    const st = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
    st.tables.artifacts[digest] = { uri, kind, shape, numel: values.length, bytes: Buffer.byteLength(payload) };
    fs.writeFileSync(stateFile, JSON.stringify(st, null, 2) + '\n');
  } catch { /* non-fatal metadata update */ }
  return { uri, sha256: `sha256:${digest}`, file };
}

export function resolveArtifactUri(uri, workspaceDir = '.burn-harness') {
  if (typeof uri !== 'string' || !uri.startsWith(ARTIFACT_URI_PREFIX)) {
    throw new Error(`invalid artifact URI: ${JSON.stringify(uri)} (expected ${ARTIFACT_URI_PREFIX}<64-hex>)`);
  }
  const expectedHex = uri.slice(ARTIFACT_URI_PREFIX.length);
  if (!/^[0-9a-f]{64}$/.test(expectedHex)) {
    throw new Error(`malformed sha256 in artifact URI: ${uri}`);
  }
  const { artifactsDir } = ensureWorkspaceDir(workspaceDir);
  const file = path.join(artifactsDir, `${expectedHex}.json`);
  if (!fs.existsSync(file)) {
    throw new Error(`artifact not found in workspace: ${uri}`);
  }
  const raw = fs.readFileSync(file, 'utf8').trim();
  const actualHex = sha256Hex(raw);
  if (actualHex !== expectedHex) {
    const err = new Error(`artifact integrity check failed for ${uri}: expected ${expectedHex}, got ${actualHex}`);
    err.code = 'artifact_integrity_failed';
    throw err;
  }
  return JSON.parse(raw);
}

export function hydrateSpecArtifacts(rawSpec, workspaceDir = '.burn-harness', baseDir = process.cwd()) {
  const s = clone(rawSpec);
  return hydrateSpecArtifactsInPlace(s, baseDir, workspaceDir);
}

export function computeSemanticDigest(values, shape = null, {
  workspaceDir = '.burn-harness',
  persistThreshold = 16,
  includeFullValues = false,
} = {}) {
  const arr = Array.from(values ?? []);
  const n = arr.length;
  const f32 = Float32Array.from(arr);
  const f32Bytes = Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
  const valueSha256 = `sha256:${sha256Hex(f32Bytes)}`;

  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  let sumSq = 0;
  let zeroCount = 0;
  let nanCount = 0;
  let infCount = 0;

  for (let i = 0; i < n; i++) {
    const v = arr[i];
    if (Number.isNaN(v)) { nanCount++; continue; }
    if (!Number.isFinite(v)) { infCount++; continue; }
    if (v === 0) zeroCount++;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
    sumSq += v * v;
  }

  const finiteCount = n - nanCount - infCount;
  const mean = finiteCount > 0 ? sum / finiteCount : null;
  const variance = finiteCount > 0 ? Math.max(0, sumSq / finiteCount - mean * mean) : null;
  const std = variance !== null ? Math.sqrt(variance) : null;
  const l2Norm = Math.sqrt(sumSq);
  const lInfNorm = finiteCount > 0 ? Math.max(Math.abs(min), Math.abs(max)) : null;

  let artifactRef = null;
  if (workspaceDir && n >= persistThreshold) {
    const stored = storeArtifact(arr, shape ?? [1, n, 1, 1], workspaceDir, 'tensor_f32');
    artifactRef = stored.uri;
  }

  return {
    shape: shape ?? [1, n, 1, 1],
    numel: n,
    value_sha256: valueSha256,
    stats: {
      min: finiteCount > 0 ? +min.toPrecision(7) : null,
      max: finiteCount > 0 ? +max.toPrecision(7) : null,
      mean: mean !== null ? +mean.toPrecision(7) : null,
      std: std !== null ? +std.toPrecision(7) : null,
      l2_norm: +l2Norm.toPrecision(7),
      l_inf_norm: lInfNorm !== null ? +lInfNorm.toPrecision(7) : null,
      zero_count: zeroCount,
      nan_count: nanCount,
      inf_count: infCount,
    },
    preview_head: arr.slice(0, 8),
    artifact_ref: artifactRef,
    values: (includeFullValues || n <= persistThreshold) ? arr : undefined,
  };
}

export const CAPABILITY_DOMAINS = {
  compose: {
    summary: 'Build, expand, and lower hierarchical/recursive specs, macros, 3D kinematic trees, weight expression trees, and symbolic backward graphs.',
    tools: [
      { bin: 'bin/compose.mjs', ops: ['compile', 'prove', 'grad-spec'], role: 'Recursive Spec v2 compiler (nested subgraphs, repeat, residual, parallel_concat, auto-repair, --compact-slots) & symbolic backward spec generator (incl. div/sqrt/sigmoid/tanh HVP)' },
      { bin: 'bin/macro_expand.mjs', ops: ['expand', 'prove'], role: 'Macro expander with recursive depth unrolling (mlp, gated_unit, residual_mlp, transformer_ffn, recurrent_cell, attention_head)' },
      { bin: 'bin/shard_spec.mjs', ops: ['<spec.json> [maxWeights]'], role: 'Split oversized linear layers (>u16 cap) into block linears + concat chain with exact reassembly proof' },
      { bin: 'bin/geom_build.mjs', ops: ['<chain.json>', '--scene-tree <tree.json>', '--solve-ik <ik.json>'], role: 'Compile 3D transform chains (incl. repeat op), hierarchical kinematic trees, and inverse kinematics into runnable engine specs' },
      { bin: 'bin/weight_lab.mjs', ops: ['run', 'gate', 'certify', 'compose', 'eval'], role: 'Analyze, compose, certify, and evaluate recursive weight expression trees over square and rectangular m x n matrices' },
    ],
  },
  verify: {
    summary: 'Validate plans, prove numerical & algebraic equivalence, localize step divergences, zoom-fuzz worst cases, and certify determinism.',
    tools: [
      { bin: 'bin/validate.mjs', ops: ['<spec.json>', '--explain <spec.json>'], role: '3-layer static/type/port validation + slot liveness & peak memory map (supports v1 & v2 specs)' },
      { bin: 'bin/verify.mjs', ops: ['<base.json> <cand.json>', '--fuzz', '--batch'], role: 'Graph-vs-graph numerical equivalence verifier over test cases or fuzzing' },
      { bin: 'bin/trace_hash.mjs', ops: ['<spec.json> [otherSpec.json] [--isolate]'], role: 'Per-step output SHA-256 fingerprinting, first-divergent-step localization, and prefix subspec isolation' },
      { bin: 'bin/diff_ref.mjs', ops: ['<spec.json>'], role: 'Differential testing against independent 34-constructor f64 JS reference implementation' },
      { bin: 'bin/soak.mjs', ops: ['<spec.json> [N=100]'], role: 'Bit-identical determinism soak proof across N fresh runs' },
      { bin: 'bin/shapecheck.mjs', ops: ['<spec.json>'], role: 'Fast static shape and dataflow checker' },
      { bin: 'bin/fuzz_cases.mjs', ops: ['<spec.json> [out.json]', '--zoom-vs <cand.json>'], role: 'Deterministic PRNG test-case generator + recursive adversarial interval narrowing (Zoom Fuzzing)' },
      { bin: 'bin/prng_check.mjs', ops: ['[algo] [seed] [count]', 'verify <manifest.json>'], role: 'Certify deterministic PRNG streams' },
      { bin: 'bin/linear_equiv.mjs', ops: ['<specA.json> <specB.json>'], role: 'Prove exact algebraic equivalence of linear, residual, branched, and unrolled recursive affine graphs' },
      { bin: 'bin/regression.mjs', ops: [], role: '40-case core release regression test suite' },
      { bin: 'bin/regression_recursive.mjs', ops: [], role: 'Recursive, Stage-2, Stage-3, Stage-4 & v3.8 audit regression test suite' },
      { bin: 'bin/engine_hardening_test.mjs', ops: [], role: '9-case checkOperation & withTensorScope hardening regression suite' },
    ],
  },
  differentiate: {
    summary: 'Compute exact reverse-mode VJPs (including BPTT over weight-tied & slot-compacted graphs), compile higher-order gradient graphs, and rank input/weight sensitivity.',
    tools: [
      { bin: 'bin/autodiff.mjs', ops: ['map', 'grad', 'prove-rules', 'verify-fd'], role: 'Audited 31-rule reverse-mode VJP map with BPTT weight-tied accumulation verified against finite differences' },
      { bin: 'bin/autodiff_prove.mjs', ops: [], role: 'Dual-oracle (hand-computed + finite-difference) proof runner for all 31 autodiff VJP rules' },
      { bin: 'bin/compose.mjs', ops: ['grad-spec --order 1|2'], role: 'Compile 1st-order VJP or 2nd-order Hessian-Vector Product into a standalone runnable WASM spec' },
      { bin: 'bin/gradcheck.mjs', ops: ['<spec.json> [analytic.json] [eps] [tol] [--weights]'], role: 'Central finite-difference Jacobian oracle over input ports and layer weights' },
      { bin: 'bin/fit_sensitivity.mjs', ops: ['<spec.json> [topK] [threshold] [--weights]'], role: 'Rank input-slot and layer-weight sensitivity from Jacobian norms' },
      { bin: 'bin/fit_hessian_diag.mjs', ops: ['<spec.json> [slot | --layer <idx>] [h]'], role: 'Second-difference Hessian diagonal estimator over input slots or layer weights with trust labels' },
    ],
  },
  optimize: {
    summary: 'Search rewrite trees, apply recursive fixpoint rewrites, mutate graphs transactionally or across competing branches, quantize DAGs within error budgets, and solve parameters via ES or closed-loop reflective GD.',
    tools: [
      { bin: 'bin/recurse.mjs', ops: ['search', 'fit', 'solve-gd [--layer 0|all] [--multi-layer]'], role: 'Recursive rewrite + quantization tree search over CheckpointBranchSet, multi-layer ES fitting, & single/multi-layer backtracking GD solver' },
      { bin: 'bin/rewrite.mjs', ops: ['list', 'apply <name|all> <spec.json> [--recursive] [--compact-slots] [--verify]'], role: 'Apply single or recursive fixpoint proof-backed graph rewrites' },
      { bin: 'bin/graph_mutate.mjs', ops: ['<spec.json> <mutation.json> [--out out.json]'], role: 'Transactional & multi-branch (CheckpointBranchSet) equivalence-gated graph mutation' },
      { bin: 'bin/opt_compare.mjs', ops: ['<base.json> <cand.json> [--tol <tol>]'], role: 'Equivalence + Fase A memory/latency oracle comparison with customizable tolerance' },
      { bin: 'bin/opt_profile.mjs', ops: ['<spec.json> [iters] --trace'], role: 'Per-phase build/bind/run timing, WASM page deltas, and step hotspot map' },
      { bin: 'bin/opt_guard.mjs', ops: ['<file> | --bytes N'], role: 'Probe engine tiered bincode byte-limit guard' },
      { bin: 'bin/cost_model.mjs', ops: ['<spec.json> [--iters N] [--table out.json]'], role: 'Layout-aware per-op execution cost model' },
      { bin: 'bin/quant_budget.mjs', ops: ['<spec.json> <cases.json> --tol <num> --mode bound|measured'], role: 'Compositional DAG Lipschitz bound (22 constructors incl. mul/norms) & empirical greedy int8 quantization planner' },
      { bin: 'bin/quant_verify.mjs', ops: ['<spec.json> [--int8|--fp16]'], role: 'Weight quantization simulation and error bound check' },
      { bin: 'bin/quant_equiv.mjs', ops: ['--in <N> --out <M>'], role: 'Verify real engine quantized state path (getQuantizedState -> loadState)' },
      { bin: 'bin/es_optimize.mjs', ops: ['<spec.json> <targets.json> [gens] [--multi-layer] [--sensitivity-guided]'], role: 'Evolution Strategies parameter optimization (single- and multi-layer)' },
      { bin: 'bin/fit_gd_verify.mjs', ops: ['<spec.json> <trajectory.json> [tol] [--layer <idx|all>] [--targets <targets.json>]'], role: 'Audit claimed gradient-descent trajectory over input slots, single layer, or joint multi-layer weights' },
    ],
  },
  rollout: {
    summary: 'Execute graphs one-shot, recurrently across time (scan), iteratively to convergence (fixpoint), unroll temporal horizons into weight-tied graphs (unroll), or run direct kernels with batch register chaining ($prev / $<idx>).',
    tools: [
      { bin: 'bin/rollout.mjs', ops: ['scan', 'fixpoint', 'unroll', 'bisect'], role: 'Recurrent state-handoff rollout, fixed-point solver with Lipschitz tracking, temporal horizon unroller, and trace bisection' },
      { bin: 'bin/run.mjs', ops: ['[--direct] <spec.json> [...]'], role: 'Validate and execute one or more specs (v1, v2, f32, or mixed f32+i64 segments)' },
      { bin: 'bin/arith.mjs', ops: ['<op> <a1,...> [b1,...]', '--batch <ops.json>'], role: 'Direct WASM f32 vector/matrix/statistical/probability arithmetic with $prev/$<idx> batch chaining' },
      { bin: 'bin/int.mjs', ops: ['<add|sub|mul|div|rem|matmul|to-f32|from-f32>', '--batch <ops.json>'], role: 'Exact i64 integer arithmetic with fail-closed overflow guard and $prev/$<idx> batch chaining' },
      { bin: 'bin/kernel.mjs', ops: ['<sin|cos|tan|floor|ceil|round|atan2|powf|hypot|fma>', '--batch <ops.json>'], role: 'Elementwise transcendental and IEEE-754 half-to-even rounding kernels with $prev/$<idx> batch chaining' },
      { bin: 'bin/geom_project.mjs', ops: ['<model.json>'], role: 'End-to-end 3D model or scene-tree to 2D screen projection' },
      { bin: 'bin/geom_batch.mjs', ops: ['<batch.json>'], role: 'Batched 3D model or scene-tree to 2D projection' },
    ],
  },
  state: {
    summary: 'Manage persistent harness workspaces, checkpoints, per-layer states, and catalog queries.',
    tools: [
      { bin: 'bin/harness.mjs', ops: ['capabilities', 'digest', 'resolve-artifact', 'run-workflow', 'branch-export', 'branch-merge', 'workspace-status'], role: 'Unified agent-harness co-processor interface, pipeline runner, and subagent coordinator' },
      { bin: 'bin/checkpoint.mjs', ops: ['export', 'verify', 'restore'], role: 'Stateful ProgramBundle export, canonical base64 SHA-256 verification, and restore' },
      { bin: 'bin/layer_state.mjs', ops: ['inspect', 'get', 'set'], role: 'Per-layer binary state inspection (with weight-tying detection) and transplant' },
      { bin: 'bin/find-op.mjs', ops: ['<constructor>', '--list'], role: 'Query the 52 constructible layer contracts' },
      { bin: 'bin/surface.mjs', ops: ['<symbol>', '--list'], role: 'Query the 169 WASM surface functions, classes, and method signatures' },
      { bin: 'bin/layout.mjs', ops: ['<constructor>', '--list'], role: 'Query the 52 rank-4 tensor layout contracts' },
    ],
  },
};

export function getCapabilityCatalog(domain = null) {
  if (domain && CAPABILITY_DOMAINS[domain]) {
    const v = CAPABILITY_DOMAINS[domain];
    return { ok: true, domain, summary: v.summary, tool_count: v.tools.length, tools: v.tools };
  }
  if (domain) {
    // Minor fix: an unknown domain used to silently return the full
    // catalog. Now it fails closed with the known domain list.
    return { ok: false, error: `unknown capability domain "${domain}" (known: ${Object.keys(CAPABILITY_DOMAINS).join(', ')})` };
  }
  return {
    ok: true,
    domains: Object.fromEntries(
      Object.entries(CAPABILITY_DOMAINS).map(([k, v]) => [
        k,
        { summary: v.summary, tool_count: v.tools.length, tools: v.tools },
      ]),
    ),
  };
}

export async function executeHarnessWorkflow(rt, workflow, { baseDir = process.cwd() } = {}) {
  const workspaceDir = path.resolve(baseDir, workflow.workspace ?? '.burn-harness');
  const { stateFile, checkpointsDir } = ensureWorkspaceDir(workspaceDir);

  let rawSpec;
  if (typeof workflow.spec === 'string') {
    const specPath = path.resolve(baseDir, workflow.spec);
    rawSpec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
    baseDir = path.dirname(specPath);
  } else if (workflow.spec && typeof workflow.spec === 'object') {
    rawSpec = clone(workflow.spec);
  } else {
    throw new Error('workflow must provide "spec" (file path or inline spec object)');
  }

  rawSpec = hydrateSpecArtifacts(rawSpec, workspaceDir, baseDir);

  const wasmWorkspace = new rt.AgentWorkspace(64);
  const stageReports = [];
  let activeSpec = null;
  let lastRunDigest = null;
  // B8 fix: partial-failure support. A stage failure used to throw
  // out of this function, discarding stage_reports and leaving
  // workspace-state.json at its initial state (the throw skipped
  // both the return and the sync below). syncState is now shared by
  // the success path and the failure path; errors before the stage
  // loop (no stage ran) still throw.
  const syncState = () => {
    const wsSnap = JSON.parse(wasmWorkspace.snapshot());
    const diskState = JSON.parse(fs.readFileSync(stateFile, 'utf8'));
    diskState.updated_at = new Date().toISOString();
    diskState.wasm_workspace = wsSnap;
    fs.writeFileSync(stateFile, JSON.stringify(diskState, null, 2) + '\n');
  };
  let failedStageName = null;
  let inStages = false;

  try {
    const ensureCompiled = (autoRepair = true) => {
      if (!activeSpec) {
        if (isRecursiveSpecV2(rawSpec)) {
          activeSpec = compileRecursiveSpec(rawSpec, { baseDir, rt, useWorkspace: true, autoRepair, validate: true });
          hydrateSpecArtifactsInPlace(activeSpec, baseDir, workspaceDir);
        } else {
          activeSpec = autoRepair ? autoRepairSpec(rawSpec).spec : clone(rawSpec);
        }
      }
      return activeSpec;
    };

    for (let idx = 0; idx < (workflow.pipeline ?? []).length; idx++) {
      const st = workflow.pipeline[idx];
      const stageName = st.stage ?? st.op;
      failedStageName = stageName;
      inStages = true;

      if (stageName === 'compile' || stageName === 'compile_v2') {
        activeSpec = compileRecursiveSpec(rawSpec, {
          baseDir,
          rt,
          useWorkspace: true,
          autoRepair: st.auto_repair ?? true,
          compactSlots: st.compact_slots ?? false,
          validate: true,
        });
        hydrateSpecArtifactsInPlace(activeSpec, baseDir, workspaceDir);
        wasmWorkspace.recordEvent('stage:compile', String(idx), `layers=${activeSpec.layers.length};steps=${activeSpec.steps.length}`);
        stageReports.push({
          stage: stageName,
          ok: true,
          layers_count: activeSpec.layers.length,
          steps_count: activeSpec.steps.length,
          num_slots: activeSpec.numSlots,
          remediation_receipt: activeSpec.meta?.remediation_receipt ?? [],
        });
      } else if (stageName === 'validate') {
        const sp = ensureCompiled(st.auto_repair ?? false);
        const exp = explainSpec(sp);
        wasmWorkspace.recordEvent('stage:validate', String(idx), `valid=${exp.valid};peak_bytes=${exp.liveness?.peak_bytes ?? 0}`);
        stageReports.push({
          stage: 'validate',
          ok: exp.valid,
          totals: exp.totals,
          peak_live_bytes: exp.liveness?.peak_bytes ?? null,
        });
      } else if (stageName === 'rollout_unroll' || stageName === 'unroll') {
        const sp = ensureCompiled(true);
        activeSpec = unrollRolloutSpec(sp, {
          steps: st.steps ?? 3,
          statePort: st.state_port ?? null,
          compactSlots: st.compact_slots ?? false,
        });
        wasmWorkspace.recordEvent('stage:unroll', String(idx), `horizon=${st.steps ?? 3};steps=${activeSpec.steps.length}`);
        stageReports.push({
          stage: stageName,
          ok: true,
          temporal_horizon: st.steps ?? 3,
          layers_count: activeSpec.layers.length,
          steps_count: activeSpec.steps.length,
          num_slots: activeSpec.numSlots,
        });
      } else if (stageName === 'search_rewrites' || stageName === 'recursive_rewrite') {
        const sp = ensureCompiled(true);
        const searchRes = await searchGraphTree(rt, sp, {
          maxDepth: st.max_depth ?? 3,
          quantTol: st.quant_tol ?? null,
          absTol: st.abs_tol ?? 1e-5,
          relTol: st.rel_tol ?? 1e-5,
        });
        activeSpec = searchRes.best_spec;
        wasmWorkspace.recordProof(
          `search_rewrites#${idx}`,
          true,
          searchRes.best_candidate.max_abs_error ?? 0,
          JSON.stringify(searchRes.best_candidate.transform_chain),
        );
        wasmWorkspace.put(
          'search_frontier',
          searchRes.best_candidate.id,
          'rewrite_candidate',
          'promoted',
          JSON.stringify(searchRes.best_candidate),
        );
        stageReports.push({
          stage: stageName,
          ok: true,
          nodes_explored: searchRes.nodes_explored,
          best_candidate: searchRes.best_candidate,
        });
      } else if (stageName === 'fit' || stageName === 'fit_hybrid') {
        const sp = ensureCompiled(true);
        const targets = typeof st.targets === 'string'
          ? JSON.parse(fs.readFileSync(path.resolve(baseDir, st.targets), 'utf8'))
          : st.targets;
        const fitRes = optimizeMultiLayerES(rt, sp, targets, {
          generations: st.generations ?? 25,
          seed: st.seed ?? 1,
          sensitivityGuided: st.sensitivity_guided ?? true,
        });
        activeSpec = fitRes.optimized_spec;
        wasmWorkspace.recordEvent('stage:fit', String(idx), `start=${fitRes.start_fitness.toFixed(4)};final=${fitRes.final_best_fitness.toFixed(4)}`);
        stageReports.push({
          stage: stageName,
          ok: true,
          dim: fitRes.dim,
          active_layers: fitRes.active_layers,
          start_fitness: fitRes.start_fitness,
          final_best_fitness: fitRes.final_best_fitness,
        });
      } else if (stageName === 'solve_gd' || stageName === 'solve-gd') {
        const sp = ensureCompiled(true);
        const targets = typeof st.targets === 'string'
          ? JSON.parse(fs.readFileSync(path.resolve(baseDir, st.targets), 'utf8'))
          : (st.targets ?? null);
        // B5 fix: a workflow stage must not silently optimize toward zero.
        // Explicit targets are required here (the CLI keeps its documented
        // default modes, declared as targets_mode, for interactive use).
        if (targets === null) {
          throw new Error('workflow solve_gd stage requires explicit targets (pass "targets" as a file path or {"cases":[...]} in the stage; to minimize toward zero, pass explicit zero "expected" values)');
        }
        const gdRes = reflectiveGdSolve(rt, sp, targets, {
          layerIndex: st.layer ?? null,
          multiLayer: st.multi_layer ?? (st.layer === 'all'),
          steps: st.steps ?? 20,
          lr: st.lr ?? 0.1,
          precondition: st.precondition ?? false,
        });
        activeSpec = gdRes.optimized_spec;
        wasmWorkspace.recordProof(`solve_gd#${idx}`, gdRes.ok, gdRes.final_loss, `steps=${gdRes.steps_taken};halvings=${gdRes.backtracking_halvings}`);
        stageReports.push({
          stage: stageName,
          ok: gdRes.ok,
          layer_index: gdRes.layer_index,
          active_layers: gdRes.active_layers,
          steps_taken: gdRes.steps_taken,
          start_loss: gdRes.start_loss,
          final_loss: gdRes.final_loss,
          trajectory_verification: gdRes.trajectory_verification,
        });
      } else if (stageName === 'grad_spec') {
        const sp = ensureCompiled(true);
        const order = st.order ?? 1;
        if (order !== 1 && order !== 2) throw new Error(`workflow grad_spec stage: order must be 1 or 2 (got ${JSON.stringify(st.order)})`);
        const wrtSlot = st.wrt_slot ?? 0;
        let gSpec = compileBackwardSpec(sp, { wrtSlot });
        const resolvedWrt = gSpec.meta.wrt_slot;
        if (order === 2) gSpec = compileBackwardSpec(gSpec, { wrtSlot: resolvedWrt });
        if (st.replace_active) activeSpec = gSpec;
        const gradVals = await withGraphScope(rt, gSpec, async ({ g, reg, bundle }) => {
          const b = bundle(gSpec.inputs, portBySlot(gSpec));
          const out = g.run(reg, b);
          const v = Array.from(out.to_array());
          out.free();
          return v;
        });
        const digest = computeSemanticDigest(gradVals, null, { workspaceDir });
        stageReports.push({
          stage: stageName,
          ok: true,
          order,
          wrt_slot: resolvedWrt,
          grad_digest: digest,
        });
      } else if (stageName === 'rollout_scan') {
        const sp = ensureCompiled(true);
        const scanRes = await rolloutScan(rt, sp, {
          steps: st.steps ?? 10,
          statePort: st.state_port ?? null,
          recordTrajectory: false,
        });
        lastRunDigest = computeSemanticDigest(scanRes.final_output.values, scanRes.final_output.shape, { workspaceDir });
        stageReports.push({
          stage: stageName,
          ok: scanRes.ok,
          steps: scanRes.steps,
          output_digest: lastRunDigest,
        });
      } else if (stageName === 'rollout_fixpoint') {
        const sp = ensureCompiled(true);
        const fpRes = await rolloutFixpoint(rt, sp, {
          statePort: st.state_port ?? null,
          maxSteps: st.max_steps ?? 100,
          tol: st.tol ?? 1e-6,
          recordTrajectory: false,
        });
        if (!fpRes.ok) {
          throw new Error(`rollout_fixpoint failed with status ${fpRes.status}: ${fpRes.error ?? 'did not converge'}`);
        }
        lastRunDigest = computeSemanticDigest(fpRes.fixed_point.values, fpRes.fixed_point.shape, { workspaceDir });
        wasmWorkspace.recordProof(`fixpoint#${idx}`, true, fpRes.final_delta_inf, `steps=${fpRes.steps_run};rho=${fpRes.empirical_lipschitz}`);
        stageReports.push({
          stage: stageName,
          ok: true,
          status: fpRes.status,
          steps_run: fpRes.steps_run,
          final_delta_inf: fpRes.final_delta_inf,
          empirical_lipschitz: fpRes.empirical_lipschitz,
          fixed_point_digest: lastRunDigest,
        });
      } else if (stageName === 'run') {
        const sp = ensureCompiled(true);
        const runOut = await withGraphScope(rt, sp, async ({ g, reg, bundle }) => {
          const b = bundle(sp.inputs, portBySlot(sp));
          const out = g.run(reg, b);
          const values = Array.from(out.to_array());
          const shape = Array.from(out.shape());
          out.free();
          return { shape, values };
        });
        lastRunDigest = computeSemanticDigest(runOut.values, runOut.shape, { workspaceDir });
        stageReports.push({
          stage: 'run',
          ok: true,
          output_digest: lastRunDigest,
        });
      } else if (stageName === 'checkpoint' || stageName === 'checkpoint_promote') {
        const sp = ensureCompiled(true);
        const tag = st.tag ?? st.branch_id ?? `ckpt-${idx}`;
        const ckptInfo = await withGraphScope(rt, sp, async ({ g, reg }) => {
          const bytes = rt.exportMultiInputProgramBundle(g, reg, true);
          const sha = sha256Hex(bytes);
          const b64File = path.join(checkpointsDir, `${tag}.b64`);
          const specFile = path.join(checkpointsDir, `${tag}.spec.json`);
          fs.writeFileSync(b64File, Buffer.from(bytes).toString('base64') + '\n');
          fs.writeFileSync(specFile, JSON.stringify(sp, null, 2) + '\n');
          return { tag, bundle_sha256: `sha256:${sha}`, bundle_bytes: bytes.length, b64_file: b64File, spec_file: specFile };
        });
        wasmWorkspace.put('search_frontier', tag, 'checkpoint_bundle', 'committed', JSON.stringify(ckptInfo));
        stageReports.push({
          stage: stageName,
          ok: true,
          ...ckptInfo,
        });
      } else {
        throw new Error(`unknown workflow pipeline stage: ${JSON.stringify(stageName)}`);
      }
    }

    inStages = false;
    syncState();

    return {
      ok: true,
      workspace: workspaceDir,
      stages_completed: stageReports.length,
      stage_reports: stageReports,
      final_output_digest: lastRunDigest,
      active_spec_summary: activeSpec ? {
        layers_count: activeSpec.layers.length,
        steps_count: activeSpec.steps.length,
        num_slots: activeSpec.numSlots,
        output_slot: activeSpec.outputSlot,
      } : null,
      next_actions: [
        'Inspect a stored tensor artifact with: node bin/harness.mjs resolve-artifact <burn://sha256:...>',
        'Check workspace proof & frontier tables with: node bin/harness.mjs workspace-status',
      ],
    };
  } catch (e) {
    // B8 fix (continued): report the failed stage, sync partial
    // state, and return a partial result instead of throwing away
    // everything the successful stages produced.
    if (!inStages) throw e;
    const msg = e instanceof Error ? e.message : String(e);
    stageReports.push({ stage: failedStageName, ok: false, error: msg });
    syncState();
    return {
      ok: false,
      error: msg,
      failed_stage: failedStageName,
      workspace: workspaceDir,
      stages_completed: stageReports.length,
      stage_reports: stageReports,
      final_output_digest: lastRunDigest,
      active_spec_summary: activeSpec ? {
        layers_count: activeSpec.layers.length,
        steps_count: activeSpec.steps.length,
        num_slots: activeSpec.numSlots,
        output_slot: activeSpec.outputSlot,
      } : null,
    };
  } finally {
    if (wasmWorkspace?.free) wasmWorkspace.free();
  }
}

export function exportSubagentBranches(rt, rawSpec, {
  workspaceDir = '.burn-harness',
  branches = [
    { branch_id: 'sub-rewrite', goal: 'structural_rewrite', instructions: 'Run node bin/recurse.mjs search on baseline.spec.json and save to candidate.spec.json' },
    { branch_id: 'sub-quant', goal: 'int8_quantization', instructions: 'Run node bin/recurse.mjs search --quant-tol 1e-2 on baseline.spec.json and save to candidate.spec.json' },
  ],
} = {}) {
  const { branchesDir } = ensureWorkspaceDir(workspaceDir);
  const baseSpec = isRecursiveSpecV2(rawSpec) ? compileRecursiveSpec(rawSpec, { autoRepair: true }) : rawSpec;
  const built = buildGraph(rt, baseSpec);
  try {
    const identity = JSON.parse(built.g.programIdentity());
    const baselineBytes = rt.exportMultiInputProgramBundle(built.g, built.reg, true);
    const baselineSha = `sha256:${sha256Hex(baselineBytes)}`;
    const baselinePath = path.join(branchesDir, 'baseline.spec.json');
    fs.writeFileSync(baselinePath, JSON.stringify(baseSpec, null, 2) + '\n');

    const packets = branches.map((b) => {
      const candidatePath = path.join(branchesDir, `${b.branch_id}.candidate.spec.json`);
      const packet = {
        branch_id: b.branch_id,
        goal: b.goal,
        baseline_state_sha256: baselineSha,
        baseline_program_identity: identity,
        baseline_spec_path: baselinePath,
        candidate_output_path: candidatePath,
        instructions: b.instructions,
      };
      fs.writeFileSync(path.join(branchesDir, `${b.branch_id}.packet.json`), JSON.stringify(packet, null, 2) + '\n');
      return packet;
    });

    return {
      ok: true,
      baseline_state_sha256: baselineSha,
      baseline_spec_path: baselinePath,
      branch_packets: packets,
    };
  } finally {
    if (built.g?.free) built.g.free();
    if (built.plan?.free) built.plan.free();
    if (built.reg?.free) built.reg.free();
  }
}

export function mergeSubagentBranches(rt, baselineSpec, candidateFiles, {
  absTol = 1e-4,
  relTol = 1e-4,
} = {}) {
  const built = buildGraph(rt, baselineSpec);
  try {
    const identity = built.g.programIdentity();
    const baselineBytes = rt.exportMultiInputProgramBundle(built.g, built.reg, true);
    const baselineSha = `sha256:${sha256Hex(baselineBytes)}`;
    const branchSet = new rt.CheckpointBranchSet(built.reg, built.g, identity, baselineSha);
    // B1 fix: the equivalence gate used to test a single input
    // (baselineSpec.inputs), so a genuinely different function that
    // coincides on that input was promoted as "equivalent". The gate
    // now evaluates a deterministic multi-case set: the baseline
    // inputs plus derived cases (zeros, ones, negated, x2, half).
    // A derived case is kept only if the BASELINE evaluates cleanly
    // on it (finite outputs); a case where the baseline itself
    // errors is not a valid discriminator and is dropped, not
    // silently treated as a pass. The surviving case names are
    // reported as equivalence_cases so the gate's strength is visible.
    const baseInputs = baselineSpec.inputs ?? {};
    const deriveInputs = (fn) => {
      const out = {};
      for (const [k, v] of Object.entries(baseInputs)) {
        out[k] = Array.isArray(v) ? v.map((x) => (Number.isFinite(x) ? fn(x) : x)) : v;
      }
      return out;
    };
    const namedCases = [
      { name: 'baseline', inputs: baseInputs },
      { name: 'zeros', inputs: deriveInputs(() => 0) },
      { name: 'ones', inputs: deriveInputs(() => 1) },
      { name: 'negated', inputs: deriveInputs((x) => -x) },
      { name: 'scaled_x2', inputs: deriveInputs((x) => x * 2) },
      { name: 'scaled_half', inputs: deriveInputs((x) => x * 0.5) },
    ];
    const cases = [baseInputs];
    const equivalenceCases = ['baseline'];
    const basePorts = portBySlot(baselineSpec);
    for (const nc of namedCases.slice(1)) {
      try {
        const b = makeBundle(rt, built.plan, nc.inputs, basePorts);
        try {
          const out = built.g.run(built.reg, b);
          const vals = Array.from(out.to_array());
          out.free();
          if (vals.every(Number.isFinite)) {
            cases.push(nc.inputs);
            equivalenceCases.push(nc.name);
          }
        } finally {
          if (b?.free) b.free();
        }
      } catch {
        // Baseline does not evaluate cleanly here: not a discriminator.
      }
    }

    const evaluations = [];
    let winner = null;

    try {
      for (let i = 0; i < candidateFiles.length; i++) {
        const file = candidateFiles[i];
        const branchId = `merge.${i + 1}`;
        let candSpec;
        try {
          candSpec = JSON.parse(fs.readFileSync(file, 'utf8'));
        } catch (e) {
          evaluations.push({ file, branch_id: branchId, accepted: false, reason: `unreadable spec: ${e.message}` });
          continue;
        }
        let ver;
        try {
          ver = verifyAndStageCandidate(rt, built, baselineSpec, candSpec, branchSet, branchId, cases, absTol, relTol);
        } catch (e) {
          // A candidate that errors on a case where the baseline is
          // clean is rejected, not allowed to abort the whole merge.
          evaluations.push({ file, branch_id: branchId, accepted: false, reason: `evaluation_error: ${e.message}` });
          continue;
        }
        const accepted = Boolean(ver.ok && ver.equivalent);
        const rec = {
          file,
          branch_id: branchId,
          accepted,
          equivalent: ver.equivalent,
          max_abs_error: ver.max_abs_error ?? null,
          receipt_digest: ver.receipt_digest ?? null,
          steps_count: candSpec.steps?.length ?? null,
        };
        evaluations.push(rec);
        if (accepted) {
          if (!winner || rec.steps_count < winner.steps_count) {
            winner = { ...rec, spec: candSpec };
          }
        }
      }
    } finally {
      if (branchSet?.free) branchSet.free();
    }

    return {
      ok: winner !== null,
      baseline_state_sha256: baselineSha,
      equivalence_cases: equivalenceCases,
      cases_count: cases.length,
      candidates_evaluated: evaluations.length,
      evaluations,
      promoted_winner: winner ? {
        file: winner.file,
        branch_id: winner.branch_id,
        steps_count: winner.steps_count,
        max_abs_error: winner.max_abs_error,
        receipt_digest: winner.receipt_digest,
      } : null,
      promoted_spec: winner ? winner.spec : null,
    };
  } finally {
    if (built.g?.free) built.g.free();
    if (built.plan?.free) built.plan.free();
    if (built.reg?.free) built.reg.free();
  }
}
