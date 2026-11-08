// burn-math Layer 3: Recursive Graph Search, CheckpointBranchSet Exploration,
// Multi-Layer ES Optimization & Closed-Loop Reflective Backtracking GD Solver (lib/recurse.mjs).

import { createHash } from 'node:crypto';
import { buildGraph, makeBundle } from './engine.mjs';
import { portBySlot, analyzePlanPorts } from './spec.mjs';
import { checkSpec } from './shapecore.mjs';
import { listRewrites, applyRewrite } from './rewrites.mjs';
import { profileSpec, installMemoryOracle } from './opt.mjs';
import { analyticalPlan, dequantizedWeights, linearLayerInfos } from './quant_budget.mjs';
import { inferShapes, backwardWithSaved, captureSavedForSpec } from './autodiff.mjs';
import { verifyGdSteps, hessianDiagSecondOrder, hessianTrust } from './fit.mjs';

const clone = (o) => JSON.parse(JSON.stringify(o));

const sha256Json = (obj) =>
  `sha256:${createHash('sha256').update(JSON.stringify(obj)).digest('hex')}`;

const LAYER_TYPE = {
  linear: 1, layerNorm: 2, batchNorm: 2, groupNorm: 2, instanceNorm: 2, rmsNorm: 2,
  conv: 3, conv1d: 3, conv2d: 3, convTranspose2d: 3, embedding: 5,
};

export function eliminateDeadSteps(spec) {
  const s = clone(spec);
  const liveSlots = new Set([s.outputSlot]);
  const liveSteps = new Set();
  for (let si = s.steps.length - 1; si >= 0; si--) {
    const st = s.steps[si];
    const outSlot = st.slots[st.slots.length - 1];
    if (liveSlots.has(outSlot)) {
      liveSteps.add(si);
      const inCount = st.kind === 'binary' ? 2 : 1;
      for (let i = 0; i < inCount; i++) liveSlots.add(st.slots[i]);
    }
  }
  const portSlots = new Set((s.ports ?? []).map((p) => p.slot));
  const consumedPorts = new Set();
  for (const si of liveSteps) {
    const st = s.steps[si];
    const inCount = st.kind === 'binary' ? 2 : 1;
    for (let i = 0; i < inCount; i++) {
      if (portSlots.has(st.slots[i])) consumedPorts.add(st.slots[i]);
    }
  }
  for (let si = 0; si < s.steps.length; si++) {
    if (liveSteps.has(si)) continue;
    const st = s.steps[si];
    const inCount = st.kind === 'binary' ? 2 : 1;
    for (let i = 0; i < inCount; i++) {
      if (portSlots.has(st.slots[i]) && !consumedPorts.has(st.slots[i])) {
        liveSteps.add(si);
        consumedPorts.add(st.slots[i]);
      }
    }
  }
  if (liveSteps.size === s.steps.length) return null;
  s.steps = s.steps.filter((_, idx) => liveSteps.has(idx));
  return s;
}

export function verifyAndStageCandidate(rt, baseBuilt, baseSpec, candSpec, branchSet, branchId, cases, absTol = 1e-5, relTol = 1e-5) {
  const chk = checkSpec(candSpec);
  if (!chk.ok) {
    return { ok: false, stage: 'staticCheck', equivalent: false, reason: 'failed static shape/dataflow check' };
  }
  const candBuilt = buildGraph(rt, candSpec);
  const basePorts = portBySlot(baseSpec);
  const candPorts = portBySlot(candSpec);

  try {
    const vc = new rt.MultiInputVerificationCases(baseBuilt.g, candBuilt.g);
    for (const c of cases) {
      const bBase = makeBundle(rt, baseBuilt.plan, c, basePorts);
      const bCand = makeBundle(rt, candBuilt.plan, c, candPorts);
      vc.addCase(bBase, bCand);
      if (bBase?.free) bBase.free();
      if (bCand?.free) bCand.free();
    }
    const ver = JSON.parse(vc.verify(baseBuilt.reg, baseBuilt.g, candBuilt.reg, candBuilt.g, absTol, relTol));
    if (vc.free) vc.free();

    let candidateBundleSha = null;
    try {
      const candBytes = rt.exportMultiInputProgramBundle(candBuilt.g, candBuilt.reg, true);
      candidateBundleSha = `sha256:${createHash('sha256').update(candBytes).digest('hex')}`;
    } catch {
      candidateBundleSha = sha256Json({ layers: candSpec.layers, steps: candSpec.steps, weights: candSpec.weights });
    }

    const baseLayerById = new Map((baseSpec.layers ?? []).map((l, idx) => [l.args[0], { l, idx }]));
    let changedLayersCount = 0;
    let unchangedLayersCount = 0;
    const candIds = new Set();
    for (let ci = 0; ci < (candSpec.layers ?? []).length; ci++) {
      const cDecl = candSpec.layers[ci];
      const regId = cDecl.args?.[0];
      candIds.add(regId);
      const cW = candSpec.weights?.[String(ci)] ?? null;
      const bMatch = baseLayerById.get(regId);
      if (!bMatch || bMatch.l.constructor !== cDecl.constructor || JSON.stringify(bMatch.l.args) !== JSON.stringify(cDecl.args)) {
        changedLayersCount++;
      } else {
        const bW = baseSpec.weights?.[String(bMatch.idx)] ?? null;
        if (JSON.stringify(bW) !== JSON.stringify(cW)) changedLayersCount++;
        else unchangedLayersCount++;
      }
    }
    for (const [bRegId] of baseLayerById.entries()) {
      if (!candIds.has(bRegId)) changedLayersCount++;
    }

    let branchReceipt = null;
    let stateDiffSummary = {
      changed_layers_count: changedLayersCount,
      unchanged_layers_count: unchangedLayersCount,
      steps_delta: (candSpec.steps ?? []).length - (baseSpec.steps ?? []).length,
      changed_byte_count: 0,
      byte_length_delta: 0,
    };

    const isSameTopology = JSON.stringify(baseSpec.steps) === JSON.stringify(candSpec.steps)
      && JSON.stringify(baseSpec.layers) === JSON.stringify(candSpec.layers)
      && baseSpec.outputSlot === candSpec.outputSlot;

    if (branchSet && branchSet.branchCount() < 8 && isSameTopology) {
      try {
        branchSet.fork(branchId);
        for (let ci = 0; ci < (candSpec.layers ?? []).length; ci++) {
          const cDecl = candSpec.layers[ci];
          const cW = candSpec.weights?.[String(ci)];
          const bMatch = baseLayerById.get(cDecl.args[0]);
          if (bMatch && Array.isArray(cW) && bMatch.l.constructor === cDecl.constructor) {
            const bW = baseSpec.weights?.[String(bMatch.idx)];
            if (!bW || bW.length === cW.length) {
              const type = LAYER_TYPE[cDecl.constructor];
              if (type !== undefined) {
                branchSet.setWeightsFlat(branchId, type, cDecl.args[0], Float32Array.from(cW));
              }
            }
          }
        }
        branchSet.stageBranch(branchId);
        const rawDiff = JSON.parse(branchSet.stateDiff(branchId));
        const sd = rawDiff.state_diff ?? rawDiff;
        stateDiffSummary = {
          changed_layers_count: changedLayersCount,
          unchanged_layers_count: unchangedLayersCount,
          steps_delta: 0,
          changed_byte_count: sd.changed_byte_count ?? 0,
          byte_length_delta: sd.byte_length_delta ?? 0,
          ranges_count: (sd.ranges ?? []).length,
        };
        const regC = new rt.LayerRegistry();
        const gC = rt.importMultiInputProgramBundle(regC, Buffer.from(branchSet.candidateBundle(branchId)));
        const planC = rt.MultiInputGraphPlan.fromBytes(gC.inputPlanV1());
        const bvc = new rt.MultiInputVerificationCases(baseBuilt.g, gC);
        for (const c of cases) {
          const bBase = makeBundle(rt, baseBuilt.plan, c, basePorts);
          const bCand = makeBundle(rt, planC, c, basePorts);
          bvc.addCase(bBase, bCand);
          if (bBase?.free) bBase.free();
          if (bCand?.free) bCand.free();
        }
        branchReceipt = JSON.parse(branchSet.verifyBranch(branchId, bvc, absTol, relTol));
        if (bvc.free) bvc.free();
        if (planC.free) planC.free();
        if (gC.free) gC.free();
        if (regC.free) regC.free();
      } catch { /* structural edits verified via MultiInputVerificationCases */ }
    } else {
      try {
        const baseBytes = rt.exportMultiInputProgramBundle(baseBuilt.g, baseBuilt.reg, true);
        const candBytes = rt.exportMultiInputProgramBundle(candBuilt.g, candBuilt.reg, true);
        const minLen = Math.min(baseBytes.length, candBytes.length);
        let diffBytes = Math.abs(candBytes.length - baseBytes.length);
        for (let i = 0; i < minLen; i++) {
          if (baseBytes[i] !== candBytes[i]) diffBytes++;
        }
        stateDiffSummary.changed_byte_count = diffBytes;
        stateDiffSummary.byte_length_delta = candBytes.length - baseBytes.length;
      } catch { /* ignore */ }
    }

    return {
      ok: true,
      equivalent: Boolean(ver.equivalent),
      max_abs_error: ver.max_abs_error,
      max_rel_error: ver.max_rel_error,
      candidate_state_sha256: candidateBundleSha,
      receipt_digest: ver.receipt_digest ?? branchReceipt?.receipt_digest ?? null,
      state_diff: stateDiffSummary,
    };
  } finally {
    if (candBuilt.g?.free) candBuilt.g.free();
    if (candBuilt.plan?.free) candBuilt.plan.free();
    if (candBuilt.reg?.free) candBuilt.reg.free();
  }
}

export async function searchGraphTree(rt, baseSpec, {
  maxDepth = 3,
  maxBranchesPerDepth = 8,
  cases = null,
  absTol = 1e-5,
  relTol = 1e-5,
  quantTol = null,
  profileIterations = 5,
} = {}) {
  installMemoryOracle();
  const evalCases = Array.isArray(cases) && cases.length > 0 ? cases : [baseSpec.inputs];
  const baseBuilt = buildGraph(rt, baseSpec);
  const identity = baseBuilt.g.programIdentity();
  let baselineSha = sha256Json(baseSpec);
  let branchSet = null;
  try {
    const baselineBytes = rt.exportMultiInputProgramBundle(baseBuilt.g, baseBuilt.reg, true);
    baselineSha = `sha256:${createHash('sha256').update(baselineBytes).digest('hex')}`;
    branchSet = new rt.CheckpointBranchSet(baseBuilt.reg, baseBuilt.g, identity, baselineSha);
  } catch {
    branchSet = null;
  }

  const countActiveParams = (sp) => {
    const usedLayers = new Set(sp.steps.map((st) => st.layer));
    let total = 0;
    for (const li of usedLayers) {
      const w = sp.weights?.[String(li)];
      if (Array.isArray(w)) total += w.length;
    }
    return total;
  };

  try {
    const baseProfile = profileSpec(rt, baseSpec, { iterations: Math.max(2, profileIterations) });
    const rootNode = {
      id: 'root',
      depth: 0,
      parent_id: null,
      transform_chain: [],
      steps_count: baseSpec.steps.length,
      active_params: countActiveParams(baseSpec),
      run_ms_median: baseProfile.steady.run_ms_median,
      equivalent: true,
      max_abs_error: 0,
      candidate_state_sha256: baselineSha,
      receipt_digest: baselineSha,
      spec: baseSpec,
    };

    const visitedHashes = new Set([sha256Json({ steps: baseSpec.steps, weights: baseSpec.weights })]);
    const searchTree = [rootNode];
    let frontier = [rootNode];
    let bestNode = rootNode;
    let branchSeq = 0;

    const availableRewrites = listRewrites().map((r) => r.name).filter((n) => n !== 'add-commute');

    for (let depth = 1; depth <= maxDepth; depth++) {
      const nextFrontier = [];
      for (const parent of frontier) {
        if (nextFrontier.length >= maxBranchesPerDepth) break;

        for (const rwName of availableRewrites) {
          if (nextFrontier.length >= maxBranchesPerDepth) break;
          let candSpec = null;
          try {
            if (rwName === 'dead-step-elimination') {
              candSpec = eliminateDeadSteps(parent.spec);
            } else {
              candSpec = applyRewrite(rwName, parent.spec);
            }
          } catch {
            candSpec = null;
          }
          if (!candSpec) continue;

          const cleaned = eliminateDeadSteps(candSpec);
          if (cleaned) candSpec = cleaned;

          const sig = sha256Json({ steps: candSpec.steps, weights: candSpec.weights });
          if (visitedHashes.has(sig)) continue;
          visitedHashes.add(sig);

          const branchId = `b.d${depth}.n${++branchSeq}`;
          const ver = verifyAndStageCandidate(rt, baseBuilt, baseSpec, candSpec, branchSet, branchId, evalCases, absTol, relTol);

          let prof = null;
          if (ver.ok && ver.equivalent) {
            prof = profileSpec(rt, candSpec, { iterations: Math.max(2, profileIterations) });
          }

          const node = {
            id: branchId,
            depth,
            parent_id: parent.id,
            transform_chain: [...parent.transform_chain, rwName],
            steps_count: candSpec.steps.length,
            active_params: countActiveParams(candSpec),
            run_ms_median: prof ? prof.steady.run_ms_median : null,
            equivalent: ver.equivalent,
            max_abs_error: ver.max_abs_error ?? null,
            candidate_state_sha256: ver.candidate_state_sha256 ?? null,
            receipt_digest: ver.receipt_digest ?? null,
            ...(ver.state_diff ? { state_diff: ver.state_diff } : {}),
            spec: candSpec,
          };
          searchTree.push(node);

          if (ver.ok && ver.equivalent) {
            nextFrontier.push(node);
            const isBetter = (node.steps_count < bestNode.steps_count)
              || (node.steps_count === bestNode.steps_count && node.active_params < bestNode.active_params)
              || (node.steps_count === bestNode.steps_count && node.active_params === bestNode.active_params && node.run_ms_median < bestNode.run_ms_median);
            if (isBetter) bestNode = node;
          }
        }
      }
      if (nextFrontier.length === 0) break;
      frontier = nextFrontier;
    }

    if (quantTol !== null && quantTol > 0) {
      try {
        const infos = linearLayerInfos(bestNode.spec);
        if (infos.length > 0) {
          const qPlan = analyticalPlan(bestNode.spec, evalCases, 'q8-per-block', quantTol, rt, infos);
          if (qPlan.quantize.length > 0) {
            const qSpec = clone(bestNode.spec);
            for (const li of qPlan.quantize) {
              const info = infos.find((x) => x.index === li);
              qSpec.weights[String(li)] = dequantizedWeights(rt, info, 'q8-per-block');
            }
            const branchId = `b.quant.${++branchSeq}`;
            const ver = verifyAndStageCandidate(rt, baseBuilt, baseSpec, qSpec, branchSet, branchId, evalCases, quantTol, quantTol);
            if (ver.ok && ver.equivalent) {
              const prof = profileSpec(rt, qSpec, { iterations: Math.max(2, profileIterations) });
              const qNode = {
                id: branchId,
                depth: bestNode.depth + 1,
                parent_id: bestNode.id,
                transform_chain: [...bestNode.transform_chain, `quant_q8(layers=[${qPlan.quantize.join(',')}],saved=${qPlan.saved}B)`],
                steps_count: qSpec.steps.length,
                active_params: countActiveParams(qSpec),
                quantized_saved_bytes: qPlan.saved,
                run_ms_median: prof.steady.run_ms_median,
                equivalent: true,
                max_abs_error: ver.max_abs_error,
                candidate_state_sha256: ver.candidate_state_sha256 ?? null,
                receipt_digest: ver.receipt_digest,
                ...(ver.state_diff ? { state_diff: ver.state_diff } : {}),
                spec: qSpec,
              };
              searchTree.push(qNode);
              bestNode = qNode;
            }
          }
        }
      } catch { /* keep structural bestNode */ }
    }

    return {
      ok: true,
      baseline: {
        state_sha256: baselineSha,
        steps_count: rootNode.steps_count,
        active_params: rootNode.active_params,
        run_ms_median: rootNode.run_ms_median,
      },
      best_candidate: {
        id: bestNode.id,
        transform_chain: bestNode.transform_chain,
        steps_count: bestNode.steps_count,
        active_params: bestNode.active_params,
        quantized_saved_bytes: bestNode.quantized_saved_bytes ?? 0,
        run_ms_median: bestNode.run_ms_median,
        max_abs_error: bestNode.max_abs_error,
        candidate_state_sha256: bestNode.candidate_state_sha256 ?? baselineSha,
        receipt_digest: bestNode.receipt_digest,
        steps_reduced: rootNode.steps_count - bestNode.steps_count,
        params_reduced: rootNode.active_params - bestNode.active_params,
      },
      nodes_explored: searchTree.length,
      search_tree: searchTree.map(({ spec: _s, ...meta }) => meta),
      best_spec: bestNode.spec,
    };
  } finally {
    if (branchSet?.free) branchSet.free();
    if (baseBuilt.g?.free) baseBuilt.g.free();
    if (baseBuilt.plan?.free) baseBuilt.plan.free();
    if (baseBuilt.reg?.free) baseBuilt.reg.free();
  }
}

const WIRED_WEIGHTED = new Set([
  'linear', 'layerNorm', 'batchNorm', 'groupNorm', 'instanceNorm', 'rmsNorm',
  'conv1d', 'conv2d', 'convTranspose2d', 'embedding',
]);

export function discoverTrainableLayout(spec) {
  const usedLayers = new Set((spec.steps ?? []).map((s) => s.layer));
  const bindings = [];
  let offset = 0;
  for (let li = 0; li < (spec.layers ?? []).length; li++) {
    const decl = spec.layers[li];
    const w = spec.weights?.[String(li)];
    if (!decl || !WIRED_WEIGHTED.has(decl.constructor) || !Array.isArray(w) || w.length === 0) continue;
    if (!usedLayers.has(li)) continue;
    bindings.push({
      layer_index: li,
      registry_id: decl.args[0],
      constructor: decl.constructor,
      offset,
      length: w.length,
      initial_weights: [...w],
    });
    offset += w.length;
  }
  return { total_dim: offset, bindings };
}

export function optimizeMultiLayerES(rt, spec, targets, {
  generations = 40,
  pop = null,
  sigma = null,
  lr = null,
  seed = 1,
  strategy = 0,
  sensitivityGuided = false,
  freezeThreshold = 0,
} = {}) {
  const layout = discoverTrainableLayout(spec);
  if (layout.total_dim === 0) {
    throw new Error('optimizeMultiLayerES: spec has no active weighted layers');
  }
  const ports = portBySlot(spec);

  const evalSpecLoss = (s) => {
    const { reg, plan, g } = buildGraph(rt, s);
    let mse = 0;
    try {
      for (const c of targets.cases) {
        const b = makeBundle(rt, plan, c.inputs, ports);
        const out = g.run(reg, b);
        const vals = Array.from(out.to_array());
        out.free();
        if (b.free) b.free();
        if (vals.length !== c.expected.length) {
          throw new Error(`case output length ${vals.length} != expected ${c.expected.length}`);
        }
        for (let i = 0; i < vals.length; i++) mse += (vals[i] - c.expected[i]) ** 2;
      }
    } finally {
      if (g.free) g.free();
      if (plan.free) plan.free();
      if (reg.free) reg.free();
    }
    return mse;
  };

  let activeBindings = layout.bindings;
  const layerSensitivity = [];
  if (sensitivityGuided && layout.bindings.length > 1) {
    const eps = 1e-3;
    const baseLoss = evalSpecLoss(spec);
    for (const b of layout.bindings) {
      let maxGradAbs = 0;
      const wBase = [...b.initial_weights];
      for (let i = 0; i < wBase.length; i++) {
        const sPlus = clone(spec);
        sPlus.weights[String(b.layer_index)][i] = wBase[i] + eps;
        const lPlus = evalSpecLoss(sPlus);
        const gAbs = Math.abs((lPlus - baseLoss) / eps);
        if (gAbs > maxGradAbs) maxGradAbs = gAbs;
      }
      layerSensitivity.push({ layer_index: b.layer_index, constructor: b.constructor, max_grad_abs: maxGradAbs });
    }
    const filtered = layout.bindings.filter((b, idx) => layerSensitivity[idx].max_grad_abs >= freezeThreshold);
    if (filtered.length > 0) activeBindings = filtered;
  }

  let activeDim = 0;
  const activeSlices = activeBindings.map((b) => {
    const sl = { ...b, active_offset: activeDim };
    activeDim += b.length;
    return sl;
  });

  const applyFlatCandidate = (flatDeltaOrAbs, isAbsolute = false) => {
    const s2 = clone(spec);
    for (const sl of activeSlices) {
      const slice = Array.from(flatDeltaOrAbs.slice(sl.active_offset, sl.active_offset + sl.length));
      s2.weights[String(sl.layer_index)] = isAbsolute
        ? slice
        : slice.map((v, idx) => sl.initial_weights[idx] + v);
    }
    return s2;
  };

  const fitnessOf = (flatVec) => -evalSpecLoss(applyFlatCandidate(flatVec, false));
  const startFitness = -evalSpecLoss(spec);

  const es = rt.EsOptimizer.strict(activeDim, strategy, seed, pop, sigma, lr);
  const trajectory = [];
  try {
    for (let gen = 0; gen < generations; gen++) {
      const cand = es.ask();
      const n = es.batchSize();
      const fits = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        fits[i] = fitnessOf(cand.slice(i * activeDim, (i + 1) * activeDim));
      }
      const rep = JSON.parse(es.tell(fits));
      trajectory.push({ gen, best: rep.best, mean: rep.mean });
    }
    const bestVec = Array.from(es.best());
    const esBestFitness = fitnessOf(bestVec);
    const useEsBest = esBestFitness >= startFitness;
    const optimizedSpec = useEsBest ? applyFlatCandidate(bestVec, false) : clone(spec);
    const finalFitness = useEsBest ? esBestFitness : startFitness;
    const flatBestWeights = activeSlices.flatMap((sl) => optimizedSpec.weights[String(sl.layer_index)]);

    return {
      ok: true,
      dim: activeDim,
      total_graph_dim: layout.total_dim,
      active_layers: activeSlices.map((s) => ({ layer_index: s.layer_index, constructor: s.constructor, length: s.length })),
      layer_sensitivity: layerSensitivity.length ? layerSensitivity : undefined,
      generations,
      seed,
      strategy,
      start_fitness: startFitness,
      final_best_fitness: finalFitness,
      best_weights: flatBestWeights,
      best_weights_by_layer: optimizedSpec.weights,
      trajectory,
      optimized_spec: optimizedSpec,
    };
  } finally {
    es.free();
  }
}

function ensurePortsConsumedForSubRun(spec) {
  const planCheck = analyzePlanPorts(spec);
  if (planCheck.valid || planCheck.code !== 'plan_unused_ports') return spec;
  const s = clone(spec);
  let nextRegId = Math.max(0, ...s.layers.map((l) => l.args[0] ?? 0)) + 1;
  let nextSlot = s.numSlots;
  for (const unusedSlot of planCheck.unused_ports) {
    const drainIdx = s.layers.length;
    s.layers.push({ constructor: 'relu', args: [nextRegId++] });
    s.steps.push({ kind: 'unary', layer: drainIdx, slots: [unusedSlot, nextSlot++] });
  }
  s.numSlots = nextSlot;
  return s;
}

// Closed-Loop Reflective Backtracking Line-Search Gradient Descent / Diagonal-Newton Solver
// supporting both single-layer (`--layer <idx>`) and joint multi-layer (`--layer all` / `--multi-layer`) BPTT.
export function reflectiveGdSolve(rt, spec, targets, {
  layerIndex = null,
  multiLayer = false,
  steps = 20,
  lr = 0.1,
  maxHalvings = 8,
  precondition = false,
  damping = 1e-2,
  tol = 1e-6,
} = {}) {
  const layout = discoverTrainableLayout(spec);
  if (layout.bindings.length === 0) {
    throw new Error('reflectiveGdSolve: spec has no active weighted layers');
  }

  const useAllLayers = multiLayer || layerIndex === 'all' || (layerIndex === null && layout.bindings.length > 1);
  let activeBindings;
  if (useAllLayers) {
    activeBindings = layout.bindings;
  } else {
    const b = layerIndex !== null
      ? layout.bindings.find((x) => x.layer_index === Number(layerIndex))
      : layout.bindings[0];
    if (!b) {
      throw new Error(`reflectiveGdSolve: layer ${layerIndex} is not a weighted layer`);
    }
    activeBindings = [b];
  }

  let totalDim = 0;
  const activeSlices = activeBindings.map((b) => {
    const sl = { ...b, slice_offset: totalDim };
    totalDim += b.length;
    return sl;
  });

  const ports = portBySlot(spec);
  const runSlot = (s, slot) => {
    const sub = ensurePortsConsumedForSubRun({ ...s, outputSlot: slot });
    const { reg, plan, g } = buildGraph(rt, sub);
    const bundle = makeBundle(rt, plan, sub.inputs, ports);
    try {
      const out = g.run(reg, bundle);
      const v = Array.from(out.to_array());
      out.free();
      return v;
    } finally {
      if (bundle?.free) bundle.free();
      if (g.free) g.free();
      if (plan.free) plan.free();
      if (reg.free) reg.free();
    }
  };

  // B5 fix: provided-but-malformed targets used to be silently ignored
  // (bare array, {cases: []}, non-object) and replaced by a hidden zero
  // default. Now they fail closed with a named error. Missing targets
  // remain a documented mode (scalar objective for 1-output specs,
  // zero-MSE otherwise) but are now DECLARED in the result as
  // targets_mode instead of silent.
  if (targets !== null && targets !== undefined
      && (!Array.isArray(targets.cases) || targets.cases.length === 0)) {
    throw new Error('reflectiveGdSolve: targets must hold a non-empty "cases" array (got '
      + (Array.isArray(targets) ? 'bare array' : typeof targets)
      + '); pass explicit targets, e.g. {"cases":[{"inputs":[...],"expected":[...]}]}');
  }
  const hasExplicitTargets = Boolean(targets?.cases);
  let zeroY0Length = null;
  const resolvedTargets = hasExplicitTargets
    ? targets
    : (() => {
        const y0 = runSlot(spec, spec.outputSlot);
        zeroY0Length = y0.length;
        return { cases: [{ inputs: clone(spec.inputs), expected: new Array(y0.length).fill(0) }] };
      })();
  const targetsMode = hasExplicitTargets
    ? 'explicit'
    : (zeroY0Length === 1 ? 'scalar_objective' : 'zero_default');

  const applyFlatWeightsToSpec = (flatW) => {
    const s = clone(spec);
    for (const sl of activeSlices) {
      s.weights[String(sl.layer_index)] = flatW.slice(sl.slice_offset, sl.slice_offset + sl.length);
    }
    return s;
  };

  // Evaluate loss and analytic BPTT gradient dL/dW across all activeSlices.
  // Aligns with scalarize() in lib/fit.mjs: when no targets are supplied and the
  // output has 1 element, the scalar output IS the loss (dout = [1]); otherwise MSE.
  const evalLossAndGrad = (flatW) => {
    const s = applyFlatWeightsToSpec(flatW);
    const shapes = inferShapes(s);
    let totalLoss = 0;
    const totalGrad = new Array(totalDim).fill(0);

    for (const c of resolvedTargets.cases) {
      s.inputs = clone(c.inputs);
      const y = runSlot(s, s.outputSlot);
      const dout = new Array(y.length);
      if (!hasExplicitTargets && y.length === 1) {
        totalLoss += y[0];
        dout[0] = 1;
      } else {
        for (let i = 0; i < y.length; i++) {
          const diff = y[i] - c.expected[i];
          totalLoss += diff * diff;
          dout[i] = 2 * diff;
        }
      }
      const saved = captureSavedForSpec(s, shapes, runSlot);
      const weightOf = (idx, args, ctor = 'linear') => {
        const vals = s.weights[String(idx)];
        if (ctor === 'rmsNorm') return { w: vals.slice(0, args[1]), b: null };
        if (ctor === 'layerNorm') return { w: vals.slice(0, args[1]), b: vals.slice(args[1], 2 * args[1]) };
        if (ctor === 'swiGlu') {
          // Tahap 7b: flat engine order [wI, bI?, wO, bO?].
          const [, dIn, dOut, bias] = args;
          const inner = dIn * dOut;
          const wI = vals.slice(0, inner);
          const bI = bias ? vals.slice(inner, inner + dOut) : null;
          const o0 = inner + (bias ? dOut : 0);
          const wO = vals.slice(o0, o0 + inner);
          const bO = bias ? vals.slice(o0 + inner) : null;
          return { wI, bI, wO, bO, hasSpecWeights: !!vals };
        }
        const [, inD, outD, bias] = args;
        const nw = inD * outD;
        return { w: vals.slice(0, nw), b: bias ? vals.slice(nw, nw + outD) : null };
      };
      const vjp = backwardWithSaved(s, shapes, saved, { [s.outputSlot]: dout }, weightOf);
      for (const sl of activeSlices) {
        const pg = vjp.paramGrads[String(sl.layer_index)];
        if (pg) {
          const flatPg = [...(pg.w ?? []), ...(pg.b ?? [])];
          for (let i = 0; i < sl.length; i++) {
            totalGrad[sl.slice_offset + i] += flatPg[i] ?? 0;
          }
        }
      }
    }
    return { loss: totalLoss, grad: totalGrad };
  };

  let curW = activeSlices.flatMap((sl) => sl.initial_weights);
  let precondScale = new Array(totalDim).fill(1);
  if (precondition) {
    const h = 1e-2;
    const hFine = h / 2;
    const baseLoss = evalLossAndGrad(curW).loss;
    precondScale = curW.map((_, j) => {
      const wP = [...curW]; wP[j] += h;
      const wM = [...curW]; wM[j] -= h;
      const lP = evalLossAndGrad(wP).loss;
      const lM = evalLossAndGrad(wM).loss;
      const hCoarse = hessianDiagSecondOrder(lP, baseLoss, lM, h);

      const wPf = [...curW]; wPf[j] += hFine;
      const wMf = [...curW]; wMf[j] -= hFine;
      const lPf = evalLossAndGrad(wPf).loss;
      const lMf = evalLossAndGrad(wMf).loss;
      const hFineEst = hessianDiagSecondOrder(lPf, baseLoss, lMf, hFine);

      const trust = hessianTrust(hCoarse, hFineEst, 0.05, 0.1);
      if (trust.status === 'trusted' && Math.abs(trust.value) > damping) {
        return 1 / Math.max(Math.abs(trust.value), damping);
      }
      return 1;
    });
  }

  let { loss: curLoss, grad: curGrad } = evalLossAndGrad(curW);
  const trajSteps = [{ weights: [...curW], loss: curLoss, grad: [...curGrad], lr }];
  let totalHalvings = 0;

  for (let s = 0; s < steps; s++) {
    const gradNormSq = curGrad.reduce((acc, g) => acc + g * g, 0);
    if (gradNormSq <= tol * tol) break;

    let stepLr = lr;
    let acceptedW = null;
    let acceptedLoss = null;
    let halvingsUsed = 0;

    for (let h = 0; h <= maxHalvings; h++) {
      const candW = curW.map((w, i) => w - stepLr * precondScale[i] * curGrad[i]);
      const { loss: candLoss } = evalLossAndGrad(candW);
      if (candLoss <= curLoss + 1e-12) {
        acceptedW = candW;
        acceptedLoss = candLoss;
        halvingsUsed = h;
        break;
      }
      stepLr *= 0.5;
    }

    if (acceptedW === null) break;
    totalHalvings += halvingsUsed;
    trajSteps[trajSteps.length - 1].lr = stepLr;

    curW = acceptedW;
    const nextEval = evalLossAndGrad(curW);
    curLoss = acceptedLoss;
    curGrad = nextEval.grad;
    trajSteps.push({ weights: [...curW], loss: curLoss, grad: [...curGrad], lr });
  }

  // B7 fix: recomputedLosses used to alias the trajectory's own loss
  // values (the same numbers compared to themselves), so
  // mismatch_count was identically 0. Now each step's loss is
  // recomputed from its recorded weights alone through fresh engine
  // evaluations, so a misrecorded or misaligned trajectory loss is
  // actually caught. (This cross-checks weights<->loss consistency;
  // it is not a second loss implementation — fit_gd_verify remains
  // the independent trajectory checker.)
  const recomputedLosses = trajSteps.map((st) => evalLossAndGrad(st.weights).loss);
  const gdVerification = verifyGdSteps(trajSteps, recomputedLosses, 1e-5);
  const optimizedSpec = applyFlatWeightsToSpec(curW);
  const isSingleLayer = activeSlices.length === 1;

  return {
    ok: gdVerification.ok && gdVerification.ascent_count === 0,
    targets_mode: targetsMode,
    layer_index: isSingleLayer ? activeSlices[0].layer_index : 'all',
    active_layers: activeSlices.map((sl) => ({
      layer_index: sl.layer_index,
      constructor: sl.constructor,
      offset: sl.slice_offset,
      length: sl.length,
    })),
    steps_taken: trajSteps.length - 1,
    backtracking_halvings: totalHalvings,
    preconditioned: precondition,
    start_loss: recomputedLosses[0],
    final_loss: recomputedLosses[recomputedLosses.length - 1],
    initial_objective: recomputedLosses[0],
    final_objective: recomputedLosses[recomputedLosses.length - 1],
    best_weights: curW,
    best_weights_by_layer: Object.fromEntries(
      activeSlices.map((sl) => [String(sl.layer_index), curW.slice(sl.slice_offset, sl.slice_offset + sl.length)]),
    ),
    gd_verification: {
      ok: gdVerification.ok,
      mismatch_count: gdVerification.mismatch_count,
      ascent_count: gdVerification.ascent_count,
      prediction_off_count: gdVerification.prediction_off_count,
      grad_norm_first: gdVerification.grad_norm_first,
      grad_norm_last: gdVerification.grad_norm_last,
    },
    trajectory_verification: {
      ok: gdVerification.ok,
      mismatch_count: gdVerification.mismatch_count,
      ascent_count: gdVerification.ascent_count,
      prediction_off_count: gdVerification.prediction_off_count,
      grad_norm_first: gdVerification.grad_norm_first,
      grad_norm_last: gdVerification.grad_norm_last,
    },
    trajectory: {
      ...(isSingleLayer
        ? { layer: activeSlices[0].layer_index }
        : { layers: activeSlices.map((sl) => sl.layer_index) }),
      ...(hasExplicitTargets ? { targets: resolvedTargets } : {}),
      steps: trajSteps,
    },
    optimized_spec: optimizedSpec,
  };
}
