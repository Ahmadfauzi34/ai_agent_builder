// burn-math Layer 2: Temporal & Execution-Time Recursion (lib/rollout.mjs)
//
// Provides:
// 1. rolloutScan: multi-step recurrent state handoff (s_{t+1} = Graph(s_t, x_t))
//    inside a single leak-free WASM graph scope with per-tick SHA-256 digests.
// 2. rolloutFixpoint: iterative fixed-point solver until ||s_{t+1} - s_t||_oo <= tol,
//    with empirical Lipschitz contraction tracking and fail-closed divergence/drift guards.
// 3. bisectDivergence: step-level divergence localization + automatic minimal
//    subgraph extraction at the exact first divergent step.
// 4. unrollRolloutSpec: temporal horizon unroller converting a 1-step recurrent
//    spec into an N-step weight-tied Spec v1 for BPTT autodiff, HVP, or verification.

import { createHash } from 'node:crypto';
import { withGraphScope } from './wasm_direct.mjs';
import { portBySlot, isFiniteF32 } from './spec.mjs';
import { compileRecursiveSpec, autoRepairSpec } from './composer.mjs';
import { compactSlotsByLiveness } from './explain.mjs';
import { inferShapes } from './autodiff.mjs';

const sha256F32 = (vals) => {
  const f32 = Float32Array.from(vals);
  const bytes = Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength);
  return `sha256:${createHash('sha256').update(bytes).digest('hex')}`;
};

function resolveStatePortSlot(spec, explicitStatePort) {
  if (explicitStatePort !== undefined && explicitStatePort !== null) {
    if (typeof explicitStatePort === 'number') return explicitStatePort;
    if (/^(?:0|[1-9]\d*)$/.test(String(explicitStatePort))) return Number(explicitStatePort);
    const symSlot = spec.meta?.symbol_table?.[String(explicitStatePort)];
    if (symSlot !== undefined) return symSlot;
    // B4 fix: an explicit symbolic name that resolves nowhere used to
    // fall through silently to the role 'state' port. Now it fails
    // closed with a named error. (Numeric and known-symbolic paths
    // above are unchanged; no explicit port keeps the old fallback.)
    const err = new Error(`rollout: unknown state-port name "${explicitStatePort}" (no such symbol in spec.meta.symbol_table); pass a slot number or a known symbolic name`);
    err.code = 'rollout_unknown_state_port';
    throw err;
  }
  const statePort = (spec.ports ?? []).find((p) => p.role === 'state');
  if (statePort) return statePort.slot;
  return spec.ports?.[0]?.slot ?? 0;
}

export function unrollRolloutSpec(rawSpec, {
  steps = 3,
  statePort = null,
  compactSlots = false,
} = {}) {
  const spec = rawSpec.schema || rawSpec.subgraphs || (Array.isArray(rawSpec.steps) && rawSpec.steps.some((s) => s.op))
    ? compileRecursiveSpec(rawSpec, { autoRepair: true })
    : JSON.parse(JSON.stringify(rawSpec));
  if (!Number.isInteger(steps) || steps < 1 || steps > 256) {
    throw new Error(`unrollRolloutSpec: steps must be an integer in 1..256, got ${JSON.stringify(steps)}`);
  }
  const targetStateSlot = resolveStatePortSlot(spec, statePort);
  const ports = portBySlot(spec);
  if (!ports.has(targetStateSlot)) {
    throw new Error(`unrollRolloutSpec: state port slot ${targetStateSlot} is not a declared port`);
  }
  const shapes = inferShapes(JSON.parse(JSON.stringify(spec)));
  const stateShape = ports.get(targetStateSlot).shape;
  const outShape = shapes.get(spec.outputSlot);
  if (!outShape || outShape.join(',') !== stateShape.join(',')) {
    throw new Error(
      `unrollRolloutSpec: outputSlot ${spec.outputSlot} shape [${outShape}] does not match state port ${targetStateSlot} shape [${stateShape}]`,
    );
  }

  const portSlots = new Set((spec.ports ?? []).map((p) => p.slot));
  let nextSlot = Math.max(spec.numSlots ?? 0, ...[...portSlots].map((s) => s + 1), 0);
  const unrolledSteps = [];
  let prevStateOutSlot = targetStateSlot;
  const tickOutputSlots = [];

  for (let t = 0; t < steps; t++) {
    const slotMap = new Map();
    for (const pSlot of portSlots) {
      slotMap.set(pSlot, pSlot === targetStateSlot ? prevStateOutSlot : pSlot);
    }
    for (let si = 0; si < spec.steps.length; si++) {
      const st = spec.steps[si];
      const inCount = st.kind === 'binary' ? 2 : 1;
      const origIn = st.slots.slice(0, inCount);
      const origOut = st.slots[inCount];
      const mappedIn = origIn.map((sl) => {
        if (!slotMap.has(sl)) {
          throw new Error(`unrollRolloutSpec: tick ${t} step ${si} reads unmapped slot ${sl}`);
        }
        return slotMap.get(sl);
      });
      const mappedOut = nextSlot++;
      slotMap.set(origOut, mappedOut);
      unrolledSteps.push({
        kind: st.kind,
        layer: st.layer,
        slots: [...mappedIn, mappedOut],
      });
    }
    prevStateOutSlot = slotMap.get(spec.outputSlot);
    tickOutputSlots.push(prevStateOutSlot);
  }

  let unrolledSpec = {
    numSlots: Math.max(nextSlot, prevStateOutSlot + 1),
    layers: JSON.parse(JSON.stringify(spec.layers)),
    steps: unrolledSteps,
    outputSlot: prevStateOutSlot,
    ports: JSON.parse(JSON.stringify(spec.ports)),
    weights: JSON.parse(JSON.stringify(spec.weights ?? {})),
    inputs: JSON.parse(JSON.stringify(spec.inputs ?? {})),
  };

  unrolledSpec = autoRepairSpec(unrolledSpec).spec;
  let compactionInfo = null;
  if (compactSlots) {
    const comp = compactSlotsByLiveness(unrolledSpec);
    unrolledSpec = comp.spec;
    compactionInfo = {
      slots_before: comp.slots_before,
      slots_after: comp.slots_after,
      slots_saved: comp.slots_saved,
    };
  }

  unrolledSpec.meta = {
    ...(spec.meta ?? {}),
    unrolled_rollout: {
      temporal_horizon: steps,
      state_port_slot: targetStateSlot,
      one_step_count: spec.steps.length,
      unrolled_steps_count: unrolledSpec.steps.length,
      weight_tied: true,
      tick_output_slots: tickOutputSlots,
      slot_compaction: compactionInfo,
    },
  };
  return unrolledSpec;
}

export async function rolloutScan(rt, rawSpec, {
  steps = 10,
  statePort = null,
  inputSequence = null,
  recordTrajectory = true,
} = {}) {
  const spec = rawSpec.schema || rawSpec.subgraphs ? compileRecursiveSpec(rawSpec, { autoRepair: true }) : rawSpec;
  if (!Number.isInteger(steps) || steps < 1 || steps > 10000) {
    throw new Error(`rolloutScan: steps must be an integer in 1..10000, got ${JSON.stringify(steps)}`);
  }
  const targetStateSlot = resolveStatePortSlot(spec, statePort);
  const ports = portBySlot(spec);
  if (!ports.has(targetStateSlot)) {
    throw new Error(`rolloutScan: state port slot ${targetStateSlot} is not a declared port`);
  }

  return await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
    let curInputs = { ...(spec.inputs ?? {}) };
    const trajectory = [];
    let lastValues = null;
    let lastShape = null;

    for (let t = 0; t < steps; t++) {
      if (Array.isArray(inputSequence) && inputSequence[t] && typeof inputSequence[t] === 'object') {
        for (const [k, v] of Object.entries(inputSequence[t])) {
          const slot = spec.meta?.symbol_table?.[k] ?? Number(k);
          curInputs[String(slot)] = v;
        }
      }
      const inStateVals = curInputs[String(targetStateSlot)];
      const inStateSha = sha256F32(inStateVals);

      const b = bundle(curInputs, ports);
      const out = g.run(reg, b);
      let vals;
      let shape;
      try {
        vals = Array.from(out.to_array());
        shape = Array.from(out.shape());
      } finally {
        out.free();
        if (b && typeof b.free === 'function') {
          try { b.free(); } catch { /* tracked by scope as fallback */ }
        }
      }

      for (let i = 0; i < vals.length; i++) {
        if (!isFiniteF32(vals[i])) {
          const err = new Error(`rolloutScan: non-finite output at tick ${t}, index ${i}: ${vals[i]}`);
          err.code = 'non_finite_rollout';
          throw err;
        }
      }

      const outSha = sha256F32(vals);
      if (recordTrajectory) {
        trajectory.push({
          tick: t,
          input_state_sha256: inStateSha,
          output_sha256: outSha,
          values: vals,
        });
      }

      const statePortDef = ports.get(targetStateSlot);
      const expectedLen = statePortDef.shape.reduce((a, c) => a * c, 1);
      if (vals.length !== expectedLen && t < steps - 1) {
        throw new Error(
          `rolloutScan: output length (${vals.length}) does not match state port ${targetStateSlot} length (${expectedLen}) for handoff`,
        );
      }
      curInputs[String(targetStateSlot)] = vals;
      lastValues = vals;
      lastShape = shape;
    }

    return {
      ok: true,
      mode: 'scan',
      steps,
      state_port_slot: targetStateSlot,
      final_output: { shape: lastShape, values: lastValues, value_sha256: sha256F32(lastValues) },
      trajectory: recordTrajectory ? trajectory : undefined,
    };
  });
}

export async function rolloutFixpoint(rt, rawSpec, {
  statePort = null,
  maxSteps = 100,
  tol = 1e-6,
  divergenceThreshold = 1e6,
  recordTrajectory = true,
} = {}) {
  const spec = rawSpec.schema || rawSpec.subgraphs ? compileRecursiveSpec(rawSpec, { autoRepair: true }) : rawSpec;
  if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 10000) {
    throw new Error(`rolloutFixpoint: maxSteps must be an integer in 1..10000, got ${JSON.stringify(maxSteps)}`);
  }
  if (typeof tol !== 'number' || !Number.isFinite(tol) || tol < 0) {
    throw new Error(`rolloutFixpoint: tol must be a finite non-negative number, got ${JSON.stringify(tol)}`);
  }

  const targetStateSlot = resolveStatePortSlot(spec, statePort);
  const ports = portBySlot(spec);
  if (!ports.has(targetStateSlot)) {
    throw new Error(`rolloutFixpoint: state port slot ${targetStateSlot} is not a declared port`);
  }
  const statePortDef = ports.get(targetStateSlot);
  const expectedLen = statePortDef.shape.reduce((a, c) => a * c, 1);

  return await withGraphScope(rt, spec, async ({ g, reg, bundle }) => {
    let curInputs = { ...(spec.inputs ?? {}) };
    let prevState = [...(curInputs[String(targetStateSlot)] ?? [])];
    if (prevState.length !== expectedLen) {
      throw new Error(`rolloutFixpoint: initial state at slot ${targetStateSlot} has length ${prevState.length}, expected ${expectedLen}`);
    }

    const trajectory = [];
    let prevDelta = null;
    // B8 fix: the last contraction ratio used to be read back out of
    // the recorded trajectory, so with recordTrajectory:false the
    // reported empirical_lipschitz was always null. Track it in a
    // scalar updated on every step instead; trajectory recording
    // stays gated by the flag.
    let lastContractionRatio = null;
    let converged = false;
    let status = 'MAX_STEPS_EXCEEDED';
    let lastShape = [...statePortDef.shape];
    let stepsRun = 0;

    for (let t = 0; t < maxSteps; t++) {
      stepsRun = t + 1;
      const b = bundle(curInputs, ports);
      const out = g.run(reg, b);
      let nextState;
      try {
        nextState = Array.from(out.to_array());
        lastShape = Array.from(out.shape());
      } finally {
        out.free();
        if (b && typeof b.free === 'function') {
          try { b.free(); } catch { /* ignore */ }
        }
      }

      if (nextState.length !== expectedLen) {
        throw new Error(`rolloutFixpoint: output length ${nextState.length} != state port length ${expectedLen}`);
      }

      for (let i = 0; i < nextState.length; i++) {
        if (!isFiniteF32(nextState[i])) {
          return {
            ok: false,
            mode: 'fixpoint',
            converged: false,
            status: 'DIVERGED_NON_FINITE',
            steps_run: stepsRun,
            state_port_slot: targetStateSlot,
            error: `non-finite state value at tick ${t}, index ${i}: ${nextState[i]}`,
            trajectory: recordTrajectory ? trajectory : undefined,
          };
        }
      }

      let deltaInf = 0;
      for (let i = 0; i < nextState.length; i++) {
        const d = Math.abs(nextState[i] - prevState[i]);
        if (d > deltaInf) deltaInf = d;
      }

      const contractionRatio = (prevDelta !== null && prevDelta > 1e-12)
        ? +(deltaInf / prevDelta).toFixed(6)
        : (prevDelta === 0 && deltaInf === 0 ? 0 : null);
      lastContractionRatio = contractionRatio;

      if (recordTrajectory) {
        trajectory.push({
          tick: t,
          delta_inf: deltaInf,
          contraction_ratio: contractionRatio,
          state_sha256: sha256F32(nextState),
          values: nextState,
        });
      }

      if (deltaInf > divergenceThreshold) {
        return {
          ok: false,
          mode: 'fixpoint',
          converged: false,
          status: 'DIVERGED_EXPLOSIVE',
          steps_run: stepsRun,
          state_port_slot: targetStateSlot,
          final_delta_inf: deltaInf,
          error: `state step delta ${deltaInf} exceeded divergenceThreshold ${divergenceThreshold} at tick ${t}`,
          trajectory: recordTrajectory ? trajectory : undefined,
        };
      }

      const prevDeltaSnapshot = prevDelta;
      prevState = nextState;
      prevDelta = deltaInf;
      curInputs[String(targetStateSlot)] = nextState;

      // Exact algebraic fixed point (zero change on tick) converges immediately.
      if (deltaInf === 0) {
        converged = true;
        status = 'CONVERGED';
        break;
      }

      // Non-zero step within tol: require at least a second step (prevDeltaSnapshot !== null)
      // AND genuine strict contraction (deltaInf < prevDeltaSnapshot * (1 - 1e-6)) so constant-drift
      // or expanding systems starting near the origin are never falsely certified as CONVERGED.
      if (deltaInf <= tol && prevDeltaSnapshot !== null) {
        if (deltaInf < prevDeltaSnapshot * (1 - 1e-6)) {
          converged = true;
          status = 'CONVERGED';
          break;
        }
        status = contractionRatio !== null && contractionRatio > 1.0
          ? 'DIVERGED_EXPANDING'
          : 'NON_CONTRACTIVE_DRIFT';
      }
    }

    const lastRatio = lastContractionRatio !== null ? lastContractionRatio : (prevDelta === 0 ? 0 : null);
    return {
      ok: converged,
      mode: 'fixpoint',
      converged,
      status,
      steps_run: stepsRun,
      max_steps: maxSteps,
      tol,
      state_port_slot: targetStateSlot,
      final_delta_inf: prevDelta,
      empirical_lipschitz: lastRatio,
      fixed_point: {
        shape: lastShape,
        values: prevState,
        value_sha256: sha256F32(prevState),
      },
      trajectory: recordTrajectory ? trajectory : undefined,
    };
  });
}

// Extract a minimal runnable sub-spec up to step `uptoStep` (inclusive) to isolate a bug or divergence.
export function extractPrefixSubspec(spec, uptoStep) {
  const k = Math.min(Math.max(0, uptoStep), spec.steps.length - 1);
  const subSteps = spec.steps.slice(0, k + 1).map((s) => ({ ...s, slots: [...s.slots] }));
  const targetOutSlot = subSteps[subSteps.length - 1].slots[subSteps[subSteps.length - 1].slots.length - 1];
  const rawSub = {
    ...JSON.parse(JSON.stringify(spec)),
    steps: subSteps,
    outputSlot: targetOutSlot,
  };
  return compileRecursiveSpec(rawSub, { autoRepair: true, validate: false });
}

export async function bisectDivergence(rt, baseSpec, candSpec, { maxTensorBytes = 1048576 } = {}) {
  const captureTrace = async (s) => {
    return await withGraphScope(rt, s, async ({ g, reg, bundle }) => {
      const b = bundle(s.inputs, portBySlot(s));
      const traced = g.runWithTrace(reg, b, 0, s.steps.length, maxTensorBytes);
      try {
        const rep = JSON.parse(traced.report());
        return (rep.steps ?? []).map((st) => ({
          index: st.index,
          layer: s.layers[s.steps[st.index]?.layer]?.constructor ?? `type:${st.layer_type}`,
          out_slot: st.output_slot ?? null,
          out_shape: st.observation?.shape ?? null,
          out_bytes: st.observation?.byte_length ?? null,
          capture_status: st.observation?.capture_status ?? null,
          value_sha256: st.observation?.value_sha256 ?? null,
          sample: st.observation?.sample ?? null,
          stats: st.observation?.stats ?? null,
        }));
      } finally {
        if (traced.free) traced.free();
      }
    });
  };

  const baseSteps = await captureTrace(baseSpec);
  const candSteps = await captureTrace(candSpec);
  const minLen = Math.min(baseSteps.length, candSteps.length);
  let firstDivergent = null;
  const uncapturedSteps = [];

  for (let i = 0; i < minLen; i++) {
    const a = baseSteps[i];
    const b = candSteps[i];
    if (a.value_sha256 && b.value_sha256) {
      if (a.value_sha256 !== b.value_sha256) {
        firstDivergent = i;
        break;
      }
    } else {
      uncapturedSteps.push(i);
      const shapeDiff = JSON.stringify(a.out_shape) !== JSON.stringify(b.out_shape);
      const sampleDiff = a.sample && b.sample && JSON.stringify(a.sample) !== JSON.stringify(b.sample);
      const statsDiff = a.stats && b.stats && JSON.stringify(a.stats) !== JSON.stringify(b.stats);
      if (shapeDiff || sampleDiff || statsDiff) {
        firstDivergent = i;
        break;
      }
    }
  }

  if (firstDivergent === null && baseSteps.length !== candSteps.length) {
    firstDivergent = minLen;
  }

  const outputSlotEqual = baseSpec.outputSlot === candSpec.outputSlot;
  let isolatedBaseSubspec = null;
  let isolatedCandSubspec = null;
  if (firstDivergent !== null) {
    isolatedBaseSubspec = firstDivergent < baseSpec.steps.length ? extractPrefixSubspec(baseSpec, firstDivergent) : baseSpec;
    isolatedCandSubspec = firstDivergent < candSpec.steps.length ? extractPrefixSubspec(candSpec, firstDivergent) : candSpec;
  }

  return {
    ok: true,
    identical_trace: firstDivergent === null && baseSteps.length === candSteps.length && outputSlotEqual && uncapturedSteps.length === 0,
    output_slot_equal: outputSlotEqual,
    output_slots: [baseSpec.outputSlot, candSpec.outputSlot],
    uncaptured_steps: uncapturedSteps,
    first_divergent_step: firstDivergent,
    divergent_step_detail: firstDivergent !== null ? {
      step_index: firstDivergent,
      baseline: baseSteps[firstDivergent] ?? null,
      candidate: candSteps[firstDivergent] ?? null,
    } : null,
    isolated_base_subspec: isolatedBaseSubspec,
    isolated_cand_subspec: isolatedCandSubspec,
  };
}
