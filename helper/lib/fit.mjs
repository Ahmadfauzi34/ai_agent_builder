// burn-math fit: shared PURE helpers for the Optimasi & fitting domain pack.
//
// No WASM imports here: bin adapters supply forward runs (engine.mjs in the
// repo, wasm_direct.mjs in the skill) and this module only does arithmetic
// over plain arrays. That keeps it byte-identical in repo and skill.
//
// Design stance (same as gradcheck): these are VERIFIERS, not optimizers.
// They measure and report; the caller decides. A measurement tool exits 0
// when the measurement succeeded, even when the finding is "ascent" or
// "untrusted" — those are findings, not tool failures. Exit 2 is reserved
// for hard failures: bad input, a recomputation mismatch, a crashed run.
//
// All three tools are forward-run-based (finite differences), so they need
// no backward pass and no engine change (core WASM is enough).

// Scalarization rule (documented, single rule, no surprises):
// a scalar output (exactly one element) IS the loss; a vector output is
// scalarized as the sum of squares. If your GD optimizes a different loss,
// express it as a spec whose output IS that loss.
export function scalarize(outputs) {
  if (outputs.length === 1) return outputs[0];
  let s = 0;
  for (const v of outputs) s += v * v;
  return s;
}

// Second-order central difference for one diagonal entry:
//   H_ii = (f(x + h e_i) - 2 f(x) + f(x - h e_i)) / h^2
// Cost note (deterministic before running): a full diagonal over n inputs
// costs 4n + 1 forward runs (plus/minus at two step sizes, plus the base).
export function hessianDiagSecondOrder(fPlus, fBase, fMinus, h) {
  return (fPlus - 2 * fBase + fMinus) / (h * h);
}

// Trust for one Hessian entry measured at two step sizes (coarse h, fine h/2).
// f32 cancellation noise grows as h shrinks, truncation error as h grows,
// so agreement across step sizes is the only honest trust signal available
// from forward runs alone. Three states, no silent numbers:
//   trusted           — the two estimates agree within tolerance.
//   zero_within_noise — both estimates are negligible; the entry is zero
//                       as far as forward runs can tell (common for linear
//                       and piecewise-linear specs, whose true Hessian is 0).
//   untrusted         — the estimates disagree and neither is negligible;
//                       the caller must not use the value.
// value reported is the finer-step estimate.
export function hessianTrust(coarse, fine, tolAbs, tolRel) {
  const diff = Math.abs(fine - coarse);
  const mag = Math.max(Math.abs(fine), Math.abs(coarse));
  if (Math.abs(fine) <= tolAbs && Math.abs(coarse) <= tolAbs) {
    return { value: fine, coarse, diff, status: 'zero_within_noise' };
  }
  if (diff <= tolAbs + tolRel * mag) {
    return { value: fine, coarse, diff, status: 'trusted' };
  }
  return { value: fine, coarse, diff, status: 'untrusted' };
}

// Rank one Jacobian {rows, cols, values} (rows = outputs, cols = inputs).
// Returns entries sorted by |value| desc, per-output top-K, a global top-K,
// threshold flags, and a summary. Pure structure: no golden values here,
// so the battery asserts ordering and completeness, not magic numbers.
export function rankJacobian(J, topK, threshold) {
  const entries = [];
  for (let o = 0; o < J.rows; o++) {
    for (let i = 0; i < J.cols; i++) {
      const value = J.values[o][i];
      entries.push({ out: o, in: i, value, abs: Math.abs(value) });
    }
  }
  entries.sort((a, b) => b.abs - a.abs);
  const perOutput = [];
  for (let o = 0; o < J.rows; o++) {
    const row = entries.filter((e) => e.out === o).slice(0, topK);
    perOutput.push({ out: o, top: row });
  }
  let sumAbs = 0;
  let maxAbs = 0;
  let above = 0;
  for (const e of entries) {
    sumAbs += e.abs;
    if (e.abs > maxAbs) maxAbs = e.abs;
    if (e.abs >= threshold) above++;
  }
  return {
    entries,
    global_top: entries.slice(0, topK),
    per_output: perOutput,
    summary: {
      total_entries: entries.length,
      max_abs: maxAbs,
      mean_abs: entries.length ? sumAbs / entries.length : 0,
      count_at_or_above_threshold: above,
    },
  };
}

// GD trajectory verdicts (pure). Inputs:
//   steps: [{weights, loss, grad?, lr?}] — the CLAIMED trajectory.
//   recomputed: [loss] — the tool's own forward-run losses, same order.
//   tol: acceptance tolerance for recomputation (abs and rel, like gradcheck).
// Hard check: recomputed loss must match claimed loss within tol. A mismatch
// is a tool failure (exit 2 in the bin adapter): the trajectory does not
// describe the spec it claims to.
// Soft findings (reported, never hidden, never fatal):
//   ascent — recomputed loss increased vs the previous step. GD with too
//     large a step legitimately ascends; that is a finding about the
//     trajectory, not proof the tool failed.
//   prediction_off — when the PREVIOUS step's grad and lr are supplied,
//     the actual decrease is compared to the first-order prediction
//     lr * |grad|^2 of the step that was taken (departure gradient: the
//     step from s-1 to s was computed from the gradient AT s-1, not the
//     arrival gradient). Ratio outside [0, 2] is flagged (overshoot gives
//     ratio < 0; a much larger than predicted drop gives ratio > 2). The
//     range is a heuristic flag, documented as such, not a verdict. For a
//     quadratic loss the ratio is exactly 1 - lr, a useful sanity check.
export function verifyGdSteps(steps, recomputed, tol) {
  const perStep = [];
  let mismatchCount = 0;
  let ascentCount = 0;
  let predictionOffCount = 0;
  for (let s = 0; s < steps.length; s++) {
    const claimed = steps[s].loss;
    const actual = recomputed[s];
    const absErr = Math.abs(actual - claimed);
    const relErr = absErr / Math.max(Math.abs(claimed), 1e-12);
    const recomputedOk = absErr <= tol || relErr <= tol;
    if (!recomputedOk) mismatchCount++;
    let descent = null;
    let delta = null;
    if (s > 0) {
      delta = recomputed[s - 1] - actual;
      descent = delta >= 0;
      if (!descent) ascentCount++;
    }
    let gradNorm = null;
    let predictedDecrease = null;
    let predictionRatio = null;
    let predictionOff = null;
    if (steps[s].grad !== undefined) {
      let sq = 0;
      for (const g of steps[s].grad) sq += g * g;
      gradNorm = Math.sqrt(sq);
    }
    // The step from s-1 to s was taken with s-1's gradient and lr.
    if (s > 0 && steps[s - 1].grad !== undefined && steps[s - 1].lr !== undefined) {
      let sq = 0;
      for (const g of steps[s - 1].grad) sq += g * g;
      predictedDecrease = steps[s - 1].lr * sq;
      if (predictedDecrease > 0) {
        predictionRatio = delta / predictedDecrease;
        predictionOff = predictionRatio < 0 || predictionRatio > 2;
        if (predictionOff) predictionOffCount++;
      }
    }
    perStep.push({
      step: s,
      loss_claimed: claimed,
      loss_recomputed: actual,
      abs_error: absErr,
      rel_error: relErr,
      recomputed_ok: recomputedOk,
      descent,
      delta,
      grad_norm: gradNorm,
      predicted_decrease: predictedDecrease,
      prediction_ratio: predictionRatio,
      prediction_off: predictionOff,
    });
  }
  const firstNorm = perStep.find((p) => p.grad_norm !== null)?.grad_norm ?? null;
  const lastNorm = [...perStep].reverse().find((p) => p.grad_norm !== null)?.grad_norm ?? null;
  return {
    per_step: perStep,
    mismatch_count: mismatchCount,
    ascent_count: ascentCount,
    prediction_off_count: predictionOffCount,
    grad_norm_first: firstNorm,
    grad_norm_last: lastNorm,
    ok: mismatchCount === 0,
  };
}
