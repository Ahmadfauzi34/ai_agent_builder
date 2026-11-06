// burn-math gates (Tahap 2a): the fail-closed static gates bin/run.mjs
// already runs, as one shared helper so every engine-calling bin can
// reject a statically defective spec BEFORE any WASM work — with the
// slot/step attribution the bare engine error lacks.
//
// Order mirrors run.mjs: arg typing, then input lengths, then the
// static shape/dataflow gate. Returns null when the spec passes, else
// the failure record (same shape run.mjs emits, so a future shared
// buildGraph — Tahap 2b — can adopt it without changing the contract).
import { checkArgValues, checkInputLengths } from './spec.mjs';
import { checkSpec } from './shapecore.mjs';

export function gateSpec(spec) {
  const argIssues = checkArgValues(spec);
  if (argIssues.length) {
    return { ok: false, stage: 'argValues', code: 'invalid_argument', issues: argIssues, error_envelope: null };
  }
  const lenIssues = checkInputLengths(spec);
  if (lenIssues.length) {
    return { ok: false, stage: 'argValues', code: 'invalid_argument', issues: lenIssues, error_envelope: null };
  }
  const checked = checkSpec(spec);
  if (checked.mismatches.length || !checked.dataflow.ok) {
    return {
      ok: false, stage: 'build', code: 'static_check_failed',
      mismatches: checked.mismatches, dataflow: checked.dataflow,
      error: checked.mismatches[0]?.message ?? checked.dataflow.issues[0] ?? 'static check failed',
      remediation_hint: 'fix the listed steps/slots/shapes, then re-run validate',
      error_envelope: null,
    };
  }
  return null;
}
