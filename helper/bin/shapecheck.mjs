#!/usr/bin/env node
// burn-math shapecheck: static shape + dataflow check of a spec (no engine).
// Usage: node shapecheck.mjs <spec.json>
//
// Propagates 4-D shapes through steps using the rules in ../lib/shapes.mjs,
// checks constructor arity vs step kind, and verifies dataflow (every input
// slot is a declared port or a previous step's output). Catches dim mismatch
// in milliseconds, before the engine spins up — the mandatory first gate of
// any graph-rewrite design loop.
//
// Prints {ok, steps:[{index,layer,kind,in_shapes,out_shape,rule}], mismatches,
// warnings, dataflow}. Exit 0 iff no mismatches and no dataflow errors;
// warnings (constructors without a reliable rule, weight/input cross-check
// mismatches) never fail.
import { loadSpec, checkArgValues, checkInputLengths, expectedWeightCount } from '../lib/spec.mjs';
import { printJsonAndExit , errText } from '../lib/runner.mjs';
import { checkSpec } from '../lib/shapecore.mjs';

const [specFile] = process.argv.slice(2);
if (!specFile || process.argv.slice(2).includes('--help') || process.argv.slice(2).includes('-h')) {
  console.error('usage: node shapecheck.mjs <spec.json>');
  process.exit(2);
}

try {
  const spec = loadSpec(specFile);
  // Ported from pack v2.4 (Audit_7 E4): strict arg/value typing BEFORE
  // the shape check, so a mistyped arg fails with a clear
  // invalid_argument instead of confusing the shape rules (or being
  // silently coerced at run time).
  const argIssues = checkArgValues(spec);
  if (argIssues.length) {
    printJsonAndExit({
      ok: false, valid: false, stage: 'static', code: 'invalid_argument',
      issues: argIssues,
      remediation_hint: 'layer args must match the catalog parameters: required-arg counts, u32 = integer in 0..4294967295, f32 = finite after fround, bool = real boolean, null only for Option params, registry ids unique; weights/inputs are finite-f32 arrays keyed by canonical u32 slots; port shapes are 4-D positive-u32 arrays',
    }, 2);
  }
  const result = checkSpec(spec);
  // Port v2.5 (F4): weight-count and input-length cross-checks, as
  // warnings (never failures — a spec without weights is still valid,
  // the engine supplies defaults). Weights are keyed by layer index.
  const warnings = result.warnings ?? [];
  for (const [layerIdxStr, values] of Object.entries(spec.weights ?? {})) {
    const li = Number(layerIdxStr);
    const decl = spec.layers[li];
    if (!decl) {
      warnings.push(`weights key '${layerIdxStr}': layer index ${li} does not exist (0..${spec.layers.length - 1})`);
    } else {
      const expected = expectedWeightCount(decl);
      if (expected === null) {
        warnings.push(`weights key '${layerIdxStr}': layer ${li} (${decl.constructor}) carries no mutable weights`);
      } else if (Array.isArray(values) && values.length !== expected) {
        warnings.push(`weights layer ${li} (${decl.constructor}): expected ${expected} floats, got ${values.length}`);
      }
    }
  }
  const portBySlot = new Map((spec.ports ?? []).map((p) => [p?.slot, p]));
  for (const [slotStr, values] of Object.entries(spec.inputs ?? {})) {
    const slot = Number(slotStr);
    const port = portBySlot.get(slot);
    if (!port) {
      warnings.push(`inputs slot ${slotStr}: no declared port for slot ${slotStr}`);
    }
  }
  // Tahap 2a: input-length mismatches FAIL (they used to be warnings
  // only, so `shapecheck bad_len.json` exited 0 with ok:true). The
  // check is checkInputLengths — the same comparison validate warns
  // about. The record keeps the shapecheck evidence (steps, mismatches,
  // dataflow, warnings) so both signals survive when they coincide.
  const lenIssues = checkInputLengths(spec);
  if (lenIssues.length) {
    printJsonAndExit({
      ok: false, valid: false, stage: 'static', code: 'invalid_argument',
      issues: lenIssues, steps: result.steps, mismatches: result.mismatches,
      dataflow: result.dataflow, warnings,
    }, 2);
  }
  result.warnings = warnings;
  printJsonAndExit({ ok: result.ok, ...result }, result.ok ? 0 : 2);
} catch (e) {
  printJsonAndExit({ ok: false, error: errText(e) }, 2);
}
