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
// warnings (constructors without a reliable rule) never fail.
import { loadSpec } from '../lib/spec.mjs';
import { printJson } from '../lib/runner.mjs';
import { checkSpec } from '../lib/shapecore.mjs';

const [specFile] = process.argv.slice(2);
if (!specFile) {
  console.error('usage: node shapecheck.mjs <spec.json>');
  process.exit(2);
}

try {
  const spec = loadSpec(specFile);
  const result = checkSpec(spec);
  printJson({ ok: result.ok, ...result });
  process.exit(result.ok ? 0 : 2);
} catch (e) {
  printJson({ ok: false, error: e.message });
  process.exit(2);
}
