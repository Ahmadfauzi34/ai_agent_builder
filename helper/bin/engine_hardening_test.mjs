#!/usr/bin/env node
// Regression tests for helper/lib/engine.mjs agent-hardening helpers.
// Covers PR #387 (checkOperation param defaults, withTensorScope LIFO free)
// plus the null/undefined-shape gap (shapes were missed by #387's guard).
//
// Usage: node bin/engine_hardening_test.mjs
// Exit 0 = all pass, 1 = any failure. Needs the wasm pkg (BURN_WASM_PKG_DIR
// or the default relative vendor path).
import { loadWasm, checkOperation, withTensorScope } from '../lib/engine.mjs';

const results = [];
const t = (name, ok, extra = '') => {
  results.push(ok);
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${extra ? '  ' + extra : ''}`);
};

const rt = await loadWasm();

// 1. Valid op -> admissible verdict (parsed JSON).
{
  const r = checkOperation(rt, 'numeric.add', [1, 4, 1, 1], [1, 4, 1, 1]);
  t('valid add -> admissible', r.status === 'admissible', r.status);
}
// 2. PR #387: null/undefined params default to empty arrays (no TypeError).
{
  const r = checkOperation(rt, 'numeric.add', [1, 4, 1, 1], [1, 4, 1, 1], null, undefined);
  t('null u32/f32 params -> no throw', r.status === 'admissible', r.status);
}
// 3-4. NEW GAP: null/undefined shapes must degrade to a structured
// "rejected" verdict, not a raw TypeError from Uint32Array.from.
for (const [label, lhs, rhs] of [['null', null, null], ['undefined', undefined, undefined], ['mixed', null, [1, 4, 1, 1]]]) {
  let status = null, threw = null;
  try {
    status = checkOperation(rt, 'numeric.add', lhs, rhs).status;
  } catch (e) { threw = String(e).slice(0, 60); }
  t(`${label} shapes -> rejected (not TypeError)`, status === 'rejected', threw ?? status);
}
// 5. Unknown op -> loud throw (not a silent verdict).
{
  let threw = false;
  try { checkOperation(rt, 'nope.unknown', [1, 4, 1, 1], [1, 4, 1, 1]); }
  catch { threw = true; }
  t('unknown op -> throws', threw);
}
// 6. withTensorScope returns fn's value on success.
{
  const v = await withTensorScope(rt, async (scope) => {
    scope.tensor([1, 2], [1, 2, 1, 1]);
    return 42;
  });
  t('scope returns value', v === 42);
}
// 7. withTensorScope rethrows the original error (teardown never masks it),
//    and a second scope still works afterwards (no corrupted state).
{
  let seen = null;
  try {
    await withTensorScope(rt, async (scope) => {
      scope.tensor([1, 2], [1, 2, 1, 1]);
      throw new Error('boom-original');
    });
  } catch (e) { seen = e.message; }
  const v2 = await withTensorScope(rt, async () => 'alive');
  t('scope rethrows original + reusable', seen === 'boom-original' && v2 === 'alive', seen);
}
// 8. scope.release() untracks: released tensor survives teardown.
{
  let kept = null;
  await withTensorScope(rt, async (scope) => {
    const a = scope.tensor([5], [1, 1, 1, 1]);
    kept = scope.release(a);
  });
  let ok = false;
  try { ok = Array.from(kept.to_array())[0] === 5; kept.free(); }
  catch { ok = false; }
  t('released tensor survives scope', ok);
}

const fails = results.filter((x) => !x).length;
console.log(`${results.length - fails}/${results.length} pass`);
process.exit(fails ? 1 : 0);
