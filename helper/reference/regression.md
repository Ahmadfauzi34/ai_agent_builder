# Regression gate (v2.5)

Run it before any release: `node bin/regression.mjs` from the pack root.
Exit 0 = every case passes; exit 1 = at least one fails, with the failing
case names in the JSON summary. The suite builds its specs in a temp dir
and removes it at the end. It pins CLI behavior (exit code + parsed
output), never just "the tool ran".

## Where the cases come from

- **Basics**: builder counts (48/169/48 + wasm sha), run/validate/verify/
  checkpoint round-trip on the shipped examples, including the known
  checkpoint sha for `linear-relu` (`8c209c9f…`) and restore values
  `[4.5, 10.5]`.
- **Gemini Fase 1 (REG-01..15)** and **Fase 2 Gelombang 2/3 (EC1–EC5,
  F1–F6)**: each note's bug is a case here, named by its note id:
  registry id 2^32 (REG-10), weights 1e39 (REG-11), weights key `""`
  (REG-12), fractional/collision slots (F2), dim-0 shapes (REG-03/F3),
  conv stride 0 (REG-04), softmax dim 0 validate + engine (REG-13),
  leakyRelu/prelu/softplus bounds (REG-13/14), prototype-pollution
  names (REG-09), `--help` in second position (REG-08), missing cases
  file exit 1 (REG-06), dirty and padding-bit checkpoint base64
  (REG-02), layout/role warnings + adapter metadata (REG-07), intAdd
  i64 warning (REG-15/F1).
- **v2.5 fixes**: `expectedWeightCount` table + validate warnings
  (F4), verify tolerance validation (F5), drain-safe stdout over the
  64KB pipe buffer via a 100-pair `--batch` (F6).

## Honest gaps (what this suite does NOT claim)

- **REG-01 end-to-end**: the stale-shape false-reject scenario from the
  note could not be reproduced end-to-end (a probe failed identically
  in both variants, because `matmul` now has a shape rule in the
  hardened variant). What the suite pins instead is the mechanism that
  closes the bug class: `shapecore` deletes a stale output shape on
  rule-less overwrite / unknown input / rule error (read the code),
  plus the rule-coverage case asserting all 14 formerly rule-less
  constructors now have shape rules. A future spec that reproduces a
  stale-shape false reject should be added as a CLI case here.
- **F4 conv bias**: the conv formulas add `out_ch` for bias
  unconditionally, per the Fase 2 note's table. If a future engine
  build makes conv bias optional, the table and
  `expectedWeightCount` must be updated together, and this doc with
  them. The warnings are warnings (not errors) precisely so a formula
  disagreement cannot falsely reject a valid spec.
- **F6 scope**: the drain-safe guarantee covers every CLI's JSON
  output path (`printJsonAndExit` in `lib/runner.mjs`, used by all
  tools; `failCli` uses it too). The write is synchronous and looped:
  a single `fs.writeSync` to a pipe may accept only part of a large
  buffer (observed during development: 146,176 of 323,003 bytes), so
  the helper loops until every byte is accepted, waiting briefly on
  EAGAIN. A drain-deferred exit was tried first and rejected: it let
  `verify --batch` fall through to single-pair mode, printing a
  second JSON error after the batch JSON. Usage text on stderr still
  exits directly, which is safe because no stdout JSON is in flight
  there. The 100-pair batch case is the live proof: its output
  exceeds 64KB, and the suite asserts the full output parses as one
  JSON value.

## Adding a case

1. Give it the note id (or a descriptive name for new bugs).
2. Assert exit code AND parsed output fields, not just "no crash".
3. If the case needs a spec, build it in the suite's temp dir, never
   by editing a shipped example.
4. Run the suite twice: once from the pack dir, once from the
   unzipped release zip (they must both pass — see the v2.5 release
   note for the procedure).
