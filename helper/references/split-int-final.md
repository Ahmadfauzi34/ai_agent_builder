# Recipe: running an int-final spec by splitting it (fork F1 = A)

## When this applies

`run.mjs` emits f32 outputs only (v1 rule, fork F1 decision A,
2026-11-05). A spec whose output slot is i64 is rejected at build
time with a message that names the split. This recipe is that
split, done by hand.

## Why the split is necessary

`run.mjs` outputs `{shape, values}` with values as JSON numbers.
JSON numbers cannot hold i64 exactly above 2^53 — above that they
round silently. `int.mjs` avoids this by using decimal strings
(`"20000104"`, never `20000104` as a number). The split keeps the
exactness: f32 parts run as numbers (they are f32 by nature), int
parts run as strings (exact), and the two never mix through a
lossy conversion.

## The walk (general)

1. **Find the boundary.** Read the rejection message: it names the
   f32 step range(s) and the int tail step range. (Equivalently:
   walk the steps in order; a step is int iff its constructor is
   `intAdd/intSub/intMul/intMatmul/intDiv/intRem/intToFloat/
   intFromFloat`, else f32. Consecutive same-kind steps form one
   segment.)
2. **Each f32 segment, in order:** build an f32-only sub-spec —
   same `layers`, only that segment's `steps`, `ports` = the
   segment's inputs (original slot ids, with their shapes),
   `outputSlot` = the segment's output (one run per output if a
   segment has several) — and run it with `run.mjs`. Its outputs
   are the next segment's inputs.
3. **Each int step, in order:** run it with `int.mjs`, values as
   decimal strings: casts via `from-f32` (f32->i64, truncates
   toward zero; rejects non-finite and out-of-i64-range inputs)
   and `to-f32` (i64->f32, rejects |x| > 16777216 unless
   `--allow-inexact`); binary ops via `add/sub/mul/div/rem/
   matmul`. Shapes: casts preserve the shape, elementwise ops
   require identical input shapes.
4. **The int tail's final values are the exact answer.** Keep them
   as strings. Do NOT convert them to JSON numbers if they may
   exceed 2^53 — that conversion is exactly the rounding this
   recipe exists to avoid.

This walk is scriptable: one sub-spec run per f32 segment, one
`int.mjs` call per int step, chained in step order.

Two honesty notes. First, f32 prefix rounding is inherent: if the
f32 part rounds a value (f32 has a 24-bit mantissa), the int tail
is exact *given* those rounded values — the recipe does not
recover precision the f32 part already lost. Second, `from-f32`
truncation toward zero is defined semantics, not a loss to hide.

## Worked example (live numbers)

Spec `tests/f1-worked.json`: f32 `add` (step 0) -> `intFromFloat`
(steps 1, 2) -> `intAdd` (step 3), output slot 6 (i64). `run.mjs`
rejects it with the split message (f32 steps 0..0, int tail steps
1..3). The walk, from `helper/`:

```
# 1. f32 prefix as its own spec (tests/f1-worked-prefix.json):
node bin/run.mjs tests/f1-worked-prefix.json
# -> outputs [20000004, 30000008]
#    (note: 20000005/30000007 rounded by f32 itself — see honesty note)

# 2. int tail with int.mjs, in step order:
node bin/int.mjs from-f32 20000004,30000008   # -> ["20000004","30000008"]
node bin/int.mjs from-f32 100,200             # -> ["100","200"]
node bin/int.mjs add 20000004,30000008 100,200 # -> ["20000104","30000208"]
```

Final exact answer: `["20000104", "30000208"]`, shape `[1,1,1,2]`.

## Int-only variant (no f32 part)

If the spec has no f32 steps (only ports feeding int steps, e.g.
`tests/f1-gap.json`), skip step 2: feed the port values directly
to the int steps with `int.mjs` in step order. The rejection
message says "There is no f32 part" in this case.

## The intToFloat alternative

If every final |x| <= 16777216 (2^24), you may instead end the
spec with `intToFloat` and run it whole with `run.mjs` — the
values are then exactly representable in f32. Above that bound
`intToFloat` rejects by design (§6.1); that rejection is the
reason the split exists, not a bug in it. `--allow-inexact` on
`int.mjs to-f32` rounds visibly — use it only when rounding is
the intended semantics.
