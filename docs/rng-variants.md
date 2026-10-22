# RNG variants (#14)

The engine ships three deterministic PRNGs. All three expose the same
sampling surface (`next_u32`, `uniform` in [0,1), `gaussian` N(0,1) via
Box–Muller), so strategies and agents can swap them without changing
any other code.

## The variants

| kind | name | design | state | when to use |
|------|------|--------|-------|-------------|
| 0 | mulberry32 | 32-bit scrambler | 32 bit | default; legacy streams must stay byte-identical |
| 1 | pcg32 | PCG-XSH-RR 64/32 (O'Neill 2014) | 64 bit | long sequential runs; far better statistical quality than mulberry32 |
| 2 | philox4x32 | counter-based (Salmon et al. 2011, Random123) | 128-bit counter + 64-bit key | parallel / seekable streams |

## Selecting one

```js
// legacy: always mulberry32
const opt = new EsOptimizer(dim, 0, seed, pop, sigma, lr);

// explicit: kind 0=mulberry32, 1=pcg32, 2=philox4x32 — strict, never silent
const opt2 = EsOptimizer.withRng(dim, 0, seed, 2, pop, sigma, lr);
```

`withRng` rejects an unknown kind with an error naming the offender
(`unknown rng_kind 7; want 0=mulberry32, 1=pcg32, 2=philox`). It also
runs the same strict hyperparameter validation as `EsOptimizer.strict`
— no silent clamping, no strategy fallback.

## Philox seeking (the point of #14)

Philox has no sequential state: output = F(key, counter). Two
different counters give two independent, reproducible streams with no
coordination — the foundation for parallel ES rollouts:

```rust
let mut a = Philox4x32::new(seed); a.seek(0);
let mut b = Philox4x32::new(seed); a.seek(1); // independent stream
```

`AnyRng::philox_seek` exposes this through the enum; it returns `Err`
(not a silent no-op) for the other two variants.

## Determinism contract

Same seed + same kind ⇒ identical stream on host and wasm. The
mulberry32 stream is pinned by a regression test and must never change:
existing deterministic runs reproduce byte-identically. PCG and Philox
are pinned against published test vectors (pcg-c-basic `(42,54)` and
the Random123 reference values), cross-checked with an independent
Python implementation of both specs.
