# wasm-opt — engine-provided low-level optimization helper

The burn WASM engine **provides** the helper: measurement comes from the engine
itself, not from outside guesses. An agent designing a low-level optimization
gets a tight loop: **propose → measure → judge**, all backed by the engine.

## CLIs

- `node bin/opt_profile.mjs <spec.json> [iterations] [--trace]`
  Per-phase wall time + linear-memory page deltas for build/bind/run
  (1 cold + steady iterations), steady-state median/min/max, `leak_free` check.
  `--trace` adds the per-step memory map (which step holds the bytes) via the
  engine's own `runWithTrace`. The `engine` section also reports `wasm_bytes`
  (bundle-size metric). JSON envelope on stdout; exit 0/2.
- `node bin/opt_compare.mjs <baseline.json> <candidate.json> [cases.json] [iterations]`
  The design loop: numerical equivalence via `MultiInputVerificationCases` plus
  profiled time/memory delta for both. Verdict is numbers: `ACCEPT`
  (equivalent + faster), `EQUIVALENT but slower`, or `REJECT` (not equivalent).
  Exit 0 iff equivalent.

## How it measures

Before the engine instantiates, `WebAssembly.Instance` is patched to capture
the exported linear memory (the Fase A memory-oracle technique). WASM linear
memory only grows, so page counts are an exact, noise-free allocator signal —
unlike host RSS.

## Spec format

```jsonc
{
  "numSlots": 6,
  "layers": [
    {"constructor": "linear", "args": [0, 2, 2, false]},
    {"constructor": "relu",   "args": [1]}
  ],
  "steps": [
    {"kind": "unary", "layer": 0, "slots": [0, 2]},
    {"kind": "unary", "layer": 1, "slots": [2, 5]}
  ],
  "outputSlot": 5,
  "ports": [
    {"slot": 0, "role": "observation", "shape": [1, 2, 1, 1],
     "layout": "feature_axis1_singleton"}
  ],
  "weights": {"0": [1, 0.5, -0.25, 2]},   // optional, column-major
  "inputs": {"0": [1.0, -2.0]}             // optional
}
```

Port shapes are 4-D (`[1, 2, 1, 1]` style). Constructors: `linear`, `relu`,
`add` (also `sigmoid`, `tanh`, `mul`, … — anything `AgentLayerSpec` exposes).

## Engine location

Defaults to `<repo>/pkg/` (the `wasm-pack build --target web --out-dir pkg`
output). Override with `BURN_WASM_PKG_DIR`. When this helper is packaged
(`scripts/package_node_host.mjs` copies it to `<pkg>/helper/`), it
automatically resolves the engine next to itself — so a packed bundle carries
the helper and the helper finds its engine.
