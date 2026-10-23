# wasm-opt — engine-provided agent tooling

The burn WASM engine **provides** the tooling: measurement comes from the engine
itself, not from outside guesses. An agent gets a tight loop:
**propose → measure → judge**, all backed by the engine — from low-level
optimization profiling to full math verification with provenance.

## CLIs

### Optimization (Fase A oracle)

- `node bin/opt_profile.mjs <spec.json> [iterations] [--trace]`
  Per-phase wall time + linear-memory page deltas for build/bind/run
  (1 cold + steady iterations), steady-state median/min/max, `leak_free` check.
  `--trace` adds the per-step map via the engine's own `runWithTrace`: each
  step traced individually (median of 3 runs) reporting `wall_ms` + `out_bytes`,
  plus a `hotspots` summary (slowest step, biggest step). `wall_ms` is traced
  wall time (includes capture overhead) — use for relative comparison between
  steps. The `engine` section also reports `wasm_bytes` (bundle-size metric).
  JSON envelope on stdout; exit 0/2.
- `node bin/opt_guard.mjs <file> | --bytes N` — probe the engine's tiered-limit
  guard (complaint #14): would a state-record payload of this size be accepted,
  and under which bincode byte limit tier? Only the input length is inspected,
  so probing is always safe. Exit 0 when accepted, 2 when rejected.
- `node bin/opt_compare.mjs <baseline.json> <candidate.json> [cases.json] [iterations]`
  The design loop: numerical equivalence via `MultiInputVerificationCases` plus
  profiled time/memory delta for both. Verdict is numbers: `ACCEPT`
  (equivalent + faster), `EQUIVALENT but slower`, or `REJECT` (not equivalent).
  Exit 0 iff equivalent.

### Core verification (the math boundary)

- `node bin/validate.mjs <spec.json>` — session-free plan check: validates the
  spec structure (ports, steps, dataflow) without creating a session. Exit 0
  when valid, 2 with structured remediation hints when not.
- `node bin/run.mjs <spec.json>` — validate → create → setWeights → bind → run,
  one shot. Prints outputs + execution gate + host provenance.
- `node bin/verify.mjs <baseline.json> <candidate.json> [cases.json]`
  Baseline-vs-candidate graph verification: numerical equivalence with
  tolerances, the verdict behind `opt_compare`.
- `node bin/checkpoint.mjs export <spec.json> | verify <bundle> <sha>`
  Export a program bundle with integrity proof (sha256), or verify a bundle's
  integrity session-free.

### Correctness probes

- `node bin/gradcheck.mjs <spec.json>` — numeric gradient (finite-difference)
  Jacobian of a spec, optionally verified against an agent-supplied analytic
  Jacobian.
- `node bin/soak.mjs <spec.json> [N=100]` — long-run determinism proof: N runs
  with fresh input bundles, all outputs bit-identical (sha256).
- `node bin/quant_verify.mjs <spec.json>` — weight-quantization simulation
  (int8/fp16) + verification of error bounds.
- `node bin/quant_equiv.mjs --in 4 --out 8 [--bias] [--scheme q8-per-block]
  [--tol auto|<num>]` — verify the engine's REAL quantized state path
  (getQuantizedState -> loadState -> forward) with ACCEPT/REJECT/MEASURED
  verdicts; --tol auto derives the analytical tolerance from the report.
- `node bin/quant_budget.mjs <spec.json> <cases.json> --tol <num>
  [--scheme q8-per-block] [--mode bound|measured]` — per-layer quantization
  budget planner: which linear layers to quantize (int8) to maximize
  checkpoint savings within an output-perturbation budget. Bound mode is
  analytical and instant (compositional worst-case); measured mode runs
  candidate plans and reports the exact delta.
- `node bin/diff_ref.mjs <spec.json>` — differential testing: engine output vs
  an independent reference implementation.
- `node bin/cost_model.mjs <spec.json>` — per-op cost table + predicted vs
  measured total time.
- `node bin/shapecheck.mjs <spec.json>` — static shape + dataflow check, no
  engine needed. Fast pre-pass before heavier tools.
- `node bin/fuzz_cases.mjs` — deterministic test-case generation for
  verify/diff_ref.
- `node bin/prng_check.mjs` — certify and re-verify deterministic PRNG streams.

### Graph rewrites (proof-backed)

- `node bin/rewrite.mjs list | apply <name> <spec.json>` — apply a cataloged,
  proof-backed graph rewrite (see `references/rewrites.v1.json`). A transform
  without a recorded proof is not in the catalog.

### Domain packs

- `node bin/weight_lab.mjs` — the manual-weighting workbench: build/analyze
  weight matrices (spectral radius, orthonormality), gate candidates, certify
  into `references/weight_repertoire.v1.json`. See `references/weights.md`.
- `node bin/geom_build.mjs <chain.json>` — 3D transform chain → engine spec.
- `node bin/geom_batch.mjs` — batched 3D model → 2D screen projection.
- `node bin/geom_project.mjs` — 3D model → 2D screen projection, end to end.
  See `references/geom3.md`.

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

Weights (optional, column-major) are supported for `linear`, all `*Norm`
(`batchNorm`, `groupNorm`, `instanceNorm`, `layerNorm`, `rmsNorm`), all conv
(`conv1d`, `conv2d`, `convTranspose2d`), and `embedding` — the same set the
engine's `setWeightsFlat` accepts. Keys are layer indices into `layers`.

## Engine location

Defaults to `<repo>/pkg/` (the `wasm-pack build --target web --out-dir pkg`
output). Override with `BURN_WASM_PKG_DIR`. When this helper is packaged
(`scripts/package_node_host.mjs` copies it to `<pkg>/helper/`), it
automatically resolves the engine next to itself — so a packed bundle carries
the helper and the helper finds its engine.
