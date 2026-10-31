# wasm-opt — engine-provided agent tooling

The burn WASM engine **provides** the tooling: measurement comes from the engine
itself, not from outside guesses. An agent gets a tight loop:
**propose → measure → judge**, all backed by the engine — from low-level
optimization profiling to full math verification with provenance.

## CLIs

### Arithmetic

- `node bin/arith.mjs <add|sub|mul|div> <a1,a2,...> <b1,b2,...>`
  Elementwise vector arithmetic through the engine, without
  hand-writing a spec (the tool builds the spec and runs it via the
  wasm-direct path). `div` is mul by the host-computed reciprocal;
  division by zero, length mismatch, unknown op, and non-numeric
  input are clean errors (exit 2). V1 limits: no `matmul` (its
  layout contract is dynamic and unprobed), no int ops (i64 family,
  rejected by the f32 run path).

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
  `node bin/validate.mjs --explain <spec.json>` prints the slot-by-slot map
  (shapes, layouts, params per layer, tensor bytes) for a valid spec, plus
  a `liveness` block: live-before-step sets and peak live bytes (usually
  smaller than the static footprint; a runtime that frees dead outputs
  promptly needs the peak, not the footprint), with a lower-bound flag
  when a slot with unknown bytes is ever live. An invalid spec fails
  with the same JSON as plain validate.
- `node bin/run.mjs <spec.json>` — validate → create → setWeights → bind → run,
  one shot. Prints outputs + execution gate + host provenance.
  `node bin/run.mjs --direct <spec.json>` — same engine, same values,
  in-process (no child process, no IPC): one fresh buildGraph per spec
  inside a `withGraphScope` arena (registry/plan/graph/bundle freed even
  on throw). Faster (single ~3-4x, batch of 10 ~8x on this machine) but
  produces NO execution_gate/host_provenance — those are ingress-session
  artifacts; direct records keep the same keys with null plus
  `mode: 'direct'`, never a fake gate.
- `node bin/verify.mjs <baseline.json> <candidate.json> [cases.json]`
  Baseline-vs-candidate graph verification: numerical equivalence with
  tolerances, the verdict behind `opt_compare`.
  `... --fuzz [N] [--fuzz-seed S] [--fuzz-lo L] [--fuzz-hi H]` generates
  cases in-process (same generator as `fuzz_cases.mjs`); conflicts with
  a cases file. Batch takes the same flags after the pairs file:
  `... --batch <pairs.json> --fuzz ...` generates each pair's cases
  from its own baseline spec; a pair with explicit cases records a
  per-pair conflict instead of aborting the batch.
- `node bin/checkpoint.mjs export <spec.json> | verify <bundle> <sha>`
  Export a program bundle with integrity proof (sha256), or verify a bundle's
  integrity session-free.

### Correctness probes

- `node bin/engine_hardening_test.mjs` — regression tests for the
  agent-hardening helpers (`checkOperation` null guards, `withTensorScope`
  teardown semantics). 9 checks, exit non-zero on failure.
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
  analytical and instant (compositional worst-case); its per-layer term
  uses the EXACT error operator norm ||ΔW||_oo from dequantized weights
  (rigorous, ~6x tighter than the report's bound_W·inDim on outliers).
  Measured mode runs candidate plans and reports the exact delta.
  `node bin/quant_budget.mjs --selftest` runs 9 regression checks
  (exact-norm inequality, rigor bound>=measured, decisions).
- `node bin/diff_ref.mjs <spec.json>` — differential testing: engine output vs
  an independent reference implementation.
- `node bin/cost_model.mjs <spec.json>` — per-op cost table + predicted vs
  measured total time.
- `node bin/shapecheck.mjs <spec.json>` — static shape + dataflow check, no
  engine needed. Fast pre-pass before heavier tools.
- `node bin/fuzz_cases.mjs` — deterministic test-case generation for
  verify/diff_ref. Discrete ports (embedding ids) get integer ids in
  [0, vocab), resolved from the consuming embedding layer.
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
Authoring weights by hand? Write row-major math notation and convert
with `toColumnMajor` (`lib/weights.mjs` — square matrices only; it is
the single place row-major becomes engine layout, never hand-write
engine layout). `affineSpec(W, b, v)` builds a complete `linear`+`add`
spec from row-major `W`. See `references/weights.md`.

## Engine location

Defaults to `<repo>/pkg/` (the `wasm-pack build --target web --out-dir pkg`
output). Override with `BURN_WASM_PKG_DIR`. When this helper is packaged
(`scripts/package_node_host.mjs` copies it to `<pkg>/helper/`), it
automatically resolves the engine next to itself — so a packed bundle carries
the helper and the helper finds its engine.

## Agent pack (single source of truth)

This directory is the repo's agent toolset, and the distributable agent
pack is BUILT from it — never edited as a side copy. The contract agents
read is `AGENT.md` (start there); `INDEX.md` maps questions to tools;
`spec-schema.md` documents the spec format.

- `node bin/find-op.mjs <name>` / `--list` — one layer constructor (50).
- `node bin/surface.mjs <name>` / `--list` — one wasm surface entry (169).
- `node bin/layout.mjs <name>` / `--list` — one constructor's layout contract (50).
- `node bin/regression.mjs` — the permanent 34-case regression gate
  (cases: `reference/regression.md`).
- `node ../scripts/build_agent_pack.mjs --out <dir>` — assemble the
  distributable pack from this directory + the engine bundle, verify it
  (counts, engine sha, smoke run), and refresh `MANIFEST.json` pins.
  `MANIFEST.json` pins the engine files `wasm_direct` integrity-checks;
  it is generated by the builder, not hand-edited — CI fails if it is
  stale after an engine change.
