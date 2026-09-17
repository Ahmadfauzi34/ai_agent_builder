# Native Graph Run Decomposition Research

Status: research boundary for issue #215.

## Context

PR #214 established that the raw `br_v1_graph_run` call through output-handle creation accounts for about 81.8% of the supported typed Python policy step on the established `Linear(6 -> 2, bias=true)` control workload. Output copying is only about 4.8% of that typed full path.

The remaining dominant bucket still combines:

```text
Python/CFFI call
ffi_status / catch_unwind
handle distinctness + type lookup
CompiledGraph::run(...)
BrV1Handle output wrapping/allocation
```

Before modifying the FFI implementation or graph runtime, this slice uses a same-runner shadow decomposition that requires **no production source change and no ABI widening**.

## Method

One research workflow runs two independent consumers on the same GitHub runner.

### 1. Direct Rust public-core consumer

A temporary external Cargo project depends on the public root crate and builds the exact same policy:

```text
Linear(6 -> 2, bias=true)
layer id = 215000
parameter dim = 14
canonical state = all zero f32
input = [0.7, -0.2, -0.4, 0.1, 0.0, 0.0]
```

It measures in release mode:

- `CompiledGraph::run(...)`, with output drop outside the timed interval;
- `Box<WasmTensor>` allocation/drop using precomputed real outputs;
- `CompiledGraph::run(...) + Box<WasmTensor>` with drop outside the timed interval.

### 2. Installed-wheel raw ABI consumer

The current CFFI wheel is built and installed into a fresh external virtual environment on that same runner. A raw public `ffi` / `lib` consumer builds the same graph/state/input and measures:

- `br_v1_graph_run(...)` through output `BrV1Handle` creation, with handle free outside the timed interval.

## Cross-boundary proof

Timing evidence is accepted only if direct Rust and raw ABI consumers have exact:

- program identity;
- binding identity;
- reference output values.

Both sides must also prove finite deterministic output and stable identities after repeated execution. Raw output handles must be deterministically freed.

## Derived decomposition

The report computes:

```text
native_run_share
    = direct CompiledGraph::run / raw br_v1_graph_run

native_run_plus_box_share
    = direct (CompiledGraph::run + Box<WasmTensor>) / raw br_v1_graph_run

boundary_residual
    = max(0, raw br_v1_graph_run - native run+box)
```

The residual intentionally groups:

```text
Python/CFFI call overhead
+ ffi_status/catch_unwind
+ ensure_distinct
+ typed handle lookup
+ BrV1Handle-specific wrapping differences
```

It is **not** presented as exact attribution to any one of those operations.

`Box<WasmTensor>` is a concrete-output allocation control. It is not claimed to be a byte-for-byte model of `Box<BrV1Handle>`; it is used only to avoid blaming ordinary owned-output allocation on the residual without evidence.

## Descriptive decision rule

Timing is research evidence only and never a CI SLA.

```text
native run+box / raw FFI >= 70%
    -> COMPILED_GRAPH_RUN_DOMINATES

otherwise residual / raw FFI >= 25%
    -> FFI_BOUNDARY_RESIDUAL_MATERIAL

otherwise
    -> MIXED_NATIVE_GRAPH_RUN_COST
```

### If `COMPILED_GRAPH_RUN_DOMINATES`

Investigate graph/runtime execution against representative graph shapes before changing the ABI. The next slice should test whether the micro-policy result scales with graph depth/width and layer composition.

### If `FFI_BOUNDARY_RESIDUAL_MATERIAL`

Only then is deeper private/cfg(test) instrumentation inside the existing FFI boundary justified to split validation/lookup from handle wrapping and `ffi_status` overhead.

### If mixed

Keep the implementation unchanged until a larger representative agent workload yields a clearer signal.

## Scope

Research-only files:

- `.github/workflows/native-graph-run-decomposition-research.yml`;
- `scripts/research_native_graph_run_decomposition.py`;
- `docs/native-graph-run-decomposition-research.md`.

No changes to production Rust source, `burn-research.ffi.v1`, Python host/facade API, graph runtime, tensor runtime, optimizer, ProgramBundle, Math Program v1-v9, Resolution/Authorization, or support matrix.

No new ABI symbol, graph batching primitive, persistent pointer/cache, NumPy, DLPack, PyO3, Go, or C++ work is introduced.
