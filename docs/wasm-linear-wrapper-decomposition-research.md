# WasmLinear Wrapper Decomposition Research

Status: evidence recorded; `WEIGHT_DIMS_RECORD_ACCESS_MATERIAL_AT_WIDER_LINEAR`.

## Context

PR #222 established `LINEAR_EXECUTION_DOMINATES`: registry-mediated Linear execution is effectively equal to standalone `WasmLinear::forward(...)` for both 32→32 and 128→128 cases, while lookup-only cost is negligible. Registry dispatch/cache work therefore has no current evidence basis.

The current standalone Linear path includes wrapper work before and after the Burn Linear kernel:

```text
clone input tensor
read input shape
validate singleton spatial axes
weight_dims() -> clone module record / inspect weight dims
validate feature axis
reshape 4D -> 2D
Burn Linear::forward
reshape 2D -> 4D
wrap WasmTensor
```

This research measures the public wrapper controls that can be isolated from an external packaged Rust consumer before any production change.

## Cases

Two square hidden-layer dimensions from the established control-policy lineage are used:

```text
Linear(32 -> 32, bias=true)
Linear(128 -> 128, bias=true)
```

Each case receives deterministic finite nonzero flat weights and deterministic finite input.

## Timing buckets

For each case the external packaged Rust release consumer measures:

1. `WasmTensor::clone()` on the exact input tensor;
2. `WasmLinear::weight_dims()` as the record/dimension-access control;
3. full standalone `WasmLinear::forward(...)`.

Measurement order rotates every repetition to reduce fixed ordering and thermal drift bias.

The controls are descriptive and are not assumed to be perfectly additive. No timing threshold is used as a CI SLA.

## Evidence

The first green research run produced:

| case | full forward median | input clone median | clone share | `weight_dims()` median | `weight_dims()` share |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32→32 | 0.0249715 ms | 0.0002270 ms | 0.91% | 0.0012065 ms | 4.83% |
| 128→128 | 0.0525185 ms | 0.0002140 ms | 0.41% | 0.0056975 ms | 10.85% |

The semantic proof remained green for both cases:

- packaged public Rust surface only;
- finite deterministic output;
- exact stable flat weights;
- stable input shape;
- stable exact `weight_dims()`.

## Interpretation

The evidence does **not** support tensor-clone optimization: the clone control remains below 1% of full forward in both measured cases.

`weight_dims()` is different. It grows from roughly 4.8% of full forward at 32→32 to roughly 10.9% at 128→128. The current implementation obtains that value by cloning the module record solely to inspect immutable weight dimensions. That is large enough to justify one narrow product slice before deeper kernel instrumentation.

Decision:

```text
WEIGHT_DIMS_RECORD_ACCESS_MATERIAL_AT_WIDER_LINEAR
KEEP_INPUT_CLONE_PATH_UNCHANGED
DO_NOT_CHANGE_REGISTRY_DISPATCH
NEXT: STORE_IMMUTABLE_LINEAR_DIMS_AND_AVOID_RECORD_CLONE_ON_FORWARD
```

The next product change should preserve the existing proof boundary:

- constructor establishes immutable `in_dim` / `out_dim` metadata;
- `weight_dims()` returns those immutable dimensions without reconstructing a module record;
- forward feature validation uses the same immutable `in_dim`;
- `load_state()` continues rejecting shape changes;
- flat weight layout/order and `set_weights_flat()` semantics remain unchanged;
- graph, registry, ABI, Python, optimizer, checkpoint, and Math semantics remain unchanged.

After that narrow change, rerun this research workload before considering deeper reshape/Burn-kernel instrumentation.

## Semantic proof boundary

Timing is accepted only if:

- packaged public `burn-research` surface alone is sufficient;
- deterministic flat weights apply and read back exactly;
- output is finite;
- output remains exactly deterministic after timing;
- weights remain exactly unchanged after timing;
- input shape remains stable;
- `weight_dims()` remains stable and exact.

## Scope

Research-only files:

- `.github/workflows/wasm-linear-wrapper-decomposition-research.yml`;
- `scripts/research_wasm_linear_wrapper_decomposition.py`;
- `docs/wasm-linear-wrapper-decomposition-research.md`.

No production Rust source, Linear implementation, registry behavior, graph validation, slot allocation, ABI v1, Python API, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No custom fused Linear primitive, no Math Program v10, no NumPy/DLPack/PyO3, and no Go/C++ work.
