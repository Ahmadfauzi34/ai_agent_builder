# WasmLinear Wrapper Decomposition Research

Status: research boundary for issue #223; evidence pending.

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

Before adding private instrumentation or changing production code, this research measures the public wrapper controls that can be isolated from an external packaged Rust consumer.

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

## Semantic proof boundary

Timing is accepted only if:

- packaged public `burn-research` surface alone is sufficient;
- deterministic flat weights apply and read back exactly;
- output is finite;
- output remains exactly deterministic after timing;
- weights remain exactly unchanged after timing;
- input shape remains stable;
- `weight_dims()` remains stable and exact.

## Interpretation

Evidence is reviewed after the report is produced:

- `weight_dims()` is material relative to full forward -> investigate immutable cached Linear dimensions while preserving weight/layout/state semantics;
- input clone is material -> investigate tensor ownership/borrow path;
- both controls are small -> residual reshape + Burn Linear kernel path dominates, justifying deeper research-only instrumentation;
- mixed or strongly size-dependent -> repeat at a larger representative policy dimension before optimization.

## Scope

Research-only files:

- `.github/workflows/wasm-linear-wrapper-decomposition-research.yml`;
- `scripts/research_wasm_linear_wrapper_decomposition.py`;
- `docs/wasm-linear-wrapper-decomposition-research.md`.

No production Rust source, Linear implementation, registry behavior, graph validation, slot allocation, ABI v1, Python API, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No custom fused Linear primitive, no Math Program v10, no NumPy/DLPack/PyO3, and no Go/C++ work.
