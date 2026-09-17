# WasmLinear Residual Decomposition Research

Status: evidence complete for issue #227.

## Context

PR #226 removed module-record reconstruction from the hot Linear dimension lookup path by storing immutable `in_dim` / `out_dim` metadata in `WasmLinear`. The existing wrapper decomposition showed `weight_dims()` becoming negligible while input clone also remained small.

The remaining standalone `WasmLinear::forward(...)` path is therefore approximately:

```text
WasmTensor [B, in, 1, 1]
  -> reshape 4D -> 2D [B, in]
  -> Burn Linear::forward
  -> reshape 2D -> 4D [B, out, 1, 1]
  -> WasmTensor
```

This research asks whether the remaining wrapper shape transitions are material or whether useful Burn Linear computation now dominates strongly enough to stop optimizing the wrapper.

## Cases

Two representative square Linear dimensions are measured:

```text
Linear(32 -> 32, bias=true)
Linear(128 -> 128, bias=true)
```

Batch size is fixed at one to stay aligned with the established control-policy research lineage.

## Timing buckets

An external packaged Rust release consumer measures:

1. equivalent `Tensor<WasmBackend, 4>` reshape `[1, dim, 1, 1] -> [1, dim]`;
2. Burn `Linear<WasmBackend>::forward(...)` on an equivalent 2D tensor;
3. equivalent output reshape `[1, dim] -> [1, dim, 1, 1]`;
4. combined analogue reshape-in -> Burn Linear -> reshape-out pipeline;
5. full public `WasmLinear::forward(...)` as the end-to-end reference.

Input tensor clones needed because Burn reshape/forward consume tensors are prepared before each timer starts, so the control timings do not intentionally include clone cost.

The Burn Linear instance is an equivalent-dimension timing analogue rather than private instrumentation of the exact `WasmLinear` internal module. The buckets are therefore descriptive controls and are not assumed perfectly additive.

## Semantic proof boundary

Timing is accepted only if:

- only packaged public `burn-research` plus public Burn APIs are used;
- Burn analogue output is finite and stable before/after timing;
- public `WasmLinear` output is finite and exactly stable;
- public wrapper weights remain exactly unchanged;
- public wrapper dimensions remain exact;
- no timing threshold is used as a CI SLA.

## Evidence

The first successful release-mode run on PR #228 produced:

| case | reshape 4D->2D | Burn Linear | reshape 2D->4D | analogue pipeline | full WasmLinear |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32->32 | 0.000872 ms | 0.022828 ms | 0.000541 ms | 0.023439 ms | 0.025312 ms |
| 128->128 | 0.001202 ms | 0.044554 ms | 0.000712 ms | 0.045270 ms | 0.047409 ms |

Derived descriptive ratios:

```text
32 -> 32
Burn Linear / analogue pipeline      ~= 97.39%
reshape-in + reshape-out / pipeline  ~= 6.03%
analogue pipeline / full wrapper     ~= 92.60%

128 -> 128
Burn Linear / analogue pipeline      ~= 98.42%
reshape-in + reshape-out / pipeline  ~= 4.23%
analogue pipeline / full wrapper     ~= 95.49%
```

Because the individually timed controls are not assumed perfectly additive, percentages can overlap slightly. The important signal is consistent across both dimensions: Burn Linear execution overwhelmingly dominates the equivalent pipeline, while both shape transitions remain small and the analogue pipeline is already close to the public wrapper cost.

All proof checks passed: Burn analogue outputs remained finite/stable, public wrapper outputs remained exact/stable, and wrapper weights/dimensions were unchanged.

## Decision

```text
STOP_LINEAR_WRAPPER_OPTIMIZATION
```

The remaining `WasmLinear` wrapper overhead is sufficiently small relative to useful Burn Linear execution that no further wrapper-specific optimization is justified by current evidence.

In particular:

- do not optimize input cloning further;
- do not add reshape caches or persistent shape-transition state;
- do not add registry/dispatch caches;
- do not introduce fused/custom Linear kernels;
- do not widen Math or ABI surfaces for this path.

Future Linear performance work should only reopen if a larger real agent workload demonstrates a new material bottleneck. Otherwise the remaining cost is treated as useful model computation rather than architectural overhead.

## Scope

Research-only files:

- `.github/workflows/wasm-linear-residual-decomposition-research.yml`;
- `scripts/research_wasm_linear_residual_decomposition.py`;
- `docs/wasm-linear-residual-decomposition-research.md`.

No production Rust source, Linear implementation, registry behavior, graph validation/slot allocation, ABI v1, Python API, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No fused Linear primitive, custom matmul kernel, Math Program v10, NumPy/DLPack/PyO3, or Go/C++ work.
