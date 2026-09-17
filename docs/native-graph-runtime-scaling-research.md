# Native Graph Runtime Scaling Research

Status: evidence proven on PR #218; decision recorded.

## Context

PR #216 established that native `CompiledGraph::run(...)` accounts for roughly 77–78% of the raw `br_v1_graph_run` path on the established `Linear(6 -> 2, bias=true)` control policy. FFI/boundary residual is material but secondary, and output boxing itself is negligible.

This slice asks how native graph execution scales as policy graphs become deeper, wider, and compositionally richer before any runtime optimization is selected.

## Research method

The workflow packages the current `burn-research` crate and creates a temporary external Rust consumer. The consumer uses only the public packaged Rust surface and runs in release mode.

All cases use the same six-feature control input lineage from #210/#212/#214/#216:

```text
[0.7, -0.2, -0.4, 0.1, 0.0, 0.0]
```

Before timing, every graph receives a deterministic finite nonzero candidate through `GraphParameterBinding`. Candidate application and exact readback are required.

### Depth axis

Hidden width remains 32 while sequential Linear depth grows:

```text
1, 2, 4, 8 Linear layers
```

ReLU is inserted between non-final Linear layers.

### Width axis

Depth remains three Linear layers with ReLU between hidden layers while hidden width grows:

```text
8, 16, 32, 64, 128
```

### Composition axis

Depth and width remain fixed at three Linear layers / width 32. The research compares:

```text
Linear-only
Linear + ReLU
Linear + Tanh
Linear + GELU
```

ReLU/Tanh/GELU cases have equal dimensions and equal graph-step count, so their spread is the clean activation-composition comparison. Linear-only is retained as a lower-step structural baseline and is not treated as an equal-step activation comparison.

## Timing boundary

Only `CompiledGraph::run(...)` is inside the timed interval. Input construction, parameter application, output materialization, and output drop are outside the interval.

Each case performs warmup runs followed by repeated timed runs. Timing is descriptive evidence only and is never a CI SLA.

## Proven evidence

Native Graph Runtime Scaling Research run #1 completed successfully on the initial PR head. All semantic proofs passed: exact candidate readback, finite deterministic output, stable program identity, stable binding identity, unchanged trainable state, and public packaged Rust surface only.

### Depth

| Case | Graph steps | Params | Median run |
| --- | ---: | ---: | ---: |
| 1 Linear | 1 | 14 | 0.026004 ms |
| 2 Linear + ReLU | 3 | 290 | 0.062861 ms |
| 4 Linear + ReLU | 7 | 2,402 | 0.1137285 ms |
| 8 Linear + ReLU | 15 | 6,626 | 0.238920 ms |

The 8-Linear case is about **9.19×** the 1-Linear case. Median time per graph step decreases from about **26.0 µs** to **15.9 µs**, so the result does not show superlinear depth collapse.

### Width

At fixed three-Linear depth / five graph steps:

| Hidden width | Params | Median run |
| --- | ---: | ---: |
| 8 | 146 | 0.069497 ms |
| 16 | 418 | 0.0767385 ms |
| 32 | 1,346 | 0.0911485 ms |
| 64 | 4,738 | 0.100305 ms |
| 128 | 17,666 | 0.123327 ms |

Width 128 is only about **1.77×** width 8 even though trainable parameter count grows by roughly 121×. At these small policy shapes, arithmetic volume / parameter count is therefore not the dominant scaling signal.

### Composition

At fixed three-Linear depth / hidden width 32:

| Composition | Graph steps | Median run |
| --- | ---: | ---: |
| Linear-only | 3 | 0.0775265 ms |
| Linear + ReLU | 5 | 0.0796805 ms |
| Linear + Tanh | 5 | 0.084746 ms |
| Linear + GELU | 5 | 0.105964 ms |

Among equal-step activation graphs, maximum/minimum spread is about **1.33×**. GELU is the most expensive of this set, but composition pressure remains much smaller than the depth/Linear-layer-count pressure.

A particularly important result is that adding two ReLU steps changes the three-Linear graph from about 0.0775 ms to only about 0.0797 ms (~1.03×). Therefore the depth result must **not** be interpreted as generic graph-step dispatch cost. The stronger signal is repeated Linear-layer execution / its surrounding per-layer runtime overhead.

## Decision

```text
SMALL_POLICY_LINEAR_LAYER_COUNT_DOMINATES
```

Interpretation:

- depth / number of Linear executions is the clearest scaling pressure;
- width growth is comparatively mild over 8→128 hidden units;
- ReLU/Tanh activation overhead is secondary;
- GELU is measurably heavier but does not dominate the policy runtime;
- no evidence supports ABI widening, output-copy work, batching primitives, or parameter-count-driven optimization here.

The next evidence slice should decompose native graph execution at the **per-layer / per-step level**, with emphasis on Linear execution versus graph bookkeeping (slot access, dispatch, tensor movement/cloning, and layer invocation). It should remain research-only until one internal component is shown to dominate.

## Semantic proof boundary

Every case proved:

- public packaged Rust surface only;
- `GraphParameterBinding` defines canonical trainable order/dimension;
- deterministic finite nonzero candidate applies successfully;
- exact candidate readback before and after timing;
- finite deterministic graph output;
- program identity remains stable;
- binding identity remains stable;
- trainable state is unchanged by repeated graph execution.

## Scope

Research-only files:

- `.github/workflows/native-graph-runtime-scaling-research.yml`;
- `scripts/research_native_graph_runtime_scaling.py`;
- `docs/native-graph-runtime-scaling-research.md`.

No production Rust source, ABI v1, Python host/facade API, graph/tensor runtime, GraphParameterBinding semantics, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No graph batching primitive, persistent pointer/cache, NumPy/DLPack/PyO3, Go, or C++ work.
