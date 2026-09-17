# Native Graph Runtime Scaling Research

Status: research boundary for issue #217; evidence pending.

## Context

PR #216 established that native `CompiledGraph::run(...)` accounts for roughly 77–78% of the raw `br_v1_graph_run` path on the established `Linear(6 -> 2, bias=true)` control policy. FFI/boundary residual is material but secondary, and output boxing itself is negligible.

The next question is therefore not whether to widen the ABI, but how native graph execution scales as policy graphs become deeper, wider, and compositionally richer.

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

The report includes:

- graph step count;
- trainable parameter count;
- hidden width and Linear depth;
- activation composition;
- median/min/p90/max native run time;
- median microseconds per graph step;
- median nanoseconds per trainable parameter;
- derived depth, width, and activation-spread ratios.

## Semantic proof boundary

Every case must prove:

- public packaged Rust surface only;
- `GraphParameterBinding` defines canonical trainable order/dimension;
- deterministic finite nonzero candidate applies successfully;
- exact candidate readback before timing;
- finite deterministic graph output;
- program identity remains stable;
- binding identity remains stable;
- trainable state is unchanged by repeated graph execution.

## Decision boundary

The research does **not** automatically select an optimization. Evidence is reviewed after the report is produced.

Interpretation:

- depth pressure clearly dominates -> inspect graph step dispatch / per-step runtime overhead;
- width pressure clearly dominates -> inspect tensor/layer kernel execution;
- activation composition clearly dominates at equal shape/step count -> inspect the specific activation implementation;
- mixed/no clear pressure -> move to a larger representative closed-loop policy before changing runtime.

## Scope

Research-only files:

- `.github/workflows/native-graph-runtime-scaling-research.yml`;
- `scripts/research_native_graph_runtime_scaling.py`;
- `docs/native-graph-runtime-scaling-research.md`.

No production Rust source, ABI v1, Python host/facade API, graph/tensor runtime, GraphParameterBinding semantics, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No graph batching primitive, persistent pointer/cache, NumPy/DLPack/PyO3, Go, or C++ work.
