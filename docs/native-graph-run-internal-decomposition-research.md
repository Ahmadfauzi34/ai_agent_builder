# Native Graph Run Internal Decomposition Research

Status: research boundary for issue #219; evidence pending.

## Context

PR #218 established that native `CompiledGraph::run(...)` scaling pressure is driven primarily by increasing Linear-layer depth on the current small control-policy lineage. Width growth and activation composition were secondary, and equal-shape activation differences were modest.

The current `CompiledGraph::run(...)` path performs more than layer execution:

```text
validate_registry_binding_internal(...)
allocate Vec<Option<WasmTensor>> slots
clone input into slot 0
for each compiled step:
    read input slot(s)
    registry.forward_layer(...) / forward_binary_layer(...)
    write output slot
return/take output slot
```

Before changing runtime semantics, this research separates those costs using only the existing public Rust package surface.

## Representative cases

Two graphs from the #218 scaling envelope are used:

- `deep_l8_w32_relu`: 8 Linear layers, width 32, ReLU between non-final Linear layers;
- `wide_l3_w128_relu`: 3 Linear layers, width 128, ReLU between non-final Linear layers.

Both use the same six-feature control input lineage and deterministic finite nonzero trainable state.

## Timing buckets

For each case the external packaged Rust release consumer measures:

1. `graph.validate_registry_binding(&registry)`;
2. allocation of a fresh `Vec<Option<WasmTensor>>` plus clone of the input into slot 0;
3. a manual direct sequential chain through public `registry.forward_layer(...)` with no slot vector;
4. a manual slot-backed sequential chain through public `registry.forward_layer(...)`, mirroring the graph's unary chain but without graph structural validation;
5. full `CompiledGraph::run(...)`.

These buckets are descriptive and are not assumed to be perfectly additive. Ratios are used to determine which part of the execution path deserves the next evidence slice.

## Semantic proof boundary

Before timing is accepted, every case must prove:

- the packaged public `burn-research` surface is sufficient;
- canonical `GraphParameterBinding` parameter ordering/dimension;
- deterministic finite nonzero candidate applies successfully;
- candidate readback is exact;
- full graph, manual direct chain, and manual slot-backed chain produce exactly equal output;
- output is finite;
- program identity remains stable;
- binding identity remains stable;
- trainable state is unchanged by repeated execution.

Timing remains descriptive research evidence only and is never a CI SLA.

## Interpretation

Evidence is reviewed after the report is produced:

- structural validation material -> investigate validation proof/caching strategy while preserving stale-structure rejection;
- slot setup / slot-backed overhead material -> investigate allocation/reuse strategy without changing graph semantics;
- manual forward chain dominates -> layer/registry execution is the next frontier;
- mixed/no clear pressure -> preserve runtime and move to a larger representative closed-loop policy before optimization.

## Scope

Research-only files:

- `.github/workflows/native-graph-run-internal-decomposition-research.yml`;
- `scripts/research_native_graph_run_internal_decomposition.py`;
- `docs/native-graph-run-internal-decomposition-research.md`.

No production Rust source, ABI v1, Python host/facade API, graph semantics, GraphParameterBinding semantics, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No skipped structural validation, graph batching primitive, persistent pointer/cache, NumPy/DLPack/PyO3, Go, or C++ work.
