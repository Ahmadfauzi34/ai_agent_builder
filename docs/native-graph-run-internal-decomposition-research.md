# Native Graph Run Internal Decomposition Research

Status: proven research boundary for issue #219.

Decision: `FORWARD_LAYER_CHAIN_DOMINATES`.

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

This research separates those costs using only the existing public Rust package surface.

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

These buckets are descriptive and are not assumed to be perfectly additive.

## Evidence

### Deep policy: 8 Linear layers, width 32

Median timings:

```text
CompiledGraph::run                  0.228616 ms
manual direct forward chain        0.214916 ms
manual slot-backed chain           0.209910 ms
validate_registry_binding          0.010698 ms
slot vector + input clone          0.001045 ms
```

Relative to full graph execution:

```text
direct forward chain               ~94.0%
slot-backed forward chain          ~91.8%
structural validation              ~4.68%
slot setup                         ~0.46%
```

### Wide policy: 3 Linear layers, width 128

Median timings:

```text
CompiledGraph::run                  0.141386 ms
manual direct forward chain         0.137022 ms
manual slot-backed chain            0.127278 ms
validate_registry_binding           0.004433 ms
slot vector + input clone           0.000451 ms
```

Relative to full graph execution:

```text
direct forward chain               ~96.9%
slot-backed forward chain          ~90.0%
structural validation              ~3.14%
slot setup                         ~0.32%
```

The slot-backed median is slightly below the direct-chain median in both cases. That small inversion is treated as timing/order/cache noise, not evidence that slot bookkeeping is beneficial. Both manual paths demonstrate the same architectural result: public layer-forward execution accounts for the overwhelming majority of full compiled-graph cost.

## Semantic proof boundary

Every case proved:

- packaged public `burn-research` surface only;
- canonical `GraphParameterBinding` parameter ordering/dimension;
- deterministic finite nonzero candidate application;
- exact candidate readback;
- exact output equality across full graph, manual direct chain, and manual slot-backed chain;
- finite output;
- stable program identity;
- stable binding identity;
- unchanged trainable state after repeated execution.

Timing remains descriptive research evidence only and is never a CI SLA.

## Decision

The evidence supports:

```text
FORWARD_LAYER_CHAIN_DOMINATES
KEEP_STRUCTURAL_VALIDATION_ON_EVERY_RUN
DO_NOT_ADD_VALIDATION_CACHE_YET
DO_NOT_ADD_SLOT_REUSE_YET
KEEP_GRAPH_SEMANTICS_UNCHANGED
```

Structural validation is measurable but secondary at roughly 3–5% of the current representative graph cost. Slot allocation plus input clone is below 0.5% in both cases. Neither justifies weakening stale-structure rejection or adding runtime slot caching.

The next evidence slice should therefore decompose the dominant Linear/registry execution path rather than change `CompiledGraph` orchestration.

## Scope

Research-only files:

- `.github/workflows/native-graph-run-internal-decomposition-research.yml`;
- `scripts/research_native_graph_run_internal_decomposition.py`;
- `docs/native-graph-run-internal-decomposition-research.md`.

No production Rust source, ABI v1, Python host/facade API, graph semantics, GraphParameterBinding semantics, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No skipped structural validation, graph batching primitive, persistent pointer/cache, NumPy/DLPack/PyO3, Go, or C++ work.
