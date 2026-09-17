# Linear Registry Dispatch Decomposition Research

Status: proven research boundary for issue #221.

Decision: `LINEAR_EXECUTION_DOMINATES`.

## Context

PR #220 established `FORWARD_LAYER_CHAIN_DOMINATES`: representative compiled-graph execution spends roughly 94–97% of its median time in the public layer-forward chain, while structural validation is only ~3–5% and fresh slot setup is below 0.5%.

For Linear steps, `LayerRegistry::forward_layer(...)` performs a layer-type dispatch plus `HashMap` lookup and then delegates to `WasmLinear::try_forward(...)`. This research measures whether that dispatch/lookup contributes materially beyond direct standalone `WasmLinear` execution.

## Cases

The external packaged Rust release consumer compares two square hidden-layer shapes from the established policy lineage:

```text
32 -> 32, bias=true
128 -> 128, bias=true
```

For each case it constructs:

- a standalone public `WasmLinear`;
- a Linear owned by `LayerRegistry` and initialized through public `AgentLayerSpec`.

Both receive exactly the same deterministic finite flat weights and the same deterministic finite input tensor.

## Timing buckets

For each case the consumer measures:

1. standalone `WasmLinear::forward(...)`;
2. `LayerRegistry::forward_layer(..., LAYER_LINEAR, ...)`;
3. `LayerRegistry::layer_exists(...)` as a lookup-only control.

Direct and registry timing order alternates on each repetition to reduce fixed ordering/drift bias.

## Evidence

### Linear 32 -> 32

Median timings:

```text
standalone WasmLinear::forward      0.0264845 ms
registry forward_layer              0.0268660 ms
registry layer_exists control       0.0001010 ms
```

Derived:

```text
registry / direct ratio             1.0144x
registry excess                     0.0003815 ms
registry excess share               ~1.42%
lookup control / registry forward   ~0.38%
```

### Linear 128 -> 128

Median timings:

```text
standalone WasmLinear::forward      0.1410440 ms
registry forward_layer              0.1411640 ms
registry layer_exists control       0.0001600 ms
```

Derived:

```text
registry / direct ratio             1.00085x
registry excess                     0.0001200 ms
registry excess share               ~0.085%
lookup control / registry forward   ~0.11%
```

The larger representative Linear makes the interpretation especially clear: registry-mediated execution is effectively equal to standalone Linear execution at the scale relevant to the current policy graph. The small 32x32 delta is measurable but still tiny relative to the layer execution itself and does not justify registry caching or a specialized dispatch path.

## Semantic proof boundary

Every case proved:

- packaged public Rust surface only;
- direct and registry-owned layers read back exactly identical flat weights;
- direct and registry outputs are exact-equal;
- outputs are finite and deterministic;
- flat weights remain unchanged after timing;
- registry layer remains present throughout measurement.

Timing remains descriptive research evidence only and is never a CI performance SLA.

## Decision

The evidence supports:

```text
LINEAR_EXECUTION_DOMINATES
KEEP_REGISTRY_DISPATCH_UNCHANGED
DO_NOT_ADD_REGISTRY_CACHE
DO_NOT_SPECIAL_CASE_LINEAR_DISPATCH
KEEP_GRAPH_AND_REGISTRY_SEMANTICS_UNCHANGED
```

The next evidence slice should decompose the standalone `WasmLinear` path itself. In the current implementation that path includes input tensor clone, shape validation, `weight_dims()` retrieval, 4D->2D reshape, Burn `Linear::forward`, and 2D->4D reshape. Public controls can first measure clone and `weight_dims()` overhead before any deeper instrumentation is justified.

The `layer_exists` control is not assumed to be exactly additive with `forward_layer`; it is only a scale reference for map lookup cost.

## Scope

Research-only files:

- `.github/workflows/linear-registry-dispatch-decomposition-research.yml`;
- `scripts/research_linear_registry_dispatch_decomposition.py`;
- `docs/linear-registry-dispatch-decomposition-research.md`.

No production Rust source, registry cache, graph validation change, slot reuse, ABI v1, Python API, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No custom fused Linear primitive, graph batching primitive, NumPy/DLPack/PyO3, Go, or C++ work.
