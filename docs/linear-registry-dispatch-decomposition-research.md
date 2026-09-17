# Linear Registry Dispatch Decomposition Research

Status: research boundary for issue #221; evidence pending.

## Context

PR #220 established `FORWARD_LAYER_CHAIN_DOMINATES`: representative compiled-graph execution spends roughly 94–97% of its median time in the public layer-forward chain, while structural validation is only ~3–5% and fresh slot setup is below 0.5%.

For Linear steps, `LayerRegistry::forward_layer(...)` performs a layer-type dispatch plus `HashMap` lookup and then delegates to `WasmLinear::try_forward(...)`. Before changing registry/runtime structure, this research asks whether that dispatch/lookup contributes materially beyond direct standalone `WasmLinear` execution.

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

## Semantic proof boundary

Timing is accepted only if:

- the packaged public Rust surface alone is sufficient;
- direct and registry-owned layers read back exactly the same flat weights;
- direct and registry outputs are exactly equal;
- outputs are finite and deterministic;
- flat weights remain unchanged after timing;
- the registry layer remains present throughout the measurement.

Timing remains descriptive research evidence only and is never a CI performance SLA.

## Interpretation

Evidence is reviewed after the report is produced:

- registry-mediated execution materially slower than standalone Linear -> investigate registry dispatch/lookup next;
- direct and registry execution near-equal -> Linear tensor/kernel execution is the next frontier;
- mixed or strongly size-dependent -> repeat on larger representative policy dimensions before optimization.

The `layer_exists` control is not assumed to be exactly additive with `forward_layer`; it is only a scale reference for map lookup cost.

## Scope

Research-only files:

- `.github/workflows/linear-registry-dispatch-decomposition-research.yml`;
- `scripts/research_linear_registry_dispatch_decomposition.py`;
- `docs/linear-registry-dispatch-decomposition-research.md`.

No production Rust source, registry cache, graph validation change, slot reuse, ABI v1, Python API, optimizer, ProgramBundle, Math v1-v9, Resolution/Authorization, or support-matrix change.

No custom fused Linear primitive, graph batching primitive, NumPy/DLPack/PyO3, Go, or C++ work.
