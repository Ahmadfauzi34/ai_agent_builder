# Agent Workspace Architecture

`AgentWorkspace` is the canonical agent-side control/working-memory layer for the WASM math coprocessor.

It follows a SQLite-like relational-state concept without embedding SQLite itself. State is represented as bounded in-memory rows, while tensor data and mathematical execution remain outside the workspace.

## Separation of responsibility

```text
Agent / LLM
    |
    v
AgentWorkspace                 control + working memory
    |
    +--> workspaceInit*        stateless orchestration
    +--> workspaceWire*
    +--> workspaceCompile
    |
    v
AgentLayerSpec / AgentGraphBuilder
    |
    v
LayerRegistry                  execution truth
    |
    v
raw protocol -> Burn           tensor/math execution
```

The workspace never owns `LayerRegistry`, `AgentGraphBuilder`, `CompiledGraph`, or `WasmTensor`.

## Workspace state

Internal tables are read-only through the generic mutation API:

- `_slots` — input/free/reserved graph-slot state
- `_layers` — reserved/initialized layer metadata
- `_proofs` — compact proof history
- `_events` — lightweight event history

Agents may create arbitrary custom tables through `put/get/query/remove`, for example `experiments`, `candidates`, `metrics`, `notes`, or `search_frontier`.

## Stateless orchestration

The convenience layer consists only of free functions:

- `workspaceInitUnary(...)`
- `workspaceInitBinary(...)`
- `workspaceWireUnary(...)`
- `workspaceWireBinary(...)`
- `workspaceCompile(...)`

These functions retain no hidden allocator or lifecycle state.

For new layers, reserve the layer ID through the workspace first:

```text
id = workspace.reserveLayerId(registry, label)
spec = AgentLayerSpec.<constructor>(id, ...)
out = workspaceInitUnary(workspace, builder, registry, spec, input, label)
graph = workspaceCompile(builder, registry, out)
```

For a layer initialized manually through a lower-level API, use `workspaceWireUnary` or `workspaceWireBinary` to reconcile the real registry state into workspace metadata before wiring the graph.

## Progressive disclosure

Convenience does not define capability. Every layer remains optional:

```text
AgentWorkspace + stateless helpers   easiest path
        |
AgentLayerSpec + AgentGraphBuilder   typed control
        |
LayerRegistry                        engine control
        |
PacketHeader + raw payload           maximum control
        |
Burn
```

An agent can move down or back up these levels in the same workflow.

## Resource bounds

The workspace is metadata-only and intentionally bounded:

- max rows: 1024
- max table name: 64 bytes
- max key: 128 bytes
- max kind/state: 64 bytes each
- max value: 4096 bytes per row

Use `workspace.limits()` for the runtime-readable limits.

Large tensor payloads belong in `WasmTensor`/`TensorView`, not workspace rows.

## Discovery

- `agentCapabilities()` describes the core math/tensor/typed facade.
- `workspaceCapabilities()` describes the relational workspace, stateless operations, ownership model, and escape hatches.

## Core invariants

- Workspace state is metadata/control state, not execution truth.
- `LayerRegistry` is authoritative for whether a layer actually exists.
- `workspaceInit*` requires an ID reserved by the same workspace before registry initialization.
- Expected validation failures must not consume graph slots or initialize unreserved layers.
- Freeing the workspace must not invalidate registry, graph builder, compiled graph, or tensors.
- Raw protocol and typed lower-level APIs remain available for advanced or novel workflows.
