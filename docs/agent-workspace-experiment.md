# Agent Workspace In-Memory Experiment

This experiment tests a SQLite-like *concept* without embedding SQLite itself.

The workspace is a relational state substrate, not a math engine and not an ownership boundary.
Burn, `LayerRegistry`, `AgentLayerSpec`, `AgentGraphBuilder`, and the raw protocol remain external and authoritative.

## Generic row model

Each row has:

- `table`
- `key`
- `kind`
- `state`
- `value`

Agents can create arbitrary user tables with `put`, inspect them with `get`/`query`, and remove them with `remove`.
Tables prefixed with `_` are internal and read-only through the generic API.

## Internal tables

- `_slots`: input/free/reserved slot state
- `_layers`: reserved/initialized layer metadata
- `_proofs`: compact proof history
- `_events`: lightweight event history

The workspace stores metadata only. Tensor data stays in Burn/WASM tensor objects.

## External truth

`reserveLayerId(registry, label)` checks both the real registry and workspace reservations.
`syncLayer(registry, spec, label)` records a layer only after the real registry confirms it exists.
`forgetLayer(id)` removes workspace metadata only and never mutates the registry.

## Goal

Compare this data-driven working-memory model with `AgentReferenceSession`.
If adding a new agent workflow can be represented as new rows/tables instead of adding Rust methods, the workspace model has stronger long-term flexibility.
