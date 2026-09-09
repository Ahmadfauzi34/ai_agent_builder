# Stateless Workspace Orchestration Experiment

This experiment keeps `AgentWorkspace` as the only convenience-state owner.

The exported helpers are free functions:

- `workspaceWireUnary(...)`
- `workspaceWireBinary(...)`
- `workspaceInitUnary(...)`
- `workspaceInitBinary(...)`
- `workspaceCompile(...)`

They do not allocate or retain any hidden session state.

## Intended flow

```text
AgentWorkspace.reserveLayerId(registry, label)
        |
        v
AgentLayerSpec::<constructor>(reserved_id)
        |
        v
workspaceInitUnary / workspaceInitBinary
        |
        +--> AgentWorkspace.reserveSlot(...)
        +--> LayerRegistry.initAgentLayer(...)
        +--> AgentWorkspace.syncLayer(...)
        +--> AgentGraphBuilder.addUnary/addBinary(...)
```

`workspaceInit*` requires the layer id to have been reserved through the same workspace first. This keeps allocator truth in one place and prevents helper-local state from drifting away from workspace metadata.

For layers initialized manually through lower-level APIs, `workspaceWire*` reconciles the real registry into workspace metadata before adding graph wiring.

## Flexibility

The helper layer is optional. Callers can freely mix:

```text
workspace helper -> typed API -> raw registry/graph API -> workspace helper
```

There is no helper object to synchronize or preserve.

## Failure boundary

Expected validation happens before registry mutation where possible. Output slots reserved by the workspace are released again when a pre-registry or graph-wiring failure is recoverable.

The existing `AgentReferenceSession` remains unchanged for compatibility/comparison. This experiment does not remove it or make it the source of truth.
