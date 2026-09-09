# AgentReferenceSession

`AgentReferenceSession` is an optional convenience orchestrator. It does **not** own `LayerRegistry` or `AgentGraphBuilder`; callers keep both core objects and may drop to the typed facade or raw protocol at any time.

## Convenience path

```js
const registry = new LayerRegistry();
const builder = new AgentGraphBuilder(8);
const session = new AgentReferenceSession(8);

const reluId = session.reserveLayerId(registry);
const relu = AgentLayerSpec.relu(reluId);
const y = session.initUnary(builder, registry, relu, session.inputSlot());

const graph = session.compile(builder, registry, y);
```

## Mixing manual and session-managed work

A caller may initialize a layer manually and only ask the session to allocate/wire the next slot:

```js
const manual = AgentLayerSpec.relu(77);
registry.initAgentLayer(manual);
const y = session.wireUnary(builder, manual, session.inputSlot());
```

If manual graph work consumes slots or a caller wants deterministic IDs, synchronize the allocator explicitly with `setNextSlot(...)` and `setNextLayerId(...)`.

## Design rule

The session provides allocation and orchestration only. Capability remains defined by the lower layers:

`AgentReferenceSession -> AgentGraphBuilder / AgentLayerSpec -> raw protocol / registry -> Burn`

The lower APIs remain available and unchanged.
