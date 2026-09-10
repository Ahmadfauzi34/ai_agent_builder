# Agent-facing WASM facade

The agent-facing path is intentionally layered so callers do not need to construct protocol bytes manually:

1. Call `agentCapabilities()` once to discover supported constructors and proof entry points.
2. Create an `AgentLayerSpec` using a typed constructor such as `relu`, `linear`, `conv2d`, `rmsNorm`, `ghost`, or `matmul`.
3. Register the spec with `LayerRegistry.initAgentLayer(spec)`.
4. Build reference execution with `AgentGraphBuilder.addUnary`, `addBinary`, `setOutput`, and `compile`.
5. Run the Burn-backed graph and compare external output with `CompiledGraph.verifyFlat` or `mathVerifyVectors`.

The typed facade is an ergonomics layer only. `LayerRegistry`, the existing binary protocol, the graph compiler/runtime, and Burn remain the sources of truth. Raw protocol APIs stay available for compatibility and advanced use.

Malformed facade configuration should be rejected before backend initialization whenever the facade can validate it deterministically. The final generated Python/Rust/JS tool does not need to ship this WASM runtime unless the product itself chooses to use it at runtime.
