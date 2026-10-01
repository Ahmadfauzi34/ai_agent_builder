# Agent-facing WASM facade

The agent-facing path is intentionally layered so callers do not need to construct protocol bytes manually:

1. Call `agentCapabilities()` once to discover supported constructors and proof entry points.
2. Create an `AgentLayerSpec` using a typed constructor such as `relu`, `linear`, `conv2d`, `rmsNorm`, `ghost`, or `matmul`.
3. Register the spec with `LayerRegistry.initAgentLayer(spec)`.
4. Build reference execution with `AgentGraphBuilder.addUnary`, `addBinary`, `setOutput`, and `compile`.
5. Run the Burn-backed graph and compare external output with `CompiledGraph.verifyFlat` or `mathVerifyVectors`.

The typed facade is an ergonomics layer only. `LayerRegistry`, the existing binary protocol, the graph compiler/runtime, and Burn remain the sources of truth. Raw protocol APIs stay available for compatibility and advanced use.

## Initial weight contract (complaint #15)

A freshly registered layer starts from **deterministic initial weights** — no
implicit random initialization, ever:

- Linear, Conv1d/Conv2d/ConvTranspose2d, Embedding, SwiGlu, Ghost, SeBlock:
  **all-zero** weights (`Initializer::Zeros`, single materialization, no RNG
  draw).
- Norm layers (batch/group/instance/layer/RMS): Burn's deterministic defaults
  (**gamma = 1, beta = 0**), unchanged from upstream semantics.
- PReLU: alpha comes from the (deterministic) layer configuration.

Two fresh registries therefore always produce identical weights, exports, and
run outputs. Callers must set real weights with `LayerRegistry.setWeightsFlat`
(or import a bundle that carries state) before relying on output values.

Malformed facade configuration should be rejected before backend initialization whenever the facade can validate it deterministically. The final generated Python/Rust/JS tool does not need to ship this WASM runtime unless the product itself chooses to use it at runtime.

## Host communication context

This facade describes the WASM-facing ergonomic path, not every supported host path.

- Node reaches this generated WASM surface through the verified `node.mjs` filesystem/`initSync` adapter.
- Native Rust consumers use the packaged Rust API directly and do not route through WASM.
- Python consumers use `burn-research.ffi.v1` through CFFI and do not route through WASM.

See [WASM host communication contract](wasm-host-communication.md) before diagnosing host-specific initialization or transport differences as reference-machine design failures.
