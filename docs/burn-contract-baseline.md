# Burn Contract Baseline

This document records the Burn semantics that the agent workflow is allowed to rely on today.

## Current execution profile

- Burn is the numerical/module execution layer. Workspace, Registry, and AgentGraphBuilder orchestrate Burn; they do not replace Burn tensor or module semantics.
- `WasmBackend` is `burn_ndarray::NdArray<f32>`.
- The current backend is intentionally non-autodiff. Autodiff is deferred by architecture and is not a missing baseline requirement.
- The WASM bridge normalizes external tensors to rank 4. Individual wrappers may reshape to the rank expected by the underlying Burn module and must validate the adapter boundary first.
- Modules are created using Burn configs on the backend default device and executed through the real Burn `forward` implementations.
- Module records remain the serialization/state boundary used by Registry state APIs and gradient-free weight bridges.

## BatchNorm consequence

Burn BatchNorm selects training or inference behavior from the backend AD state. With the current non-AD NdArray backend, BatchNorm uses inference behavior and does not update `running_mean` / `running_var` during `forward`.

This is a deliberate baseline property. Regression tests serialize BatchNorm state before and after forward calls and require byte-for-byte equality.

If an autodiff backend is introduced later, this assumption becomes invalid by design: BatchNorm may update running state during training forwards. That future change must introduce an explicit execution-state contract and new failure/recovery audit before the backend can be promoted.

## Contract boundaries

The following are accepted adapter constraints rather than Burn constraints:

- rank-4 `WasmTensor` bridge;
- graph slots and `CompiledGraph` execution order;
- Workspace lifecycle and transactional initialization;
- init identity fingerprints and proof metadata.

The following remain Burn-owned semantics and must not be silently redefined by the wrapper:

- backend/device behavior;
- tensor dtype/rank requirements of each Burn operation;
- module parameter and record semantics;
- forward numerical behavior;
- autodiff behavior when/if an AD backend is enabled;
- running-state behavior of stateful modules.

## Promotion rule

A backend or training-mode change is not a transparent implementation detail. Any change that alters `Backend::ad_enabled`, device semantics, dtype behavior, or stateful-module forward behavior requires a Burn contract-conformance audit before it becomes the default agent execution profile.
