# Python ReLU Layer Spec Proof

Issue: #229

## Purpose

Close the concrete Python/FFI representability gap for nonlinear policies without changing core activation semantics. The core already supports `AgentLayerSpec::relu`; this slice only exposes that existing capability through ABI v1 and the typed Python host.

## Added surface

```text
br_v1_layer_relu(uint32_t layer_id, br_v1_handle **out)
        ↓
existing LayerSpec(AgentLayerSpec) opaque handle
        ↓
ReluLayerSpec
```

`LinearLayerSpec` and `ReluLayerSpec` share one internal `_LayerSpec` type boundary. Registry and graph-builder methods accept only that layer-spec base; arbitrary owned handles or Python objects are not accepted.

## Proof workload

A fresh installed wheel constructs:

```text
Linear(6 -> 8, bias=true)
  -> ReLU
  -> Linear(8 -> 2, bias=true)
```

The canonical trainable-state length must be exactly:

```text
(6 * 8 + 8) + (8 * 2 + 2) = 74
```

so ReLU contributes zero trainable parameters.

A deterministic zero-bias candidate sets both Linear weight matrices to all ones. For `x=[1,1,1,1,1,1]`, the policy output is positive. For `-x`, the first Linear output is negative and ReLU clamps it to zero, so the final output is zero. A zero-bias purely Linear chain would satisfy odd symmetry `f(-x)=-f(x)`; this graph does not, providing a direct nonlinear-behavior proof.

The same proof also requires stable graph/binding identities and exact state/output replay after stateful ProgramBundle export/import.

## Non-goals

No additional activation variants, ABI v2, graph/controller semantics, trainable-state changes, optimizer changes, Math Program v10, NumPy/DLPack/PyO3, or Go/C++ support.

The next workload step, after this boundary is proven and merged, is a separate nonlinear closed-loop agent research slice.
