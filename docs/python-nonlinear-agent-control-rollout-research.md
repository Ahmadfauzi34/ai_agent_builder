# Python Nonlinear Agent Control Rollout Research

Issue: #231

## Purpose

Validate the newly representable nonlinear Python policy in the same deterministic closed-loop control workload already used for the Linear-policy host proof. This is a workload/evidence slice, not a new runtime semantic.

## Controlled change

The established control policy is:

```text
Linear(6 -> 2, bias=true)
```

This research changes only policy representation to:

```text
Linear(6 -> 8, bias=true)
  -> ReLU
  -> Linear(8 -> 2, bias=true)
```

Everything else remains aligned with the established control workload:

- seed `20_260_917`;
- 24 generations;
- population 8;
- sigma 0.15;
- learning rate 0.06;
- four deterministic scenarios;
- horizon 16;
- identical environment dynamics, disturbances, state/action/imitation costs, terminal cost, and action clamping.

The canonical trainable-state dimension is:

```text
(6 * 8 + 8) + (8 * 2 + 2) = 74
```

ReLU contributes no trainable parameters.

## Ownership boundary

Python owns environment dynamics, reward, rollout schedule, and horizon. The compiled graph owns policy execution only. ES owns candidate generation/update only. `GraphParameterBinding` remains the sole canonical source for parameter length/order.

Candidate batches use `ask_f32()` and contiguous f32 windows are passed directly to `apply_flat`; no Python-side canonical offset derivation is permitted.

## Proof boundary

The fresh installed-wheel research must prove:

- typed ReLU host surface is available;
- binding length is exactly 74;
- candidate batches/windows are contiguous native f32;
- all fitness values stay finite;
- graph and binding identities remain stable across training;
- optimizer lifecycle/cardinality remains stable across all generations;
- stateful ProgramBundle replay restores exact trainable state;
- replay reward matches the final champion within `1e-9`.

Timing buckets (`ask_f32`, candidate apply, rollout, tell) are evidence only and are not CI performance thresholds.

## Interpretation

Training-quality labels are descriptive. A weak gain does not fail the semantic proof: with the same population/generation budget, moving from 14 to 74 parameters may expose search-scaling limits. That outcome should trigger optimizer/search evidence before any further activation or model-surface expansion.

A strong gain establishes that the nonlinear consumer path is operational, but it still does not justify widening the ABI without a separate representability need.

## Non-goals

No ABI changes, additional activation variants, optimizer algorithm changes, graph/controller semantics, Linear wrapper optimization, zero-copy/NumPy/DLPack/PyO3, Math Program v10, or Go/C++ work.
