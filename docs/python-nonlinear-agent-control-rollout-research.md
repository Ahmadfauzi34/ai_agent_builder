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

## Evidence

The nonlinear fresh-wheel run passed every semantic proof. Its final state replayed exactly and the replay reward matched the final champion within `1e-9`.

On the same PR head, the existing Linear control workflow and this nonlinear workflow used the same environment and ES budget:

| Metric | Linear 6→2 | Linear 6→8→ReLU→8→2 | Relative nonlinear cost/result |
| --- | ---: | ---: | ---: |
| Trainable parameters | 14 | 74 | 5.29× |
| Zero-policy reward | -2.407589 | -2.407589 | identical |
| Initial champion reward | -4.697449 | -4.396120 | descriptive |
| Final champion reward | -0.521000 | -1.207370 | nonlinear final cost 2.32× Linear |
| Gain from zero-policy | 78.36% | 49.85% | lower under same search budget |
| Gain from initial champion | 88.91% | 72.54% | both train successfully |
| `ask_f32` median | 0.0366 ms | 0.0645 ms | 1.76× |
| candidate apply median | 0.0910 ms | 0.2569 ms | 2.82× |
| rollout median | 11.8040 ms | 36.3822 ms | 3.08× |
| tell median | 0.1081 ms | 0.1258 ms | 1.16× |
| ProgramBundle bytes | 501 | 1282 | 2.56× |

Higher (less negative) reward is better. The nonlinear policy clearly learns from its initial champion and improves over the zero-policy baseline, so the newly exposed nonlinear consumer path is operational. However, this simple control task does not demonstrate a capability advantage over the smaller Linear policy under the same fixed ES budget.

The dominant practical increase is rollout execution, while candidate apply and ask cost also rise. The parameter dimension rises 5.29× but rollout median rises about 3.08×; this is consistent with the extra graph/layer execution plus a larger search space rather than a regression in the ReLU boundary itself.

## Evidence decision

```text
NONLINEAR_CONSUMER_PATH_OPERATIONAL
SEARCH_AND_EXECUTION_SCALING_VISIBLE
DO_NOT_WIDEN_ACTIVATION_SURFACE_YET
DO_NOT_REOPEN_LINEAR_WRAPPER_OPTIMIZATION
MEASURE_NONLINEAR_ES_BUDGET_SCALING_NEXT
```

The next useful experiment is optimizer/search-budget scaling on the same 74-parameter policy. It should vary search budget without changing environment/reward or adding another activation, so we can distinguish insufficient ES exploration from a genuine lack of task benefit.

## Interpretation

Training-quality labels are descriptive. A weak comparative result does not fail the semantic proof: with the same population/generation budget, moving from 14 to 74 parameters exposes a larger search problem and a larger execution graph.

The evidence does not justify another activation variant. ReLU already closes the nonlinear representability gap. Additional surface should wait for a separate representability need.

## Non-goals

No ABI changes, additional activation variants, optimizer algorithm changes, graph/controller semantics, Linear wrapper optimization, zero-copy/NumPy/DLPack/PyO3, Math Program v10, or Go/C++ work.
