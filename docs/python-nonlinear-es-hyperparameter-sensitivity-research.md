# Python Nonlinear ES Hyperparameter Sensitivity Research

Issue: #237

## Purpose

Search-budget evidence showed that the 74-parameter nonlinear policy benefits substantially from additional generations, while doubling population alone did not help at the same evaluation count. This slice asks whether the long champion plateaus are especially sensitive to the fixed OpenES exploration radius (`sigma`) or update step size (`learning_rate`) before any optimizer-algorithm change.

## Fixed workload

Every case keeps:

- fresh installed wheel and typed Python host path;
- policy `Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)`;
- canonical parameter dimension 74;
- seed `20_260_917`;
- population 8;
- 48 generations;
- exactly 384 candidate evaluations;
- four deterministic scenarios and 16-step horizon;
- identical dynamics, disturbances, reward/cost terms, and action clamping;
- OpenES strategy 0;
- Python-owned environment/reward/schedule, graph-owned policy execution, and ES-owned candidate generation/update.

## One-variable-at-a-time matrix

| Case | Sigma | Learning rate |
| --- | ---: | ---: |
| BASE | 0.15 | 0.06 |
| S_LOW | 0.08 | 0.06 |
| S_HIGH | 0.25 | 0.06 |
| LR_LOW | 0.15 | 0.03 |
| LR_HIGH | 0.15 | 0.10 |

No case changes sigma and learning rate simultaneously.

## Stagnation evidence

In addition to reward histories and standard timing buckets, each case reports:

- champion improvement count;
- exact generations where champion improvements occur;
- first and last improvement generation;
- longest consecutive run of generation transitions with no champion improvement;
- count of `NO_IMPROVEMENT` optimizer flags.

This allows the experiment to distinguish a higher final reward from merely different plateau behavior.

## Proof boundary

All cases must preserve:

- exact 74-parameter binding;
- contiguous native-f32 `ask_f32()` batches and candidate windows;
- exact 384 candidate evaluations;
- finite fitness;
- stable batch cardinality;
- stable graph identity;
- stable binding identity.

The validated case with the highest final champion reward is used only for exact ProgramBundle state/reward replay. That selection is research evidence plumbing, not a production optimizer policy.

## Interpretation

- sigma variant leads: exploration radius is the immediate search-dynamics lever;
- learning-rate variant leads: update size is the immediate lever;
- BASE remains best/tied while plateaus persist: fixed OpenES dynamics may be the next algorithm-level research target;
- any improvement still needs to be interpreted against the simple 14-parameter Linear policy, which remains a strong control for whether added nonlinear capacity is useful on this task.

## Non-goals

No optimizer algorithm changes, adaptive sigma, annealing/schedules, ABI/facade changes, activation expansion, graph/controller changes, Linear wrapper work, Math Program v10, NumPy/DLPack/PyO3, or Go/C++ work.
