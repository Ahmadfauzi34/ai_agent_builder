# Python Nonlinear ES Search-Budget Scaling Research

Issue: #233

## Purpose

The nonlinear Python policy path is semantically proven and operational, but the 74-parameter policy finished worse than the 14-parameter Linear policy when both used the same OpenES population and generation count. This research asks a narrower question before changing optimizer algorithms or adding more model surface:

> Is the observed gap mainly a search-budget scaling effect?

## Fixed workload

The following remain unchanged from the nonlinear closed-loop workload:

- fresh installed wheel and typed Python host path;
- policy `Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)`;
- canonical parameter dimension 74;
- seed `20_260_917`;
- sigma 0.15 and learning rate 0.06;
- four deterministic scenarios;
- 16-step horizon;
- identical dynamics, disturbances, reward/cost terms, action clamping, and host ownership boundaries.

No optimizer algorithm or core/runtime semantic changes are allowed in this slice.

## Search-budget matrix

Each case creates a fresh optimizer and resets the graph to the same zero trainable state:

| Case | Population | Generations | Candidate evaluations |
| --- | ---: | ---: | ---: |
| A | 8 | 24 | 192 |
| B | 16 | 24 | 384 |
| C | 8 | 48 | 384 |
| D | 16 | 48 | 768 |

B and C have the same candidate-evaluation count, so they provide the cleanest breadth-versus-depth comparison.

## Required evidence

For every case the report records:

- zero-policy, initial champion, and final champion reward;
- baseline/training gain ratios;
- generation-best and champion histories;
- optimizer flags;
- exact candidate evaluation count;
- median/min/p90/max ask, apply, rollout, and tell timings;
- total training wall time;
- finite-fitness and stable-batch proofs.

Graph and binding identities must remain stable across every case.

After all cases finish, the validated case with the highest final champion reward is used only for stateful ProgramBundle replay. Exact trainable-state replay and reward replay within `1e-9` are mandatory. This selection is evidence plumbing, not a production promotion rule.

## Interpretation

- D materially better than A: larger search budget helps; inspect breadth/depth pattern before optimizer changes.
- B better than C at the same 384 evaluations: population breadth helps more.
- C better than B: additional generations help more.
- Neither B/C/D materially improves A: the simple task may favor the smaller Linear policy, or current OpenES settings/algorithm may be mismatched. Gather optimizer evidence before changing model surface.
- Runtime growth dominated by rollout count remains expected useful graph computation and is not evidence for reopening Linear wrapper micro-optimization.

## Non-goals

No optimizer algorithm changes, ABI changes, additional activation variants, graph/controller changes, Linear wrapper optimization, zero-copy/NumPy/DLPack/PyO3, Math Program v10, or Go/C++ work.
