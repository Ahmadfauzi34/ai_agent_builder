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

## Observed evidence

The dedicated installed-wheel run passed all semantic/replay checks.

| Case | Evaluations | Final champion reward | Baseline gain | Training wall |
| --- | ---: | ---: | ---: | ---: |
| A | 192 | -1.207370 | 49.85% | 8.04 s |
| B | 384 | -1.236603 | 48.64% | 14.96 s |
| C | 384 | -0.827027 | 65.65% | 15.90 s |
| D | 768 | -0.770807 | 67.98% | 29.94 s |

Higher reward (less negative) is better.

At the same 384 candidate evaluations, C materially outperformed B:

```text
B: population 16 × 24 generations -> -1.236603
C: population  8 × 48 generations -> -0.827027
```

Therefore the equal-budget evidence is:

```text
DEPTH_BETTER_AT_384_EVALS
```

The largest budget D also improved over A:

```text
A: -1.207370
D: -0.770807
```

which reduces final cost magnitude by about 36.2% relative to A. The report classifies this as:

```text
LARGER_BUDGET_IMPROVES_FINAL_REWARD
```

D was selected only for checkpoint replay. Its exact reward `-0.7708074646643113` replayed exactly within `1e-9`, and parameter state, graph identity, and binding identity all remained exact/stable.

Median candidate rollout time stayed essentially flat across the matrix (~35.6–36.0 ms), while training wall time scaled approximately with candidate-evaluation count. Median candidate apply also remained ~0.23–0.25 ms. The additional cost is therefore dominated by performing more useful rollout evaluations, not by a new binding or wrapper pathology.

For context, the previously validated 14-parameter Linear policy on the same control task finished around `-0.5210`. Even D does not match that simpler policy on this task, so the evidence does not justify adding more activation/model surface.

## Interpretation

The 74-parameter nonlinear policy benefits from more search budget, but the strongest signal is **generation depth**, not population breadth:

1. B doubled population while keeping 24 generations and did not improve A.
2. C kept population 8 but doubled generations and clearly improved.
3. D added both breadth and depth and achieved the best nonlinear result, but only modestly improved over C relative to the doubled evaluation count.
4. The simple control task still favors the smaller Linear policy under the tested OpenES settings.

The next optimizer investigation should therefore focus on **search/update dynamics across generations** before changing the optimizer algorithm. A useful next slice is to inspect whether stagnation / `NO_IMPROVEMENT` phases are driven by fixed sigma or learning-rate behavior, while keeping the algorithm and objective unchanged.

## Non-goals

No optimizer algorithm changes, ABI changes, additional activation variants, graph/controller changes, Linear wrapper optimization, zero-copy/NumPy/DLPack/PyO3, Math Program v10, or Go/C++ work.
