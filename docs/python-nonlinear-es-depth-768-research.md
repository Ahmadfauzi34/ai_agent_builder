# Python Nonlinear ES Depth at Equal 768 Evaluations

Issue: #236

## Purpose

Test whether generation depth remains more useful than population breadth for the 74-parameter nonlinear policy when both alternatives receive exactly 768 candidate evaluations.

The fixed policy is:

```text
Linear(6 -> 8, bias=true)
  -> ReLU
  -> Linear(8 -> 2, bias=true)
```

Environment dynamics, reward, scenarios, horizon, seed, sigma, learning rate, installed-wheel host path, ABI, graph semantics, and optimizer algorithm remain unchanged.

## Controlled matrix

Each case uses a fresh `EsOptimizer.strict` from the same seed and the graph is reset to the exact zero-policy state before training.

```text
C: population 8,  generations 48 = 384 evaluations
D: population 16, generations 48 = 768 evaluations
E: population 8,  generations 96 = 768 evaluations
```

C -> E isolates doubling generation depth at fixed population.

D vs E is the primary comparison: both use exactly 768 candidate evaluations, so the difference is search breadth versus generation depth rather than total evaluation budget.

## Required proof

For every case:

- canonical parameter dimension is exactly 74;
- `ask_f32()` returns contiguous native-f32 storage;
- candidate windows feed `GraphParameterBinding.apply_flat` directly;
- fitness remains finite;
- exact candidate-evaluation count and batch cardinality remain stable;
- graph and binding identities remain unchanged;
- final champion reward is reproducible after re-apply.

The best validated case is selected only for evidence replay. Stateful ProgramBundle import must reproduce exact parameter state and reward within `1e-9`.

Timing and reward comparisons are evidence only, not CI thresholds.

## Interpretation

- E > D: depth remains the stronger lever at equal 768 evaluations. Keep optimizer algorithm and hyperparameters unchanged; inspect depth saturation next.
- D > E: breadth becomes useful at larger budget and a balanced population/depth schedule deserves investigation.
- E > C but E <= D: depth helps, but breadth+depth remains stronger.
- C, D, and E all plateau above the simpler Linear-policy result: investigate task fit or fixed OpenES hyperparameters before adding model surface.

## Non-goals

No optimizer algorithm change, no sigma/learning-rate tuning, no ABI/model/activation change, no graph/controller semantic change, no Linear wrapper work, no zero-copy/NumPy/DLPack/PyO3, no Math Program v10, and no Go/C++ support.
