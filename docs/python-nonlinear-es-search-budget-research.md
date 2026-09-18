# Python Nonlinear ES Search Budget Research

Issue: #233

## Purpose

Measure whether the 74-parameter nonlinear policy is primarily constrained by the OpenES search budget before changing optimizer semantics or widening model/activation surface.

The policy and closed-loop task are fixed:

```text
Linear(6 -> 8, bias=true)
  -> ReLU
  -> Linear(8 -> 2, bias=true)
```

The graph has exactly 74 trainable parameters. Environment dynamics, reward, scenarios, horizon, seed, sigma, learning rate, action clamp, typed Python host path, and ABI remain unchanged.

## Controlled matrix

Each case uses a fresh `EsOptimizer.strict` instance from the same seed and the graph is reset to the exact zero-policy state before the case begins.

```text
A: population 8,  generations 24 = 192 candidate evaluations
B: population 16, generations 24 = 384 candidate evaluations
C: population 8,  generations 48 = 384 candidate evaluations
D: population 16, generations 48 = 768 candidate evaluations
```

Cases B and C have the same evaluation count. Their comparison isolates population breadth versus additional generation depth without changing objective semantics.

## Proof boundary

For every case the installed-wheel consumer proves:

- canonical parameter dimension remains 74;
- `ask_f32()` yields contiguous native-f32 candidate batches;
- candidate windows feed `GraphParameterBinding.apply_flat` directly;
- fitness remains finite;
- batch cardinality and exact candidate-evaluation count remain stable;
- graph and binding identities do not change;
- final champion reward is reproducible after re-apply.

After all four cases, the case with the highest validated final champion reward is selected only for evidence replay. Stateful ProgramBundle export/import must reproduce exact parameter state and reward within `1e-9`.

The selection is not a production champion or optimizer policy decision.

## Evidence fields

Each case reports zero-policy reward, initial/final champion reward, baseline/training gain ratios, generation-best history, champion history, optimizer flags, candidate count, median ask/apply/rollout/tell timing, and total training wall time.

Timing and training-quality classifications are evidence only. CI pass/fail is based on semantic invariants, not a required speedup or reward threshold.

## Interpretation

- If larger budgets substantially improve final reward, search scaling is visible and the next investigation should focus on breadth/depth behavior before changing optimizer algorithms.
- If B beats C at the same 384 evaluations, population breadth is more useful on this landscape.
- If C beats B, extra generations/depth are more useful.
- If neither improves meaningfully, the simple task may favor the smaller Linear policy or current OpenES settings may be mismatched; gather optimizer evidence before expanding model surface.
- Runtime growth proportional to rollout count remains expected useful computation and does not by itself justify reopening Linear wrapper micro-optimization.

## Non-goals

No optimizer algorithm change, ABI change, activation addition, graph/controller semantic change, zero-copy/NumPy/DLPack/PyO3, Math Program v10, Linear wrapper work, or Go/C++ support.
