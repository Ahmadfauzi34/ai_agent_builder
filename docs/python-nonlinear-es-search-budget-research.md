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

## Observed evidence

The dedicated installed-wheel matrix passed all semantic proofs.

| Case | Population | Generations | Evaluations | Final reward | Baseline gain | Training wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 8 | 24 | 192 | -1.207370 | 49.85% | 8.05 s |
| B | 16 | 24 | 384 | -1.236603 | 48.64% | 15.28 s |
| C | 8 | 48 | 384 | -0.827027 | 65.65% | 16.08 s |
| D | 16 | 48 | 768 | -0.770807 | 67.98% | 30.42 s |

Zero-policy reward is `-2.407589`. Higher reward is better.

At equal 384 candidate evaluations, C beats B by `0.409576` reward. Doubling population while keeping 24 generations did not help: B is slightly worse than A. Doubling generations from 24 to 48 at population 8 materially improves the final reward: C beats A by about `0.380343`.

D is the best observed case and beats A by `0.436563`, but D improves over C by only about `0.056219` despite adding another 384 candidate evaluations. This shows diminishing benefit from adding breadth after sufficient generation depth.

Median rollout time stays approximately `36.2 ms` across all cases, while total training wall time grows roughly with evaluation count. Candidate apply remains around `0.22-0.24 ms` median, so the additional budget is spent mostly on useful rollout computation rather than host marshalling.

Case D was selected for evidence replay. ProgramBundle state replay was exact and reward reproduced exactly at `-0.7708074646643113`; checkpoint size was 1282 bytes.

## Decision

```text
NONLINEAR_POLICY_SEARCH_BUDGET_SCALING_VISIBLE
DEPTH_OVER_BREADTH_AT_EQUAL_EVALUATION_COUNT
DO_NOT_WIDEN_ACTIVATION_SURFACE
DO_NOT_CHANGE_OPTIMIZER_ALGORITHM_YET
```

The 74-parameter nonlinear policy is not blocked by semantics. Additional search budget helps, but current evidence says generation depth is substantially more useful than population breadth on this landscape.

The next narrow question should compare equal 768-evaluation budgets with more depth, for example population 8 × 96 generations versus the already observed population 16 × 48 case, before changing optimizer algorithms or model surface.

## Evidence fields

Each case reports zero-policy reward, initial/final champion reward, baseline/training gain ratios, generation-best history, champion history, optimizer flags, candidate count, median ask/apply/rollout/tell timing, and total training wall time.

Timing and training-quality classifications are evidence only. CI pass/fail is based on semantic invariants, not a required speedup or reward threshold.

## Interpretation

- Larger budget materially improves final reward, so search scaling is visible.
- C beats B at the same 384 evaluations, so additional generations/depth are more useful than population breadth for this workload.
- D adds a smaller gain over C, suggesting diminishing returns from breadth at the tested settings.
- Runtime scales with rollout count as expected useful computation; this is not evidence for reopening Linear wrapper micro-optimization.
- The nonlinear policy remains operational even though this simple task may still favor the smaller Linear policy.

## Non-goals

No optimizer algorithm change, ABI change, activation addition, graph/controller semantic change, zero-copy/NumPy/DLPack/PyO3, Math Program v10, Linear wrapper work, or Go/C++ support.
