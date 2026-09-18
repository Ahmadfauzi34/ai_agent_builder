# Python OpenES Deep Learning-Rate Sensitivity Research

Issue: #254

## Purpose

Search-state telemetry from #252/#253 shows that the corrected-sigma search remains active during the final lifetime-champion plateau. This slice tests whether fixed learning rate controls the late-stage diminishing returns before any adaptive schedule is introduced.

## Fixed workload

- installed Python wheel + typed host;
- policy `Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)`;
- parameter dimension 74;
- seed `20_260_917`;
- population 8;
- sigma **0.08**;
- 192 generations;
- 1536 candidate evaluations per case;
- same deterministic four-scenario, 16-step task;
- OpenES strategy 0.

## One-variable matrix

| Case | Sigma | Learning rate |
| --- | ---: | ---: |
| LR_LOW | 0.08 | 0.03 |
| LR_BASE | 0.08 | 0.06 |
| LR_HIGH | 0.08 | 0.10 |

Only learning rate changes.

## Evidence

For every case record milestone champion rewards at generations 48/96/144/192, final reward, plateau behavior, interval improvement count, fitness std, diversity, normalized antithetic contrast, mean-norm path length, exact candidate count, and stable graph/binding identity.

LR_BASE must reproduce the validated corrected-sigma milestones. The highest final-reward case is used only for exact stateful ProgramBundle replay.

## Interpretation

- LR_LOW leads at 192 after not leading at 48: evidence for an early/late step-size tradeoff and a later schedule experiment.
- LR_HIGH leads at 192: LR 0.06 is not obviously too aggressive.
- LR_BASE leads/ties: no evidence for changing LR dynamics.
- a leader change with depth supports testing a schedule, but this slice does not implement one.

## Scope

Exactly three research-only files. No adaptive LR/sigma, OpenES algorithm change, ABI/facade change, graph/model/Math/ProgramBundle semantic change.

## Observed evidence

The dedicated installed-wheel run passed all candidate-count, identity, telemetry, milestone-continuity, and ProgramBundle replay proofs.

| Case | LR | Gen 48 | Gen 96 | Gen 144 | Gen 192 | Improvements | Longest plateau | Last improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_LOW | 0.03 | -0.902505 | -0.608087 | -0.497353 | -0.368239 | 42 | 10 | 192 |
| LR_BASE | 0.06 | -0.645007 | -0.390989 | -0.324997 | -0.309271 | 53 | 14 | 178 |
| LR_HIGH | 0.10 | **-0.489408** | **-0.330293** | **-0.308263** | **-0.304647** | 45 | **33** | 188 |

Higher reward (less negative) is better.

LR_HIGH leads at every measured depth. Relative to LR_BASE, its cost-magnitude advantage is approximately:

```text
generation 48   ~24.12%
generation 96   ~15.52%
generation 144   ~5.15%
generation 192   ~1.50%
```

Therefore the experiment classifies:

```text
HIGH_LEARNING_RATE_LEADS_AT_192
LEARNING_RATE_LEADER_STABLE_WITH_DEPTH
schedule_hypothesis_supported = false
```

The original narrow hypothesis that LR 0.06 might be too aggressive in the late stage is **not supported**. LR 0.10 remains better through generation 192.

However, the advantage of LR 0.10 compresses strongly with depth. Its final 48-generation interval records only three champion improvements and a longest plateau of 33 generations, versus eight improvements / 14 generations for LR_BASE and nine improvements / 10 generations for LR_LOW.

Late-interval telemetry:

| Case | Improvements 145–192 | Fitness std median | Diversity median | Normalized antithetic contrast | Mean-norm path |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR_LOW | 9 | 0.04916 | 0.07573 | 0.90842 | 0.55983 |
| LR_BASE | 8 | 0.01239 | 0.07573 | 0.67726 | 0.68327 |
| LR_HIGH | 3 | 0.01547 | 0.07573 | 0.89003 | 1.11656 |

LR_HIGH therefore keeps moving its search-mean magnitude substantially and retains strong normalized antithetic signal even while lifetime-champion improvement becomes rare.

This is not evidence for lowering learning rate from the start: LR_LOW is materially worse at every milestone. It also does not yet prove that a late learning-rate schedule would help, because the current API does not switch LR on the same optimizer trajectory and this experiment intentionally compares fixed rates only.

The highest final-reward LR_HIGH state replays exactly through stateful ProgramBundle at reward `-0.3046468133328745`.

## Decision

```text
KEEP_SIGMA_0_08
FIXED_LR_0_10_OUTPERFORMS_0_06_AND_0_03
LOW_LR_FROM_START_IS_NOT_SUPPORTED
LR_HIGH_ADVANTAGE_COMPRESSES_WITH_DEPTH
LR_HIGH_LATE_PLATEAU_IS_LONGER
KEEP_OPENES_ALGORITHM_UNCHANGED
DO_NOT_ADD_ADAPTIVE_LR_YET
REFINE_FIXED_LR_AROUND_0_10_NEXT
```

The next narrow slice should refine the fixed learning-rate region around the current winner (for example 0.08 / 0.10 / 0.12) at the same deep horizon before designing an in-place learning-rate schedule or widening optimizer state/control APIs.
