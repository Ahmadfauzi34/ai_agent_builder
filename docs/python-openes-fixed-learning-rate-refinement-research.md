# Python OpenES Fixed Learning-Rate Refinement Research

Issue: #256

## Purpose

PR #255 showed that LR 0.10 leads LR 0.06 and LR 0.03 at every measured depth under sigma 0.08, but its advantage narrows at generation 192 and its final plateau is longer.

This slice refines the fixed-rate region around 0.10 before introducing any adaptive schedule.

## Fixed workload

- installed Python wheel + typed host;
- policy `Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)`;
- parameter dimension 74;
- seed `20_260_917`;
- population 8;
- sigma 0.08;
- 192 generations;
- 1536 candidate evaluations per case;
- identical deterministic four-scenario control task.

## Matrix

| Case | Learning rate |
| --- | ---: |
| LR_008 | 0.08 |
| LR_010 | 0.10 |
| LR_012 | 0.12 |

Only learning rate changes.

## Evidence

For each case record milestone rewards at 48/96/144/192, final reward, champion plateau metrics, interval improvement counts, fitness std, diversity, normalized antithetic contrast, mean-norm path length, exact candidate count, and stable program/binding identities.

LR_010 must reproduce the validated #255 milestones. The best final case is used only for exact ProgramBundle replay.

## Interpretation

- LR_012 leads materially: the optimum fixed-rate region remains above 0.10.
- LR_008 leads: 0.10 overshoots the best fixed-rate region at deep horizon.
- LR_010 leads/ties: 0.10 is a reasonable fixed operating point and fixed-LR tuning should stop unless a new workload changes the evidence.
- leader changes with depth: a later schedule experiment becomes more credible, but this slice does not implement one.

## Scope

Exactly three research-only files. No adaptive LR/sigma, optimizer algorithm, ABI/facade/control API, graph/model/Math/ProgramBundle semantic change.

## Observed evidence

The installed-wheel research run passed all candidate-count, identity, telemetry, LR_010 continuity, and ProgramBundle replay proofs.

| Case | LR | Gen 48 | Gen 96 | Gen 144 | Gen 192 | Improvements | Longest plateau | Last improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR_008 | 0.08 | -0.552942 | -0.354409 | -0.307413 | **-0.303202** | **50** | 17 | 175 |
| LR_010 | 0.10 | **-0.489408** | -0.330293 | -0.308263 | -0.304647 | 45 | 33 | 188 |
| LR_012 | 0.12 | -0.547216 | **-0.324844** | -0.317236 | -0.317236 | 42 | **84** | **108** |

Higher reward (less negative) is better.

The leader changes with depth:

```text
generation 48   -> LR_010
generation 96   -> LR_012
generation 144  -> LR_008
generation 192  -> LR_008
```

At the final horizon, LR_008 beats LR_010 only narrowly:

```text
LR_008 = -0.3032023985
LR_010 = -0.3046468133
```

which is about a **0.47%** reduction in cost magnitude.

The important result is therefore not that LR 0.08 is universally superior. LR 0.10 is clearly stronger early and remains competitive deep into the run, while LR 0.08 continues improving later and finishes slightly better.

LR_012 shows the strongest overshoot/stagnation signal. It reaches its final champion at generation 108, makes no champion improvement during generations 145–192, and accumulates an 84-generation longest plateau despite healthy diversity and order-one normalized antithetic signal.

Late interval (145–192):

| Case | Champion improvements | Fitness std median | Diversity median | Normalized antithetic contrast | Mean-norm path |
| --- | ---: | ---: | ---: | ---: | ---: |
| LR_008 | **4** | 0.01223 | 0.07573 | **0.94881** | 1.31113 |
| LR_010 | 3 | 0.01547 | 0.07573 | 0.89003 | 1.11656 |
| LR_012 | **0** | 0.01935 | 0.07573 | 0.88235 | **1.64348** |

So LR_012 continues moving strongly while producing no new lifetime champion in the final interval. That is consistent with an update step that is too aggressive for this late-stage region, although the research does not prove a causal overshoot mechanism.

The highest final-reward LR_008 state replays exactly through stateful ProgramBundle.

## Decision

```text
FIXED_LR_OPTIMUM_IS_DEPTH_DEPENDENT
LR_010_IS_STRONGER_EARLY
LR_008_IS_SLIGHTLY_BETTER_LATE
LR_012_SHOWS_CLEAR_DEEP_STAGNATION
STOP_BROAD_FIXED_LR_TUNING
KEEP_SIGMA_0_08
KEEP_OPENES_ALGORITHM_UNCHANGED
SCHEDULE_HYPOTHESIS_IS_NOW_JUSTIFIED
DO_NOT_ADD_SCHEDULE_CONTROL_WITHOUT_CONTINUITY_PROOF
```

The next research question is no longer another broad fixed-LR sweep. It is whether a continuous optimizer trajectory that starts at LR 0.10 and later reduces to LR 0.08 can retain the stronger early progress while improving late-stage refinement.

That experiment requires preserving optimizer search state across the LR transition. If the current public optimizer surface cannot mutate learning rate in place, the next step should first define and prove the minimal state/control boundary rather than approximating a schedule by restarting the optimizer.
