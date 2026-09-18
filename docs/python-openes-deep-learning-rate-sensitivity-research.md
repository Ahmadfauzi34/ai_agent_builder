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

## Status

Candidate evidence slice. Results are recorded only after exact-head research and global regression gates are green.
