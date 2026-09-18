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

## Status

Candidate evidence slice. Results are recorded only after exact-head research and global regression gates are green.
