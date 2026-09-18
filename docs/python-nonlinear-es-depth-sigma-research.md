# Python Nonlinear ES Depth After Sigma Correction Research

Issue: #236

## Purpose

The earlier nonlinear search-budget matrix established that generation depth was more useful than population breadth at 384 candidate evaluations, but it used the original OpenES setting `sigma=0.15`.

The subsequent hyperparameter-sensitivity study (#237/#238) changed that premise. At 384 evaluations, `sigma=0.08` materially improved the nonlinear policy and shortened champion plateaus. This slice asks the next controlled question:

> Does generation depth remain the stronger search lever after correcting sigma?

## Fixed workload

The research keeps:

- fresh installed wheel and typed Python host;
- `Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)`;
- exactly 74 trainable parameters;
- deterministic four-scenario 2D control workload;
- 16-step horizon;
- seed `20_260_917`;
- OpenES strategy 0;
- **sigma 0.08**;
- learning rate 0.06;
- Python-owned environment/reward/schedule;
- graph-owned policy execution;
- optimizer-owned candidate generation/update;
- `ask_f32()` candidate storage and direct contiguous candidate windows into `apply_flat`.

No optimizer semantics or model surface change in this slice.

## Matrix

| Case | Population | Generations | Evaluations |
| --- | ---: | ---: | ---: |
| C_LOW_SIGMA | 8 | 48 | 384 |
| D_LOW_SIGMA | 16 | 48 | 768 |
| E_LOW_SIGMA | 8 | 96 | 768 |

`C_LOW_SIGMA` anchors the validated low-sigma 8×48 setting from #238.

`D_LOW_SIGMA` versus `E_LOW_SIGMA` is the key equal-768 breadth-versus-depth comparison.

## Proof requirements

For each independent optimizer run:

- graph resets to exact zero trainable state;
- finite fitness across all evaluations;
- exact requested batch/cardinality and candidate count;
- stable program and binding identities;
- contiguous native-f32 candidate windows;
- zero/initial/final champion rewards;
- generation-best and champion histories;
- champion improvement generations/count;
- longest champion plateau;
- first/last champion improvement generation;
- `NO_IMPROVEMENT` count;
- ask/apply/rollout/tell timing and total wall time.

The highest validated final-reward case is used only for exact stateful `ProgramBundle` replay. Parameter state must replay exactly and reward must match within `1e-9`.

## Decision boundary

- `E_LOW_SIGMA > D_LOW_SIGMA` at equal 768 evaluations: depth remains the stronger lever after sigma correction.
- `D_LOW_SIGMA > E_LOW_SIGMA`: breadth becomes more useful at the larger corrected-sigma budget.
- `E_LOW_SIGMA > C_LOW_SIGMA`: further depth still buys useful search progress.
- long plateaus persist even with corrected sigma: fixed OpenES dynamics may be approaching an algorithm-level limitation, which would justify a separate optimizer-algorithm research slice.
- nonlinear policy remains worse than the established simple Linear policy: task simplicity remains an important explanation; do not widen activation/model surface solely for capability claims.

## Scope

Research-only:

- `.github/workflows/python-nonlinear-es-depth-sigma-research.yml`
- `docs/python-nonlinear-es-depth-sigma-research.md`
- `scripts/research_python_nonlinear_es_depth_sigma.py`

No production Rust, ABI/facade, optimizer algorithm, adaptive sigma/learning-rate schedule, graph/binding semantics, activation/model surface, Math v1-v9, ProgramBundle schema, Resolution/Authorization, or support-matrix change.

Timing and training-quality values are descriptive evidence only, never CI performance thresholds.

## Status

Candidate research slice. Results are not recorded here until the exact PR head produces a successful report artifact and all regression gates remain green.
