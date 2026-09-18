# Python OpenES Search-State Telemetry Research

Issue: #252

## Purpose

The corrected-sigma nonlinear OpenES lineage improves through 192 generations but with clear diminishing returns. This slice explains the late champion plateaus using search-state telemetry already returned by the supported typed Python host.

It does not change OpenES.

## Fixed lineage

```text
policy        = Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)
parameters    = 74
population    = 8
sigma         = 0.08
learning_rate = 0.06
seed          = 20_260_917
generations   = 192
evaluations   = 1536
```

Environment, reward, horizon, graph/binding semantics, and ProgramBundle semantics remain identical to #242/#243.

## Existing telemetry

Every `EsOptimizer.tell()` report already returns:

- generation best/worst/mean fitness;
- fitness standard deviation;
- lifetime-best improvement;
- stagnation counter;
- candidate diversity;
- sigma and learning rate;
- optimizer search-mean norm;
- lifetime-best parameter norm;
- diagnostic flags.

The host also already owns the evaluated candidate-fitness vector. Because OpenES emits candidates as antithetic `(+epsilon, -epsilon)` pairs, the research derives the absolute pair-fitness contrast and the same contrast normalized by the generation fitness std. This is descriptive telemetry only; candidate ordering and `tell()` input remain unchanged.

The research records these values every generation through the typed host API. No raw ABI or new facade method is needed.

## Interval analysis

The same four 48-generation intervals are summarized:

- 1–48
- 49–96
- 97–144
- 145–192

Each interval records champion progress plus:

- diversity distribution;
- fitness-std distribution;
- median population-mean fitness;
- median generation-best fitness;
- maximum stagnation;
- search mean-norm start/end/delta and norm-path length;
- champion best-norm start/end/delta;
- raw antithetic pair-fitness contrast;
- fitness-std-normalized antithetic pair contrast.

The final lifetime-champion plateau is also summarized separately.

## Interpretation boundary

This evidence can distinguish a lifetime-best plateau from obvious population-collapse signals.

It cannot reconstruct search direction because ABI v1 does not expose the full optimizer mean vector. `mean_norm` only reports the vector magnitude.

Therefore this slice must not claim:

- optimizer mean-vector direction;
- exact search-center displacement;
- exact mean-to-champion distance.

If those become necessary, they require a separate ABI/facade design decision.

## Proof

Require:

- exact 1536 candidate evaluations;
- finite telemetry fields;
- stable program and binding identities;
- fixed sigma/LR in every report;
- exact ProgramBundle replay of the final champion;
- milestone continuity with the deterministic #243 rewards at generations 48/96/144/192 within a narrow descriptive tolerance.

## Scope

Exactly three research-only files:

- `.github/workflows/python-openes-search-state-telemetry-research.yml`
- `docs/python-openes-search-state-telemetry-research.md`
- `scripts/research_python_openes_search_state_telemetry.py`

No ABI widening, no facade method, no optimizer algorithm/adaptive schedule, no graph/model/Math/ProgramBundle semantic change.

## Observed evidence

The dedicated installed-wheel run passed all telemetry, cardinality, identity, milestone-continuity, and final ProgramBundle replay proofs.

| Interval | Champion improvements | Median diversity | Median fitness std | Median normalized antithetic contrast | Mean-norm path | Max stagnation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–48 | 17 | 0.07574 | 0.70674 | 0.68932 | 0.90685 | 7 |
| 49–96 | 16 | 0.07494 | 0.10811 | 0.86305 | 1.13373 | 4 |
| 97–144 | 13 | 0.07488 | 0.02121 | 0.93967 | 1.12308 | 10 |
| 145–192 | 8 | 0.07573 | 0.01239 | 0.67726 | 0.68327 | 14 |

The parameter-space population diversity does **not** collapse as training progresses. Its median stays close to 0.075 across all four intervals and no `DIVERSITY_COLLAPSE` signal is observed.

Raw objective contrast does compress strongly. Median raw antithetic pair contrast falls approximately:

```text
0.43705 -> 0.08269 -> 0.01972 -> 0.00764
```

and median generation fitness std falls:

```text
0.70674 -> 0.10811 -> 0.02121 -> 0.01239
```

However OpenES standardizes fitness before using the antithetic pair differences. The corresponding normalized pair contrast remains order-one rather than collapsing:

```text
0.68932 -> 0.86305 -> 0.93967 -> 0.67726
```

This means the late search still contains directional information even though absolute reward differences have become much smaller.

### Final champion plateau

The lifetime champion last improves at generation 178. Generations 179–192 form a 14-generation champion plateau.

During that plateau:

- median diversity = `0.07682`;
- median fitness std = `0.01080`;
- median normalized antithetic contrast = `0.79056`;
- median normalized antithetic RMS = `1.19214`;
- no diversity-collapse flags occur;
- no all-fitness-equal flags occur;
- `mean_norm` changes from `3.87872` to `3.99471`;
- mean-norm path length is `0.23194`;
- stagnation reaches 14.

Therefore the final lifetime-best plateau is **not evidence that the optimizer stopped searching**. The search-center magnitude still changes and the normalized antithetic directional signal remains active.

The report classifies this as:

```text
SEARCH_NORM_STILL_MOVING_ON_FINAL_CHAMPION_PLATEAU
DEPTH_RETURNS_DIMINISHING_BUT_ACTIVE
DEPTH_SATURATION_NOT_OBSERVED_AT_192
```

The final champion and ProgramBundle replay remain exactly continuous with #243 at reward `-0.3092712352204114`.

## Decision

```text
NO_PARAMETER_DIVERSITY_COLLAPSE
RAW_OBJECTIVE_CONTRAST_COMPRESSES
NORMALIZED_ANTITHETIC_SIGNAL_REMAINS_ACTIVE
SEARCH_CENTER_NORM_STILL_MOVES
CHAMPION_PLATEAU_IS_NOT_SEARCH_STOP
KEEP_ABI_V1_UNCHANGED
KEEP_OPENES_ALGORITHM_UNCHANGED
TEST_DEEP_LEARNING_RATE_SENSITIVITY_AT_SIGMA_0_08_NEXT
```

This evidence does not yet prove that the fixed learning rate is too high. It only makes learning-rate sensitivity the next narrow hypothesis: the optimizer continues to receive normalized directional signal and continues to move while lifetime-best gains become smaller and less frequent.

A follow-up should therefore hold population, sigma, task, model, seed, and depth fixed while comparing fixed learning rates at the deep horizon before introducing adaptive schedules.

The full optimizer mean vector is still not exposed through ABI v1. Since existing telemetry already distinguishes champion stagnation from obvious search collapse, this research does not justify widening ABI solely for additional observability.
