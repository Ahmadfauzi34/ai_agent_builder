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
- champion best-norm start/end/delta.

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

## Status

Candidate evidence slice. Results are recorded only after the exact PR head produces a successful report artifact and global regression gates remain green.
