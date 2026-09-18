# Python Nonlinear ES Depth Saturation Research

Issue: #242

## Purpose

PR #241 proved that, after correcting sigma to 0.08, population 8 × 96 generations materially outperforms population 16 × 48 at the same 768 candidate evaluations. The champion still improved at generation 96.

This research therefore asks a narrower question before any OpenES algorithm change:

> Does the same corrected-sigma trajectory begin to saturate when continued to 192 generations?

## Fixed workload

The installed-wheel typed Python host runs the same deterministic nonlinear control workload:

```text
policy        = Linear(6 -> 8) -> ReLU -> Linear(8 -> 2)
parameters    = 74
population    = 8
sigma         = 0.08
learning_rate = 0.06
seed          = 20_260_917
horizon       = 16
```

Environment dynamics, four scenarios, reward/cost terms, action clamp, graph/binding semantics, ProgramBundle semantics, and optimizer algorithm remain unchanged.

## Continuous trajectory design

A single OpenES instance runs for 192 generations.

Evidence milestones:

| Generation | Candidate evaluations |
| ---: | ---: |
| 48 | 384 |
| 96 | 768 |
| 144 | 1152 |
| 192 | 1536 |

Using one trajectory means later milestones continue the exact earlier optimizer state instead of paying for independent restarts.

## Required evidence

The report records:

- zero-policy and initial champion reward;
- champion reward at every milestone;
- full generation-best and champion histories;
- exact candidate-evaluation count;
- champion improvement generations/count;
- per-interval improvement counts for 1–48, 49–96, 97–144, 145–192;
- longest champion plateau overall and per interval;
- first/last improvement generation;
- NO_IMPROVEMENT counts overall and per interval;
- median ask/apply/rollout/tell timing and total wall time;
- finite fitness and stable batch cardinality;
- stable program and binding identities.

The final 192-generation champion must replay exact trainable state through stateful ProgramBundle and reproduce reward within 1e-9.

## Interpretation

No reward or timing number is a brittle CI threshold.

- any champion improvement in generations 145–192 means depth saturation is not fully observed at 192;
- no champion improvement in the final interval is a saturation signal worth investigating;
- a lower final-interval improvement count than earlier intervals is descriptive evidence of diminishing returns, not by itself an optimizer failure;
- semantic/cardinality/replay failure invalidates search interpretation.

## Scope

Exactly three research-only files:

- `.github/workflows/python-nonlinear-es-depth-saturation-research.yml`
- `docs/python-nonlinear-es-depth-saturation-research.md`
- `scripts/research_python_nonlinear_es_depth_saturation.py`

No production Rust, ABI/facade, OpenES algorithm/adaptive schedule, graph/binding semantics, model/activation surface, Linear runtime, Math v1-v9, ProgramBundle schema, Resolution/Authorization, or support-matrix change.

## Observed evidence

The dedicated installed-wheel run passed all semantic, cardinality, identity, and final checkpoint-replay proofs.

| Milestone | Evaluations | Champion reward | Interval improvements | Interval longest plateau |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 384 | -0.645007 | 17 | 7 gen |
| 96 | 768 | -0.390989 | 16 | 4 gen |
| 144 | 1152 | -0.324997 | 13 | 10 gen |
| 192 | 1536 | **-0.309271** | **8** | **14 gen** |

Higher reward (less negative) is better.

The trajectory continues to improve after generation 96:

```text
48  -> 96   cost magnitude reduction ≈ 39.38%
96  -> 144  cost magnitude reduction ≈ 16.88%
144 -> 192  cost magnitude reduction ≈  4.84%
```

The improvement count also declines by interval:

```text
1–48    : 17 champion improvements
49–96   : 16
97–144  : 13
145–192 : 8
```

This is a clear diminishing-return pattern, but not full saturation. The final interval still produces eight new champions, with the last improvement at generation 178. The final 14 generations form the longest observed plateau.

The report therefore classifies the run as:

```text
DEPTH_SATURATION_NOT_OBSERVED_AT_192
DEPTH_RETURNS_DIMINISHING_BUT_ACTIVE
```

The final champion reward is `-0.3092712352204114`, corresponding to an ~87.15% cost reduction versus the zero-policy baseline `-2.407589173214531`.

The final state replays exactly through stateful `ProgramBundle`, including exact parameter state and reward replay within `1e-9`.

Median candidate rollout remains the dominant cost (~35.51 ms). Median parameter apply (~0.221 ms), `ask_f32` (~0.057 ms), and `tell` (~0.117 ms) remain secondary. Total training wall time is ~63.2 s for 1536 candidate evaluations.

## Decision

```text
KEEP_OPENES_ALGORITHM_UNCHANGED
KEEP_SIGMA_0_08_FOR_THIS_RESEARCH_LINEAGE
DEPTH_RETURNS_DIMINISHING_BUT_ACTIVE
DEPTH_SATURATION_NOT_OBSERVED_AT_192
DO_NOT_WIDEN_MODEL_OR_ACTIVATION_SURFACE
MEASURE_ONE_DEEPER_CONTINUATION_BEFORE_ALGORITHM_CHANGE
```

The next evidence slice should extend the same fixed-population, fixed-sigma lineage one step further before considering adaptive sigma, annealing, or optimizer-algorithm changes.
