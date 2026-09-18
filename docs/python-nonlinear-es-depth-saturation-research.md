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

## Status

Candidate research slice. Evidence will be recorded only after the exact PR head produces a successful report artifact and all regression gates remain green.
