# Runtime architecture artifact proof

Issue: #279

## Purpose

This proof session exercises the architecture boundaries identified in the post-growth code/documentation audit through supported/public consumer surfaces.

It is not a feature or optimization slice.

## Consumer surfaces

The proof has two independent runtime consumers:

1. a fresh installed Python wheel in a temporary venv outside the repository checkout;
2. an external Cargo project depending on the actual packaged burn-research .crate.

Repository-private APIs are not the proof surface.

## Artifact set

### Python installed-wheel artifact

The Python consumer proves:

- structural program and binding identities remain stable across learned-weight mutation;
- a structurally mismatched registry is rejected by the already-compiled graph;
- GraphParameterBinding valid apply works;
- wrong-length and non-finite candidates are rejected without partial learned-state mutation;
- truncated ProgramBundle import into an already-populated target registry does not replace/corrupt that target;
- valid stateful ProgramBundle replay preserves graph identity, binding identity, learned parameters, and output;
- invalid tell does not consume the pending optimizer batch and a corrected retry succeeds;
- ask called twice before tell replaces the pending batch and advances candidate generation deterministically;
- state-preserving learning-rate mutation leaves the next candidate batch identical and affects trajectory only after the next tell.

### External packaged-Rust artifact

The packaged Rust consumer proves:

- Math Program v9 causal additive masking is composed from v8 arithmetic + indicesLike + comparison;
- v9 plan, identity, and output replay exactly through from_plan;
- ProgramBundle replays graph/learned state;
- ProgramBundle does not serialize OpenES generation/search trajectory state;
- ProgramBundle does not serialize AgentWorkspace metadata.

## Output

The workflow uploads:

runtime-architecture-artifact-report.json

Schema:

burn-research.runtime-architecture-artifact-proof.v1

## Boundary interpretation

A PASS supports these architecture statements for the exact tested commit:

- structure != mutable learned state;
- learned state != program/binding identity;
- ProgramBundle checkpoint != optimizer state;
- ProgramBundle checkpoint != AgentWorkspace state;
- optimizer control capability != host scheduling policy;
- Math v9 composition is replayable through the public packaged Rust surface.

## Scope

Exactly three proof files:

- .github/workflows/runtime-architecture-artifact-proof.yml
- scripts/audit_runtime_architecture_artifacts.py
- docs/runtime-architecture-artifact-proof.md

No production Rust, ABI, Python facade, graph, binding, optimizer, Math Program, ProgramBundle, Workspace, Resolution/Authorization, or support-matrix change.

## Observed evidence

The first integrated workflow run completed successfully on both supported/public external-consumer surfaces.

### Installed Python wheel

All runtime checks passed:

| Boundary | Observed result |
| --- | --- |
| program identity across weight mutation | stable |
| binding identity across weight mutation | stable |
| structurally mismatched registry | rejected |
| valid flat candidate | applied |
| wrong-length flat candidate | rejected with exact prior state preserved |
| non-finite flat candidate | rejected with exact prior state preserved |
| truncated ProgramBundle into populated target | rejected; target remained usable and unchanged |
| valid ProgramBundle replay | identity, binding, parameters, and output exact |
| invalid tell retry | pending batch preserved; corrected tell succeeded |
| ask -> ask before tell | second batch replaced the first pending batch |
| replacement RNG semantics | second batch matched the deterministic next RNG batch |
| learning-rate mutation | next batch remained identical |
| learning-rate mutation while pending | rejected |
| post-tell LR effect | subsequent candidate trajectory diverged |

The stateful ProgramBundle payload in this run was 456 bytes for the small proof graph.

### External packaged Rust consumer

The actual packaged `.crate` was consumed from a separate Cargo project.

The consumer proved:

- Math Program v9 causal additive masking with exactly 8 composed steps;
- exact v9 plan replay;
- exact v9 program identity replay;
- exact v9 output replay;
- exact ProgramBundle graph/learned-state replay;
- original optimizer generation = 1 while a freshly constructed optimizer after model replay starts at generation 0;
- original AgentWorkspace retains its marker while a fresh workspace after model replay has no marker.

Therefore the runtime evidence supports:

```text
STRUCTURE != MUTABLE_LEARNED_STATE
LEARNED_STATE != PROGRAM_OR_BINDING_IDENTITY
PROGRAM_BUNDLE != OPTIMIZER_SEARCH_STATE
PROGRAM_BUNDLE != AGENT_WORKSPACE_STATE
OPTIMIZER_CONTROL_CAPABILITY != HOST_SCHEDULE_POLICY
MATH_V9_COMPOSITION_REPLAYS_THROUGH_PACKAGED_RUST
```

### Provenance note

GitHub pull-request checkout uses an ephemeral merge ref by default. The initial report exposed only the checkout commit, which is insufficient as an exact source-branch provenance field even though the runtime semantics were valid.

The harness now records both:

- `source_head_sha`: exact PR source head supplied by the workflow event;
- `checkout_commit`: the actual commit executed by GitHub Actions.

The final-head rerun must pass with this unambiguous provenance before merge.
