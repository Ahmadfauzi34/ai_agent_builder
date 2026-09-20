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

## Status

Candidate proof harness. Observed evidence will be recorded only from the workflow artifact produced by the exact PR head.
