## Purpose

<!-- What concrete problem does this PR solve? What result should exist after merge? -->

## Architectural context

<!-- Where does this change sit in the current architecture? Link related issues/PRs/docs and explain the surrounding boundary. -->

## Design and ownership

<!-- Describe the design. State which component owns the new behavior/state before and after this PR. Include a small flow/diagram when it makes the boundary clearer. -->

## Invariants / proof boundary

<!-- What must remain true? Which invariants are newly proved or strengthened? Keep old proofs slice-local; do not forbid future capability unless that is the actual protocol contract. -->

## Identity / state / authority impact

<!-- Explain any effect on structural identity, parameter-binding identity, mutable state, checkpoint/replay, provenance, Resolution/Authorization, or promotion authority. Write `none` when unaffected. -->

## Non-goals

<!-- Explicitly name nearby work that this PR does NOT attempt. This prevents later readers from inferring accidental scope. -->

## Validation and evidence

<!-- CI gates, research workload, tests, artifact audit, benchmarks, or replay evidence that justify the change. Distinguish measured evidence from assumptions. -->

## Continuation guidance

<!-- If another contributor/agent continues this work, what is the next safe frontier? What should they not reinterpret or duplicate? -->

## Surface-change checklist

- [ ] Runtime behavior changes
- [ ] Public WASM surface changes
- [ ] Math Program semantics/version changes
- [ ] Graph-plan semantics changes
- [ ] Optimizer algorithm/state changes
- [ ] Checkpoint/state identity changes
- [ ] Resolution/Authorization semantics changes

<!-- For every checked item, explain the migration/proof consequence above. Leaving all boxes unchecked means the PR is intentionally surface-neutral. -->
