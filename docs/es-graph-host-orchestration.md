# ES graph host-orchestration boundary

Status: **accepted baseline**

Related work: #174, #176, #177, #178, #258, #275

## Why this document exists

The repository now has enough graph, parameter-binding, optimizer, and checkpoint capability that a future contributor could reasonably ask where an ES training/evaluation controller should live.

Earlier development comments suggested that the internal Rust `Objective` trait might eventually become the place where a Burn graph is executed. The packaged research path changed that conclusion. The current evidence supports keeping graph-training orchestration at the **host boundary**, while keeping deterministic structural and numerical invariants in the Rust/WASM core.

This document records that decision so future work does not need to reconstruct the intent from historical conversations or infer architecture from an old comment.

This is an evidence-based baseline, not a permanent prohibition. A core controller may be proposed later if a concrete invariant cannot be enforced compositionally at the host boundary.

## Decision

Use the existing proof-bounded core primitives as independent capabilities and compose objective policy in the host:

```text
                    Rust / WASM core

  EsOptimizer          GraphParameterBinding        CompiledGraph
  ask / tell           canonical coordinates        structural run
       |                         |                        |
       +-------------------------+------------------------+
                                 |
                                 v
                           host orchestration

                    dataset / reward / penalty
                    evaluation scheduling
                    candidate evaluation policy
                    experiment policy
                                 |
                                 v
                 ProgramBundle(include_state=true)
                         checkpoint / replay
```

The host owns the policy that answers **what is good**. The core owns the invariants that answer **what is the graph**, **which parameter coordinate is which**, **whether a candidate may be applied**, and **how optimizer state transitions occur**.

## Component ownership

### `EsOptimizer`

Owns:

- optimizer state;
- candidate generation through `ask()`;
- the strict `ask -> tell` lifecycle;
- candidate dimension supplied at construction;
- validation that submitted fitness values are finite and have the expected cardinality;
- optimizer-specific update rules and diagnostics;
- validation and state-continuity semantics for supported in-place optimizer controls such as OpenES learning-rate mutation.

Does not own:

- a graph;
- a `LayerRegistry`;
- parameter ordering;
- dataset traversal;
- an objective/reward function;
- checkpoint promotion authority;
- automatic learning-rate scheduling or policy for when an optimizer control should be changed.

The internal `src/es/objective.rs` trait remains a plain-Rust proof-of-life/test abstraction. It is not the required integration point for graph execution.

### `GraphParameterBinding`

Owns:

- canonical graph-level trainable owner discovery;
- unique first-use ordering from the graph plan;
- offsets, lengths, and weight layouts;
- graph-parameter binding identity;
- exact flat reads;
- full prevalidation before mutation;
- finite-only candidate application;
- fail-closed behavior for unsupported parameterized layers.

Does not own:

- optimizer state;
- fitness/reward semantics;
- controller policy;
- graph execution scheduling.

### `CompiledGraph`

Owns:

- the canonical graph plan;
- structural program identity;
- validation that the current registry still matches the compiled structural binding;
- graph execution.

Mutable trainable values may change while the structural identity remains stable. A structural registry change requires recompilation/revalidation rather than silently changing the compiled graph meaning.

`CompiledGraph` does not become an optimizer or controller.

### Host orchestration

The host composes the capabilities above and owns policy such as:

- training/evaluation dataset selection;
- number and order of graph evaluations per candidate;
- regression/classification/task loss;
- penalties and regularization;
- held-out evaluation;
- reward shaping;
- generation limits and stopping policy;
- candidate promotion policy;
- experiment bookkeeping;
- policy for whether and when to invoke supported state-preserving optimizer controls.

The canonical lifecycle is:

```text
graphParameterLayout.total_len
    -> EsOptimizer.strict(...)
    -> ask
    -> slice candidates only by canonical layout dimension
    -> setGraphParametersFlat
    -> CompiledGraph.run one or more times
    -> host computes finite objective / fitness
    -> tell
    -> repeat
    -> best
    -> setGraphParametersFlat(best)
    -> exportProgramBundle(include_state=true)
```

The host must not invent a second parameter ordering or reorder coordinates before applying them.

## State-preserving optimizer control

Current main exposes a narrow state-preserving OpenES learning-rate control through the supported surfaces.

The ownership split is:

```text
core capability
    set_learning_rate(lr)
    validate OpenES-only / finite-positive / generation-boundary use
    preserve mean / RNG / generation / lifetime best / stagnation / sigma
    reject mutation while an ask batch is pending

host policy
    decide whether a learning-rate change is useful
    decide when to request it
    choose fixed or scheduled values for a workload
    record and interpret experiment evidence
```

This capability does **not** move scheduling policy into the optimizer. The core owns the validity and continuity of the state mutation; the host owns the decision to request the mutation.

The continuity proof for #275 demonstrates the intended boundary: two identical seeded optimizers remain identical through a completed generation; changing learning rate on one leaves its current state and next candidate batch unchanged; trajectory divergence is allowed only after the identical next batch is consumed by `tell()` with a different update scale.

Therefore:

```text
STATE_PRESERVING_CONTROL_CAPABILITY
    !=
AUTOMATIC_ADAPTIVE_POLICY
```

A future automatic schedule or adaptive optimizer rule is a separate semantic proposal and requires its own evidence and proof boundary.

## Why the controller is not in the core today

Research #174 proved that the packaged host can execute the complete lifecycle:

```text
ask -> apply -> graph.run(s) -> objective -> tell
```

without adding a graph-owning optimizer facade. Candidate-apply overhead was strongly amortized when an objective performs multiple graph forwards, so a public reusable binding/cache abstraction was not justified by that workload.

Research #177 then used the same optimizer, graph, binding, and checkpoint primitives with materially different host objective policies. A pure MSE objective and an MSE-plus-L2 policy produced different best candidates while preserving the same core contracts. No new invariant was discovered that required Rust to own the objective policy.

Therefore the current decision is:

```text
KEEP_HOST_ORCHESTRATION
```

Moving orchestration into Rust/WASM now would add ownership, mutability, or callback coupling without closing a demonstrated correctness gap.

## Checkpoint and replay boundary

Research #176 established that the existing stateful Program Bundle is sufficient as the learned graph checkpoint/replay artifact:

```text
trained graph + registry
    -> exportProgramBundle(include_state=true)
    -> fresh LayerRegistry
    -> importProgramBundle
    -> equivalent structural identity
    -> equivalent binding identity/layout
    -> exact flat learned parameters
    -> equivalent graph outputs/loss
```

Malformed/truncated import is staged and fail-closed against an already populated target registry.

The architectural decision is:

```text
KEEP_EXISTING_STATEFUL_PROGRAM_BUNDLE
```

Do not introduce a second checkpoint schema merely because an optimizer/controller exists.

## Identity boundaries

Keep the following concepts separate:

```text
I_program  = structural graph/program identity
I_binding  = canonical trainable-owner/layout identity
mutable parameter values = learned state, not structural identity
ProgramBundle = transport/checkpoint artifact
bundle SHA = evidence about particular serialized bytes, not protocol identity
Authorization/Proof = separate authority plane
```

In particular, **ProgramBundle byte SHA must not be used as state identity, promotion authority, or proof authority**.

Burn state serialization can contain internal tensor identifiers that differ between fresh processes even when the reconstructed learned graph is semantically equivalent. Within a loaded process, repeated export/re-export can still be exact. The semantic replay invariants are the contract; cross-process byte equality is not.

## What stays in the core

Host orchestration does not mean moving safety checks out of the core. These invariants remain core responsibilities:

- canonical parameter coordinate order;
- exact candidate dimension;
- finite-only graph parameter values;
- full prevalidation before first graph-parameter mutation;
- structural graph/registry binding validation;
- strict optimizer lifecycle/cardinality/finite-fitness validation;
- staged/atomic ProgramBundle import;
- deterministic structural identity semantics.

A host may choose policy, but it should not be able to redefine these invariants.

## Non-goals of this decision

This architecture record does not add or require:

- a `GraphController` in Rust;
- a graph-owning `EsOptimizer`;
- JS callbacks into Rust for arbitrary objective functions;
- a new public WASM controller surface;
- a reusable graph-parameter cache;
- a new checkpoint format;
- a new state identity;
- autodiff/backpropagation;
- Math Program changes;
- graph-plan changes;
- Resolution/Authorization coupling;
- automatic learning-rate schedules or other host-policy promotion into the core.

It also does not require every host to copy Node implementation details. Node, the native Rust package, and the verified Python host are supported surfaces with host-appropriate integration code; any future host should preserve the same semantic lifecycle rather than copying one host's implementation details.

## When a core controller may be reconsidered

Do not add a core controller only to reduce host glue or because an old comment implied one was planned.

A proposal should identify at least one concrete invariant that cannot be reliably enforced with the existing composition. Examples that could justify reconsideration include:

- an unavoidable partial-mutation window between candidate apply and evaluation;
- ambiguous candidate ownership/order that cannot be resolved by `GraphParameterBinding`;
- an optimizer lifecycle race that host composition cannot close;
- a required atomic transaction spanning optimizer state and graph state;
- a demonstrated cross-host semantic mismatch that a shared core controller would eliminate;
- a measured performance problem whose solution genuinely requires persistent core ownership rather than an internal optimization hidden behind existing APIs.

If such a gap appears, the proposal should state:

1. the missing invariant;
2. the failing workload/proof;
3. why existing boundaries cannot enforce it;
4. the minimum new ownership introduced;
5. how old slice-local proofs remain valid;
6. how checkpoint, identity, and authorization boundaries remain separate.

## Continuation guide

For future ES + graph work, prefer this order:

```text
1. define the workload and invariant
2. use the existing canonical graph-parameter layout
3. keep objective/reward policy in the host
4. prove ask -> apply -> run -> objective -> tell
5. if using a state-preserving optimizer control, mutate only at its proven lifecycle boundary and prove continuity
6. prove best-state replay through ProgramBundle
7. measure before adding cache/controller abstractions
8. add a core abstraction only for a demonstrated missing invariant
```

When a PR changes one of these boundaries, its description should explicitly document:

- purpose and problem being solved;
- architectural context;
- ownership before and after the change;
- invariants proved by the PR;
- non-goals and unchanged boundaries;
- migration/continuation guidance;
- evidence or CI/research gates used to justify the decision.

That makes the PR itself useful as an architecture record while this document preserves the accepted result in the repository after the PR is merged.
