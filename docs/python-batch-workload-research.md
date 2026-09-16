# Python batch workload research

Status: **research candidate pending CI evidence**

Related: #184, #186, #188, #189

## Question

The Python wheel and typed facade are already supported on the narrow verified matrix. The next question is not whether Python can call the runtime; it is whether the current copy-based tensor boundary creates enough practical pressure to justify widening the ABI or adding NumPy/zero-copy integration.

This research therefore compares two host evaluation patterns over the same Linear graph and dataset:

```text
row-by-row
  N x tensor create -> graph.run -> output copy

batched
  1 x Tensor[N,F,1,1] -> graph.run -> output[N,1,1,1] copy
```

Linear already treats axis 0 as batch and axis 1 as feature width, so this comparison does not introduce new execution semantics.

## Boundary

Preserve:

```text
Rust core semantics
    != language-neutral ABI v1
    != typed Python facade
    != host evaluation policy
    != performance evidence
```

The research must not add or alter `br_v1_*` symbols, Math Program semantics, graph semantics, optimizer semantics, ProgramBundle format, or Resolution/Authorization.

No NumPy, DLPack, buffer-protocol, zero-copy, PyO3, Go, or C++ work belongs in this slice.

## Workload

The research script builds the current CFFI wheel, installs it into a fresh temporary venv outside the checkout, and runs only through the typed facade.

The deterministic workload uses:

- rank-4 f32 tensors;
- a feature-width-8 `Linear(8 -> 1, bias)` graph;
- 96 training rows and 32 held-out rows;
- `GraphParameterBinding.total_len` as the sole optimizer dimension source;
- strict ES with fixed seed/population/generations;
- a Python-owned regression objective;
- one persistent batch tensor reused across candidate evaluations;
- stateful `ProgramBundle` export/import into a fresh registry after training.

The row and batch paths must produce equivalent outputs within f32 tolerance.

## Measurements

Timing is separated into:

- tensor creation;
- graph execution;
- output copy;
- ES ask;
- canonical parameter apply;
- Python objective arithmetic;
- ES tell;
- checkpoint export/import/replay.

The workflow uploads a machine-readable JSON report.

These timings are **evidence only**. They are not CI performance thresholds, SLAs, or portable benchmark claims. GitHub runner noise and host scheduling make absolute times unsuitable for a stable contract.

The useful comparison is primarily within one run: row-by-row versus batched evaluation under the same machine/process/runtime.

## Semantic gates

The workflow may fail only on semantic/proof conditions, not timing targets:

1. installed package must come from `site-packages`, not the repository checkout;
2. row and batch outputs must agree;
3. candidates, fitness, and outputs must remain finite;
4. ES must materially improve deterministic held-out loss over the zero-vector baseline;
5. program and binding identities must remain stable while mutable parameters change;
6. best parameters must read back exactly in canonical order;
7. ProgramBundle replay must preserve program identity, binding identity, learned state, and held-out output/loss.

## Decision rule after evidence

Do not infer a need for zero-copy from row-wise overhead alone.

If batching and persistent input tensors reduce crossing/copy cost enough that output copy is not the dominant batched candidate-evaluation cost, keep the current copy-based boundary.

If output transfer remains materially dominant even in the batched/persistent-tensor path, open a separate design issue for the minimum Python tensor interoperability improvement and prove that proposal against this workload before changing ABI v1.
