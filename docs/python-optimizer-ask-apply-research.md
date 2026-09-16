# Python optimizer ask-to-apply transport research

Status: **measurement slice; product API unchanged**

Related: #196, #199, #200, #203

## Question

After #200, `host.GraphParameterBinding.apply_flat(...)` can borrow a compatible native-f32 buffer for one ABI call, but `host.EsOptimizer.ask()` still materializes its candidate batch as a Python `list[float]`.

The remaining host path is therefore:

```text
EsOptimizer.ask()
    -> Python list[batch_size * dim]
    -> candidate extraction
    -> GraphParameterBinding.apply_flat(...)
```

This research asks whether candidate extraction/materialization is now large enough to justify another Python product boundary. It does **not** assume that a new optimizer return type, facade helper, or ABI batch primitive is needed.

## Optimizer lifecycle and cardinality contract

The optimizer constructor accepts a `population` configuration, but consumers must not infer returned candidate cardinality from that request. The observed and now explicitly proven lifecycle contract is:

```text
idle:
  optimizer.batch_size == 0

ask:
  batch = optimizer.ask()
  batch_size = optimizer.batch_size
  batch_size > 0
  len(batch) == batch_size * binding.total_len

consume/evaluate:
  apply exactly batch_size candidates

tell:
  len(fitness) == batch_size
  optimizer.tell(fitness)
  optimizer.batch_size == 0
```

The first harness run correctly failed before producing evidence because it assumed `population=4` implied `batch_size=4`. The second harness run also correctly failed before evidence because it read `optimizer.batch_size` before `ask()`, when the optimizer is idle and reports zero.

The benchmark now follows the lifecycle directly: `ask -> read batch_size -> validate returned length -> apply/evaluate -> tell -> assert idle`. This is a harness correction and proof strengthening, not a product or optimizer semantic change.

## Installed-wheel methodology

`scripts/research_python_optimizer_ask_apply.py` builds the current wheel, installs it into a fresh external virtual environment, removes repository `PYTHONPATH`, and runs the supported `burn_research_ffi.host` surface from `site-packages`.

The large transport case uses:

```text
16 x Linear(64 -> 64, bias=true)
parameter dim        = 66,560
requested population = 4
candidate count       = optimizer.batch_size after ask()
```

Three host-side paths are compared with independent optimizers using the same strategy, seed, requested population, sigma, and learning rate:

```text
A. current baseline
ask() list
  -> per-candidate list slice
  -> apply_flat(list)

B. per-candidate conversion
ask() list
  -> per-candidate list slice
  -> array('f', candidate)
  -> apply_flat(f32 buffer)

C. whole-batch conversion
ask() list
  -> one array('f', whole returned batch)
  -> contiguous memoryview candidate windows
  -> apply_flat(f32 view)
```

The script records separately:

- `ask()` time;
- list-slice time;
- list-to-f32 conversion time;
- memoryview-window creation time;
- `apply_flat()` time;
- post-ask transport time;
- combined ask + transport time.

A fixed-candidate control also records list-fallback versus f32-buffer `apply_flat()` cost so candidate extraction/conversion can be interpreted separately from the already-proven #200 transport fast path.

## Objective-loop control

A small deterministic `Linear(8 -> 1)` workload runs all three variants through complete optimizer generations:

```text
idle batch_size == 0
 -> ask
 -> read positive batch_size
 -> candidate extraction
 -> apply
 -> graph objective over fixed rows
 -> tell(batch_size fitness values)
 -> idle batch_size == 0
```

It records ask/slice/conversion/view/apply/objective/tell/full-generation timing separately. Timing remains evidence only.

## Semantic proof boundary

The research fails if transport variants change semantics. It proves:

- optimizer is idle with `batch_size == 0` before each `ask()`;
- `optimizer.batch_size` is read only after `ask()` and is then the candidate-cardinality authority;
- `ask()` length is exactly `batch_size * dim`;
- batch size is stable across generations and same-config transport variants;
- `tell()` consumes exactly `batch_size` fitness values and returns the optimizer to idle `batch_size == 0`;
- same-seed candidate sequences have identical f32 digests across variants;
- per-candidate and whole-batch f32 conversion preserve the already-f32 candidate values;
- memoryview candidate windows remain one-dimensional, C-contiguous, native `f`, and exact length;
- graph program identity remains stable;
- binding identity remains stable;
- non-finite f32 candidates still receive the existing atomic core rejection;
- the deterministic objective produces identical fitness history across variants;
- a stateful `ProgramBundle` round-trip preserves program identity, binding identity, and final learned state.

The benchmark never treats timing as a correctness condition.

## Interpretation rule

The JSON report always uses `verdict = PASS` only for semantic/proof success. A separate research `decision` interprets timing.

For this first slice, `PYTHON_BUFFER_RETURN_WORTH_PROTOTYPING` is emitted only if the current whole-batch list-to-`array('f')` path still improves median post-ask transport by at least 20% versus the current list-slice/list-apply path. Otherwise the report emits `KEEP_CURRENT_API`.

That 20% value is **not** a CI threshold and does not make timing a support guarantee. It is only a conservative research trigger: if a buffer path wins even while paying the extra list-to-array conversion that a future direct buffer return could avoid, a narrow Python-only optimizer-buffer prototype becomes worth measuring separately.

This research alone does not justify `BATCH_BOUNDARY_WORTH_PROTOTYPING`; no new native batch primitive is being measured here.

## Current result

Two harness attempts produced **no timing evidence** and therefore no product conclusion:

1. requested `population` was incorrectly treated as returned `batch_size`;
2. `batch_size` was then read before `ask()`, while the optimizer was idle and correctly reported zero.

Both assumptions have been removed. The current harness follows the proven lifecycle and will only emit a timing report after all lifecycle, semantic-equivalence, identity, atomicity, and replay checks pass.

Raw timing fields and the resulting decision will be added here only after the corrected exact-head installed-wheel research workflow completes successfully and uploads its report.

## Non-goals

This slice does not change:

- `EsOptimizer.ask()` public return type;
- any `br_v1_*` ABI symbol;
- Rust optimizer algorithms;
- core `GraphParameterBinding` semantics;
- graph execution;
- Math Program v1-v9;
- `ProgramBundle` schema;
- Resolution/Authorization;
- Python support matrix.

It adds no NumPy, DLPack, PyO3, persistent CFFI pointer cache, or third-party array dependency.
