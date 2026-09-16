# Python optimizer ask-to-apply transport research

Status: **measurement slice; product API unchanged**

Related: #196, #199, #200, #203

## Question

After #200, `host.GraphParameterBinding.apply_flat(...)` can borrow a compatible native-f32 buffer for one ABI call, but `host.EsOptimizer.ask()` still materializes its population as a Python `list[float]`.

The remaining host path is therefore:

```text
EsOptimizer.ask()
    -> Python list[population * dim]
    -> candidate extraction
    -> GraphParameterBinding.apply_flat(...)
```

This research asks whether candidate extraction/materialization is now large enough to justify another Python product boundary. It does **not** assume that a new optimizer return type, facade helper, or ABI batch primitive is needed.

## Installed-wheel methodology

`scripts/research_python_optimizer_ask_apply.py` builds the current wheel, installs it into a fresh external virtual environment, removes repository `PYTHONPATH`, and runs the supported `burn_research_ffi.host` surface from `site-packages`.

The large transport case uses:

```text
16 x Linear(64 -> 64, bias=true)
parameter dim = 66,560
population    = 4
```

Three host-side paths are compared with independent optimizers using the same strategy, seed, population, sigma, and learning rate:

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
  -> one array('f', whole population)
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
ask
 -> candidate extraction
 -> apply
 -> graph objective over fixed rows
 -> tell
```

It records ask/slice/conversion/view/apply/objective/tell/full-generation timing separately. Timing remains evidence only.

## Semantic proof boundary

The research fails if transport variants change semantics. It proves:

- optimizer population cardinality is unchanged;
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

Pending the first exact-head GitHub Actions run. Raw timing fields and the resulting decision will be added here only after the installed-wheel research workflow completes successfully.

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
