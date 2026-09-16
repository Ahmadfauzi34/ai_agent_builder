# Python optimizer ask-to-apply transport research

Status: **measured; product API unchanged**

Related: #196, #199, #200, #203, #204

## Question

After #200, `host.GraphParameterBinding.apply_flat(...)` can borrow a compatible native-f32 buffer for one ABI call, but `host.EsOptimizer.ask()` still materializes its candidate batch as a Python `list[float]`.

The remaining host path is therefore:

```text
EsOptimizer.ask()
    -> Python list[batch_size * dim]
    -> candidate extraction
    -> GraphParameterBinding.apply_flat(...)
```

This research asks whether candidate extraction/materialization is large enough to justify another Python product boundary. It does **not** assume that a new optimizer return type, facade helper, or ABI batch primitive is needed.

## Optimizer lifecycle and cardinality contract

The optimizer constructor accepts a `population` configuration, but consumers must not infer returned candidate cardinality from that request. The core implementation defines `batch_size()` as `last_candidates.len()`, while `awaiting_fitness` separately tracks whether `tell()` is currently legal.

The contract used by this research is:

```text
new optimizer:
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
  optimizer.batch_size == batch_size

next ask:
  may replace the previous last_candidates batch
```

Three earlier harness attempts produced no timing evidence while this lifecycle was being made explicit:

1. requested `population=4` was incorrectly treated as returned `batch_size=4`;
2. `batch_size` was then read before the first `ask()`, when `last_candidates` is empty and the value is correctly zero;
3. the harness then incorrectly expected `tell()` to clear `last_candidates`, but the implementation intentionally keeps the last candidate batch while only clearing the internal `awaiting_fitness` flag.

Those runs were harness discoveries only and were not used as performance evidence.

## Installed-wheel methodology

`scripts/research_python_optimizer_ask_apply.py` builds the current wheel, installs it into a fresh external virtual environment, removes repository `PYTHONPATH`, and runs the supported `burn_research_ffi.host` surface from `site-packages`.

The large transport case uses:

```text
16 x Linear(64 -> 64, bias=true)
parameter dim        = 66,560
requested population = 4
observed batch_size   = 4
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

The script records separately `ask()`, list slicing, f32 conversion, memoryview-window creation, `apply_flat()`, post-ask transport, and combined ask+transport time. Timing is evidence only and never a CI threshold.

## Valid measurement

The first valid exact-head report came from Python Optimizer Ask Apply Research run #7 on head `a66ab14b423df6568f4ba64082c6dbd79fe453d3` and reported `verdict = PASS`.

Large-case medians:

| Path / component | Median |
| --- | ---: |
| `ask()` — list baseline | 35.539 ms |
| list slice per candidate | 0.183 ms |
| list `apply_flat()` per candidate | 5.513 ms |
| baseline post-ask transport | 23.746 ms |
| baseline ask + transport | 67.808 ms |
| per-candidate `array('f')` conversion | 1.281 ms |
| per-candidate buffer apply | 2.905 ms |
| per-candidate-array post-ask transport | 18.525 ms |
| whole-batch list -> `array('f')` conversion | 5.112 ms |
| memoryview candidate window | 0.0010 ms |
| whole-batch-view buffer apply per candidate | 2.871 ms |
| whole-batch-view post-ask transport | 16.944 ms |
| whole-batch-view ask + transport | 60.499 ms |

Fixed-candidate control at the same 66,560-parameter dimension:

```text
list fallback apply median = 5.406 ms
f32 buffer apply median    = 2.863 ms
list / buffer ratio        = 1.888x
```

Derived results:

```text
per-candidate array post-ask speedup = 1.282x
whole-batch buffer post-ask speedup  = 1.402x
list-slice share of slice+apply      = 3.22%
```

So list slicing itself is **not** the primary remaining problem. The material cost is the list-based apply/marshalling path. The whole-batch buffer variant reduced post-ask transport from 23.746 ms to 16.944 ms, about **28.6%**, even though it still paid about 5.112 ms to convert the already-materialized Python list into `array('f')`.

The full `ask + transport` median improved from 67.808 ms to 60.499 ms, about **10.8%**. This also shows that optimizer `ask()` work itself remains the largest single component in this workload; transport optimization should remain narrow rather than expanding into a new native batch execution model.

## Objective-loop control

A small deterministic `Linear(8 -> 1)` workload ran all three variants through complete optimizer generations:

```text
initial batch_size == 0
 -> ask
 -> read positive batch_size
 -> candidate extraction
 -> apply
 -> graph objective over fixed rows
 -> tell(batch_size fitness values)
 -> batch_size still reflects the last candidate batch
```

Median full-generation times were approximately:

```text
list slice                 9.238 ms
per-candidate array        9.255 ms
whole-batch buffer view    9.207 ms
```

At only 9 parameters, transport choice is effectively irrelevant relative to the objective work. This reinforces that the fast path matters primarily at larger parameter dimensions rather than justifying a universal API expansion.

## Semantic proof boundary

The valid report proved:

- a new optimizer begins with `batch_size == 0` because `last_candidates` is empty;
- `optimizer.batch_size` after `ask()` is the candidate-cardinality authority;
- `ask()` length is exactly `batch_size * dim`;
- batch size is stable across generations and same-config transport variants;
- `tell()` consumes exactly `batch_size` fitness values and does not erase the last candidate batch;
- same-seed candidate sequences have identical f32 digests across variants;
- per-candidate and whole-batch f32 conversion preserve candidate values at the binding boundary;
- memoryview candidate windows remain one-dimensional, C-contiguous, native `f`, and exact length;
- graph program identity remains stable;
- binding identity remains stable;
- non-finite f32 candidates still receive the existing atomic core rejection;
- the deterministic objective produces identical fitness history across variants;
- stateful `ProgramBundle` round-trip preserves program identity, binding identity, and final learned state.

## Decision

The report decision is:

```text
PYTHON_BUFFER_RETURN_WORTH_PROTOTYPING
```

Reason: the whole-batch buffer path improves median post-ask transport by about 28.6% **despite** paying an extra list -> `array('f')` conversion. That is enough evidence to justify a separate **Python-only prototype** that asks whether `EsOptimizer` can expose the already-returned native f32 batch through a typed buffer surface without first materializing a Python list.

The next prototype should preserve current `ask() -> list[float]` compatibility and should first test a separate method/helper such as a buffer-return path. It should reuse the existing ABI v1 f32-buffer handle/copy machinery if possible and should not widen the native ABI unless a separate proof demonstrates that Python-only transport cannot solve the measured cost.

This research does **not** justify:

- changing the existing `EsOptimizer.ask()` return type;
- an ABI batch primitive;
- a graph-owned controller;
- a core binding cache;
- NumPy/DLPack/PyO3 dependencies.

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
