# Python optimizer direct-f32 prototype research

Status: **prototype measurement; supported product API unchanged**

Related: #200, #203, #204, #205, #206

## Question

#204 established that the remaining large-parameter Python optimizer transport cost is not primarily list slicing. At 66,560 parameters and observed batch size 4, the whole-batch f32-buffer transport path reduced median post-ask transport by about 28.6% even though it still paid an extra Python list -> `array('f')` conversion.

The current supported facade still does:

```text
host.EsOptimizer.ask()
  -> ABI v1 f32-buffer handle
  -> CFFI float[] copy
  -> Python list[float]
```

This prototype asks whether Python can avoid that list materialization entirely while keeping ABI v1 and the existing `ask()` contract unchanged.

## Existing ABI capability

ABI v1 already exposes:

```text
br_v1_es_ask
br_v1_f32_buffer_len
br_v1_f32_buffer_copy
br_v1_handle_free
```

The prototype therefore uses only existing public ABI operations:

```text
br_v1_es_ask
  -> owned f32-buffer handle
  -> br_v1_f32_buffer_len
  -> allocate standard-library array('f')
  -> ffi.from_buffer('float[]', destination)
  -> br_v1_f32_buffer_copy
  -> free temporary ABI buffer handle
  -> memoryview candidate windows
  -> existing GraphParameterBinding.apply_flat f32-buffer fast path
```

No new `br_v1_*` symbol is introduced.

## Installed-wheel methodology

`scripts/research_python_optimizer_direct_f32.py` builds the current wheel, installs it into a fresh external virtual environment, removes repository `PYTHONPATH`, and imports the package from `site-packages`.

The large case reuses the established workload:

```text
16 x Linear(64 -> 64, bias=true)
parameter dimension = 66,560
requested population = 4
candidate count = optimizer batch_size after ask
```

Two same-seed optimizer paths are compared:

```text
A. supported facade baseline
host.EsOptimizer.ask()
  -> list[float]
  -> per-candidate list slice
  -> apply_flat(list)

B. direct-f32 prototype
raw br_v1_es_ask
  -> existing f32-buffer handle
  -> direct copy into array('f')
  -> contiguous memoryview candidate window
  -> apply_flat(f32 view)
```

The prototype records separately:

- `ask`/materialization time;
- candidate-window creation time;
- `apply_flat` time;
- post-ask transport time;
- `ask + transport` time;
- complete deterministic objective-generation time.

Semantic digest computation is deliberately outside the timed regions.

## Semantic proof boundary

No timing result is accepted unless the installed-wheel run proves:

- both variants use the same strict optimizer configuration and seed;
- a new optimizer reports zero batch size before its first ask;
- returned cardinality is read from `batch_size` after ask;
- returned flat length equals `batch_size * parameter_dim`;
- same-seed candidate batches have identical f32 byte digests across facade and direct-buffer paths;
- candidate ordering is unchanged;
- `tell()` receives exactly one fitness per candidate and preserves the established last-candidate cardinality behavior;
- direct candidate windows are one-dimensional, C-contiguous native f32 views of exact parameter length;
- graph program identity remains stable;
- GraphParameterBinding identity remains stable;
- parameter application continues to use existing core finite/fail-closed semantics;
- deterministic objective fitness history is identical across variants;
- stateful ProgramBundle replay preserves program identity, binding identity, and final parameter state;
- every raw ABI optimizer/buffer/report handle opened by the prototype is deterministically freed.

Timing is evidence only and never a CI correctness threshold.

## Decision rule

The report always uses `verdict = PASS` only for semantic/proof success.

A separate research decision is emitted:

```text
PUBLIC_ASK_F32_WORTH_IMPLEMENTING
```

only when median `ask + transport` improves by at least 10% over the existing facade-list path. Otherwise:

```text
KEEP_LIST_API_ONLY
```

The 10% trigger is not a performance guarantee or support threshold. It is only a conservative signal for whether a later additive Python product slice is worth implementing and proving separately.

## Positive-result continuation

If the prototype returns `PUBLIC_ASK_F32_WORTH_IMPLEMENTING`, the next product PR should remain narrow:

```text
existing ask() -> list[float]      remains unchanged
new additive ask_f32()             returns standard-library f32 buffer
ABI v1                              unchanged
Rust optimizer algorithm           unchanged
GraphParameterBinding core         unchanged
```

Any product promotion must add first-class installed-wheel proof for compatibility, lifecycle, exact candidate bytes, malformed/closed-handle behavior, and checkpoint/objective semantics. The prototype itself does not add that API.

## Non-goals

This slice does not change:

- `EsOptimizer.ask()` public return type;
- Python support status;
- any `br_v1_*` symbol;
- Rust optimizer algorithms;
- core GraphParameterBinding semantics;
- graph execution;
- Math Program v1-v9;
- ProgramBundle schema;
- Resolution/Authorization.

It adds no NumPy, DLPack, PyO3, persistent CFFI pointer cache, native batch graph primitive, Go, or C++ support.

## Current result

No performance conclusion is recorded until the dedicated exact-head installed-wheel workflow completes successfully and uploads `python-optimizer-direct-f32-report.json`.
