# Python optimizer direct-f32 prototype research

Status: **measured; supported product API unchanged**

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
observed batch size = 4
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
- complete deterministic objective-generation time as a descriptive control.

Semantic digest computation is deliberately outside the timed regions.

The product decision uses only the large-case `ask + transport` comparison. The objective control is not used to select the product verdict because the direct prototype invokes the existing low-level `tell` ABI directly while the baseline invokes the facade `tell()` path. Objective timing is therefore descriptive only; objective candidate digests, fitness history, identities, and replay are the semantic evidence that matters.

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
- deterministic objective fitness history is identical across variants;
- stateful ProgramBundle replay preserves program identity, binding identity, and final parameter state;
- every raw ABI optimizer/buffer/report handle opened by the prototype is deterministically freed.

Finite/fail-closed parameter application is not reimplemented by this prototype. It continues to be owned by the existing `GraphParameterBinding.apply_flat` core/ABI boundary, and the exact-head Python Wheel Proof plus Python F32 Buffer Fast Path Research remain the regression guards for wrong-length/non-finite atomic rejection on that same buffer path.

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

## Valid measurement

The first valid installed-wheel measurement came from **Python Optimizer Direct F32 Research run #2** on head:

```text
fc4297c5fa096c971baf9ba770bd5d8067ab63fa
```

The report returned:

```text
verdict  = PASS
decision = PUBLIC_ASK_F32_WORTH_IMPLEMENTING
```

Large-case medians:

| Path / component | Facade list | Direct f32 |
| --- | ---: | ---: |
| `ask` / materialization | 35.516 ms | 11.062 ms |
| candidate window | 0.187 ms | 0.0026 ms |
| `apply_flat` per candidate | 5.476 ms | 2.872 ms |
| post-ask transport | 23.227 ms | 11.656 ms |
| `ask + transport` | 58.544 ms | 22.869 ms |

Derived evidence:

```text
direct ask speedup             = 3.211x
direct ask + transport speedup = 2.560x
```

So the direct-buffer prototype reduces median `ask + transport` from about **58.54 ms to 22.87 ms**, roughly a **61% reduction**, while avoiding any ABI widening. The optimizer-side materialization alone improves from about **35.52 ms to 11.06 ms**, showing that Python-list construction was a substantial part of the remaining host cost at this parameter dimension.

The direct candidate bytes are exactly identical to the facade-list candidate bytes across all seven measured generations. Both paths observed batch size 4, and program identity plus binding identity remained stable.

## Objective semantic control

The deterministic 9-parameter objective workload produced identical candidate digests and identical fitness history for all five generations. Stateful ProgramBundle replay preserved program identity, binding identity, and final parameter state in both variants.

The measured full-generation medians were close:

```text
facade list       10.809 ms
direct f32        10.702 ms
```

These numbers are descriptive only and are not part of the product decision, because the prototype and facade use different Python wrappers around the same existing `tell` ABI. The important objective result is semantic equivalence, not the small timing difference.

## Decision

The evidence supports:

```text
PUBLIC_ASK_F32_WORTH_IMPLEMENTING
```

This means a separate additive Python product slice is justified. It does **not** mean this research PR itself changes the supported facade.

The narrow continuation is:

```text
existing ask() -> list[float]      remains unchanged
new additive ask_f32()             returns standard-library array('f')
ABI v1                              unchanged
Rust optimizer algorithm           unchanged
GraphParameterBinding core         unchanged
```

A product PR should implement the direct-copy operation behind the typed facade rather than requiring callers to use the raw `ffi/lib` escape hatch. It should reuse existing `br_v1_es_ask`, `br_v1_f32_buffer_len`, and `br_v1_f32_buffer_copy`, and add first-class installed-wheel proof for exact byte equivalence, lifecycle/cardinality, ordinary `ask()` compatibility, deterministic handle cleanup, and closed-handle/error behavior.

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

## Final merge requirements

On the exact documentation head require:

- Python Optimizer Direct F32 Research green with report upload;
- Python Optimizer Ask Apply Research green;
- Python Wheel Proof green;
- Python F32 Buffer Fast Path Research green;
- Python Apply Decomposition Research green;
- Python Binding Scaling Research green;
- Python Workload Research green;
- full Rust AI CI green through native package, FFI/CFFI, WASM/Node, graph/ES/checkpoint, Math v1-v9, Resolution, host tests, and artifact upload;
- no blocking reviews or unresolved review threads;
- diff remains exactly the three research/prototype files for #206.
