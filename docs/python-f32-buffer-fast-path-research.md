# Python standard-library f32 buffer fast-path research

Status: **evidence captured — stateless native-f32 buffer fast path justified**

Related: #192, #194, #195

## Why this research exists

The candidate-apply decomposition proved that Python host marshalling is a material share of typed-facade `GraphParameterBinding.apply_flat` cost at larger parameter counts.

For the 66,560-parameter case, the prior evidence was approximately:

```text
Python normalization      0.850 ms
CFFI allocation/copy     0.532 ms
raw ABI/core apply       1.317 ms
facade end-to-end        2.716 ms
```

The host side therefore deserved a cheaper proof before any core cache, ABI widening, NumPy dependency, DLPack path, or PyO3 redesign was considered.

## Research question

Can the Python standard-library `array('f')` plus CFFI `ffi.from_buffer` pass an already-contiguous native f32 candidate to the existing `const float *` ABI with materially less host overhead while preserving exact state semantics and safe object lifetime?

## Boundary

Preserve:

```text
Python buffer transport
    != language-neutral ABI v1
    != GraphParameterBinding semantics
    != future facade implementation
```

This PR is research only. It does not change:

- any `br_v1_*` symbol;
- `GraphParameterBinding` ordering, validation, identity, finite-only contract, or atomicity;
- Python facade implementation;
- graph execution semantics;
- optimizer semantics;
- Math Program v1-v9;
- ProgramBundle format;
- Resolution/Authorization.

It adds no NumPy, DLPack, PyO3, third-party array library, native ABI symbol, binding cache, or controller abstraction.

## Installed-wheel consumer

The workflow builds the current wheel, installs it into a fresh temporary venv outside the checkout, and imports only the public package surface:

```python
from burn_research_ffi import ffi, lib
from burn_research_ffi import GraphParameterBinding, ...
```

The raw graph is built through documented `br_v1_*` calls. A structurally identical graph is built through the typed facade for semantic comparison. No private facade handle is accessed.

## Buffer contract under test

The research helper accepts only a candidate exposing a buffer with all of these properties:

```text
ndim == 1
C-contiguous
itemsize == 4
format == 'f'
len == binding.total_len
```

Anything else fails closed before calling the ABI.

The research explicitly proves rejection of:

- `array('d')` / 8-byte float elements;
- non-contiguous sliced memoryviews;
- byte-oriented buffers with the wrong format;
- a correct f32 buffer with the wrong element count.

The helper then acquires CFFI storage with:

```python
ffi.from_buffer("float[]", candidate_array)
```

For a persistent CFFI view, both the backing `array('f')` and the CFFI cdata view remain strongly referenced for the full timed loop.

## Case matrix

```text
1 owner   / Linear(64 -> 64) x1   = 4,160 params
4 owners  / Linear(64 -> 64) x4   = 16,640 params
16 owners / Linear(64 -> 64) x16  = 66,560 params
```

This spans the parameter/owner range where host marshalling became visible in #194.

## Measured paths

After warm-up, repeated samples record:

1. Python list normalization;
2. CFFI list allocation/copy with `ffi.new("float[]", values)`;
3. `array('f')` construction cost for context;
4. buffer contract validation;
5. `ffi.from_buffer` acquisition alone;
6. raw ABI/core apply with a preallocated CFFI array created from a list;
7. raw apply with a freshly validated/acquired `array('f')` CFFI view each iteration;
8. raw apply with one persistent `array('f')` CFFI view;
9. optional fresh-view `memoryview(array('f'))` apply if CFFI accepts it;
10. current typed-facade end-to-end apply for context.

Timing is evidence only. It is not a CI threshold, SLA, or cross-machine benchmark contract.

## Semantic proof

The first CI evidence run passed every required semantic assertion:

1. package import came from installed `site-packages`, not the checkout;
2. ABI version remained v1;
3. `array('f').itemsize == 4`;
4. raw and facade bindings exposed the expected canonical parameter count;
5. raw and facade program identities matched;
6. raw and facade binding identities matched;
7. list-backed and `array('f')`-backed ABI apply read back the exact deterministic f32 candidate;
8. program/binding identities remained stable across mutation and timing;
9. graph output remained finite;
10. raw buffer-backed execution output exactly matched the typed-facade/list-compatible path after the same state;
11. persistent-view backing storage remained strongly referenced;
12. incompatible/non-contiguous/wrong-length buffers were rejected before ABI use;
13. no private facade handle access occurred.

CFFI also accepted a contiguous `memoryview(array('f'))` on the measured Python 3.12/Linux runner. That is useful evidence, but the standard-library `array('f')` path remains the cleanest primary proof because its element type and ownership are explicit.

## Observed evidence

Median timings from the first CI run were:

| Case | Params | Facade end-to-end | Fresh `array('f')` view + apply | Persistent view + apply | Validation | `from_buffer` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 owner / w64 | 4,160 | 0.369 ms | 0.188 ms | 0.183 ms | 0.00047 ms | 0.00037 ms |
| 4 owners / w64 | 16,640 | 1.387 ms | 0.735 ms | 0.729 ms | 0.00044 ms | 0.00036 ms |
| 16 owners / w64 | 66,560 | 5.542 ms | 2.864 ms | 2.857 ms | 0.00048 ms | 0.00034 ms |

The fresh stateless buffer path saved approximately:

```text
4,160 params    0.181 ms  (~49%)
16,640 params   0.652 ms  (~47%)
66,560 params   2.679 ms  (~48%)
```

Most importantly, holding a persistent CFFI view did **not** provide a meaningful additional benefit. The fresh-view / persistent-view median ratios were approximately:

```text
4,160 params    1.025x
16,640 params   1.007x
66,560 params   1.0025x
```

At the largest case, fresh view cost only about 0.007 ms more than persistent view. Buffer validation and `ffi.from_buffer` acquisition themselves were sub-microsecond medians and effectively negligible relative to core apply.

The raw preallocated CFFI-list pointer and persistent `array('f')` pointer also had nearly identical apply medians, confirming that `from_buffer` does not add a hidden per-element copy before the ABI call.

### Construction cost matters

Building a new `array('f')` from a generic Python sequence is still O(N):

```text
4,160 params    ~0.079 ms
16,640 params   ~0.315 ms
66,560 params   ~1.272 ms
```

Therefore the fast path should **not** silently convert every generic `Sequence[float]` into an array. Its strongest use case is when the caller/optimizer already owns a compatible contiguous native-f32 buffer. The current generic Sequence/list path remains the compatibility fallback.

## Decision

The evidence justifies a **stateless optional native-f32 buffer fast path in the Python facade**, but does not justify a persistent pointer/session API.

A later implementation should detect/accept an explicitly compatible one-dimensional C-contiguous native-f32 buffer, validate it fail-closed, acquire a fresh CFFI view for that one call, invoke the unchanged `br_v1_binding_apply_flat`, and release the temporary CFFI view when the call returns.

This keeps lifetime ownership simple:

```text
Python backing buffer alive
    -> validate
    -> ffi.from_buffer
    -> synchronous ABI call
    -> temporary CFFI view released
```

No pointer is cached across calls.

The implementation must preserve the existing generic `Sequence[float]` path unchanged as a fallback. It should not auto-convert arbitrary sequences to `array('f')` merely to enter the fast path.

Do **not** add NumPy, DLPack, PyO3, a new `br_v1_*` symbol, a persistent binding session, or a core binding cache based on this result.

## Current decision labels

```text
KEEP_ABI_V1_UNCHANGED
KEEP_GENERIC_SEQUENCE_FALLBACK
ADD_STATELESS_COMPATIBLE_F32_BUFFER_FAST_PATH_NEXT
DO_NOT_CACHE_CFFI_POINTERS_ACROSS_CALLS
DO_NOT_ADD_BINDING_CACHE_YET
```
