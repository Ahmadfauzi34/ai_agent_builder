# Python standard-library f32 buffer fast-path research

Status: **research candidate pending CI evidence**

Related: #192, #194, #195

## Why this research exists

The candidate-apply decomposition proved that Python host marshalling is a material share of typed-facade `GraphParameterBinding.apply_flat` cost at larger parameter counts.

For the 66,560-parameter case, the evidence was approximately:

```text
Python normalization      0.850 ms
CFFI allocation/copy     0.532 ms
raw ABI/core apply       1.317 ms
facade end-to-end        2.716 ms
```

The host side therefore deserves a cheaper proof before any core cache, ABI widening, NumPy dependency, DLPack path, or PyO3 redesign is considered.

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

## Required semantic proof

Every measured case must prove:

1. package import comes from installed `site-packages`, not the checkout;
2. ABI version remains v1;
3. `array('f').itemsize == 4` on the runner;
4. raw and facade bindings expose the exact expected canonical parameter count;
5. raw and facade program identities match;
6. raw and facade binding identities match;
7. list-backed and `array('f')`-backed ABI apply both read back the exact deterministic f32 candidate;
8. program/binding identities remain stable across mutation and timing;
9. graph output remains finite;
10. raw buffer-backed execution output exactly matches the typed-facade/list-compatible path after the same candidate state;
11. persistent-view backing storage remains strongly referenced;
12. incompatible/non-contiguous/wrong-length buffers are rejected before ABI use;
13. no private facade handle access occurs.

A `memoryview` path is optional evidence. If CFFI rejects its representation, the research records that limitation rather than weakening validation or forcing support.

## Decision rule

If `array('f') + ffi.from_buffer` fresh-view or persistent-view apply removes a substantial fraction of the host marshalling cost while preserving the above semantic and lifetime proof, open a **separate facade implementation issue**.

Any later facade change must:

- preserve the current generic `Sequence[float]` compatibility path;
- make the fast path explicit and fail closed on incompatible buffers;
- preserve stable ABI/error semantics and finite/atomic core behavior;
- add no mandatory third-party dependency;
- keep ABI v1 unchanged.

If the measured gain is small, unstable, or depends on fragile buffer lifetime/type assumptions, retain the current facade and stop this optimization line.
