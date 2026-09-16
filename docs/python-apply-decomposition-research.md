# Python candidate apply decomposition research

Status: **evidence captured — prefer host transport investigation before core cache**

Related: #190, #192, #193

## Why this research exists

The installed Python binding-scaling research showed that facade-observed `GraphParameterBinding.apply_flat` becomes material on larger graphs while remaining approximately linear.

At 16 trainable owners / 66,560 parameters, the previous workload observed roughly:

```text
facade apply  ~= 5.5 ms
graph run     ~= 6.7 ms
```

That does **not** prove the core binding path itself needs a cache. The facade timing combines several layers:

```text
Python sequence normalization
  + CFFI float[] allocation/copy
  + versioned ABI call
  + core binding validation
  + canonical owner setters
```

This slice separates those costs before any optimization design.

## Boundary

Preserve:

```text
Python host transport
    != language-neutral ABI v1
    != GraphParameterBinding semantics
    != future optimization design
```

This research does not change:

- `br_v1_*` symbols;
- binding ordering, identity, validation, or atomicity;
- Python facade behavior;
- graph execution semantics;
- optimizer semantics;
- Math Program v1-v9;
- ProgramBundle format;
- Resolution/Authorization.

No cache, reusable validation session, NumPy, DLPack, buffer-protocol API, or controller is introduced.

## Consumer construction

The workflow builds the current wheel, installs it into a fresh venv outside the checkout, and imports both public surfaces from that installed package:

```python
from burn_research_ffi import ffi, lib
from burn_research_ffi import GraphParameterBinding, ...
```

The raw graph/binding is built only through documented `br_v1_*` calls. The research does not access private handle fields from facade objects.

For each case a second, structurally identical graph is built through the typed facade. Both graphs use the same layer ids, shapes, and deterministic f32 candidate.

## Case matrix

```text
1 owner   / Linear(64 -> 64)    = 4,160 params
1 owner   / Linear(128 -> 128)  = 16,512 params
4 owners  / Linear(64 -> 64)    = 16,640 params
16 owners / Linear(64 -> 64)    = 66,560 params
```

This keeps both flat-vector length and owner-count effects visible without repeating the entire #192 matrix.

## Measured paths

After warm-up, repeated samples record:

1. **Python normalization only**

   ```python
   [float(value) for value in candidate]
   ```

2. **CFFI allocation/copy only**

   ```python
   ffi.new("float[]", candidate)
   ```

3. **Raw preallocated ABI/core apply**

   A single `float[]` is allocated before timing, then reused across calls to `br_v1_binding_apply_flat`.

4. **Raw allocate + apply**

   A fresh CFFI `float[]` is created for every raw ABI apply.

5. **Typed facade end-to-end apply**

   ```python
   binding.apply_flat(graph, registry, candidate)
   ```

The report records median/min/p90/max plus descriptive ratios/deltas. Timing is evidence only and is not a CI threshold or SLA.

## Semantic proof

For every case, the research proves:

1. the installed module comes from `site-packages`, not the repository checkout;
2. raw ABI version remains v1;
3. raw and facade bindings expose the same expected canonical parameter count;
4. raw and facade program identities are equal;
5. raw and facade binding identities are equal;
6. the same deterministic f32 candidate applies successfully through both paths;
7. both paths read the candidate back exactly;
8. identities remain stable across mutation and timing loops;
9. raw and facade execution outputs are finite and exactly equal after applying the same state;
10. no private facade handle access is used.

Checkpoint replay is already covered by #192 and the existing regression workflows, so this micro-decomposition does not duplicate that proof.

## Observed evidence

The first CI evidence run passed all semantic assertions. Median timings were:

| Case | Params | Python normalize | CFFI alloc/copy | Raw preallocated apply | Raw allocate+apply | Facade end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 owner / w64 | 4,160 | 0.055 ms | 0.034 ms | 0.084 ms | 0.119 ms | 0.175 ms |
| 1 owner / w128 | 16,512 | 0.220 ms | 0.133 ms | 0.232 ms | 0.370 ms | 0.665 ms |
| 4 owners / w64 | 16,640 | 0.221 ms | 0.134 ms | 0.335 ms | 0.482 ms | 0.698 ms |
| 16 owners / w64 | 66,560 | 0.850 ms | 0.532 ms | 1.317 ms | 1.863 ms | 2.716 ms |

The decomposition is internally consistent. For the largest case:

```text
Python normalization      0.850 ms
CFFI allocation/copy     0.532 ms
raw ABI/core apply       1.317 ms
-------------------------------
component sum            2.698 ms
facade end-to-end        2.716 ms
```

The component sum is within about 0.018 ms of the observed facade median. Separately, `raw allocate+apply - raw preallocated apply` is about 0.546 ms, closely matching the independently measured 0.532 ms CFFI allocation/copy cost.

This means the facade cost is not dominated by a hidden single layer. At 66,560 parameters, raw preallocated ABI/core apply accounts for about **48.5%** of facade median time, while Python normalization plus CFFI allocation/copy account for roughly the other half.

The same broad pattern appears across the matrix. Owner count increases raw/core work, but the previous #192 scaling evidence already showed approximately linear behavior rather than a pathological owner-count explosion.

## Decision

Current evidence does **not** justify adding a `GraphParameterBinding` cache or weakening validation/atomicity.

The next investigation should target the Python transport boundary first because it represents a comparable or slightly larger share of facade cost and can potentially be improved without changing ABI v1 or core binding semantics.

The preferred next proof is a standard-Python contiguous f32 buffer path, such as `array('f')` or another buffer-protocol-compatible object used with CFFI `from_buffer`, while retaining the existing list/Sequence path for compatibility.

That proposal must be measured before modifying the facade. It must prove lifetime safety, contiguity/type checks, exact state application, unchanged status/error semantics, and no new third-party dependency.

Do **not** add NumPy, DLPack, PyO3, a new `br_v1_*` symbol, or a binding cache from this result alone.

## Current decision labels

```text
KEEP_ABI_V1_UNCHANGED
KEEP_BINDING_ATOMICITY_AND_IDENTITY_VALIDATION
DO_NOT_ADD_BINDING_CACHE_YET
INVESTIGATE_STANDARD_PYTHON_F32_BUFFER_FAST_PATH_NEXT
```
