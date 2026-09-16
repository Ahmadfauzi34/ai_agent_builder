# Python candidate apply decomposition research

Status: **research candidate pending CI evidence**

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

For every case, the research must prove:

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

## Decision rule

Do not infer the next optimization until the decomposition is observed.

If raw preallocated ABI/core apply remains close to facade end-to-end cost, the next investigation may target the core apply path while preserving atomicity and identity validation.

If Python normalization and CFFI allocation account for a substantial share, prefer a host-side transport improvement before touching core semantics. Any transport proposal should first consider standard Python buffer-capable objects and should not add a third-party dependency or ABI widening without a separate proof.

If neither side is sufficiently dominant relative to graph execution, keep the current implementation unchanged.
