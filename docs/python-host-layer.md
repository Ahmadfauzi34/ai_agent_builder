# Python Host Layer v1

Status: **first-class host surface proven; stateless f32 candidate fast path pending exact-head CI proof**

Related: #184, #186, #188, #190, #192, #194, #196, #198, #199, #201

## Purpose

The Python package has proven more than foreign-function access: an installed wheel can build and execute graphs, bind trainable state canonically, run host-owned objectives with the existing optimizer, and export/import `ProgramBundle` with exact replay.

The product boundary is:

```text
Rust reference machine
    -> burn-research.ffi.v1
    -> burn_research_ffi.host
    -> Python-owned datasets / objectives / scheduling / applications
```

Python is a first-class host surface. It is **not** part of the Rust core and it does not reverse the dependency direction.

## Public namespace

The primary typed Python host namespace is:

```python
from burn_research_ffi import host
```

with host API identity:

```text
burn-research.python-host.v1
```

The namespace exposes the already-proven typed objects:

- `Registry`
- `LinearLayerSpec`
- `GraphBuilder`
- `Graph`
- `Tensor`
- `GraphParameterBinding`
- `EsOptimizer`
- `ProgramBundle`
- `Status`
- `BurnResearchError`
- `ClosedHandleError`
- ABI/host capability inspection

The package root continues to re-export the same facade objects for compatibility.

Raw CFFI `ffi` / `lib` remain available from `burn_research_ffi` as a low-level diagnostic/escape-hatch surface. They are deliberately **not** exported by `burn_research_ffi.host`.

## Architectural boundary

```text
Python Host Layer
    != language-neutral ABI
    != Rust reference machine
    != controller/policy ownership
```

The host layer may own:

- Python object lifetime and context-manager ergonomics;
- dataset loading and preprocessing outside the reference machine;
- objective functions and fitness calculation;
- experiment/training scheduling;
- application/agent integration;
- Python-side transport optimizations that preserve ABI semantics.

It must not duplicate or redefine:

- graph execution;
- canonical parameter ordering/identity;
- finite/fail-closed parameter application;
- optimizer semantics;
- `ProgramBundle` encoding/replay;
- Math Program semantics;
- Resolution/Authorization semantics.

There is intentionally no giant `PythonHost` object owning graph + optimizer + policy. Composition remains explicit so the existing proof boundaries stay visible.

## Runtime identity

`host_capabilities()` reports the package-level host contract and delegates ABI identity to the existing foreign layer:

```text
schema          = burn-research.python-host.v1
version         = 1
abi_version     = 1
abi_schema      = burn-research.ffi.v1
orchestration   = host_owned
typed           = true
raw_ffi_primary = false
```

This is **not** a second native ABI. `br_v1_*` remains unchanged.

## Installed-wheel proof

The supported Python claim requires a wheel installed into a fresh venv outside the repository checkout.

The wheel proof uses `burn_research_ffi.host` for the main workload and proves:

1. package import comes from `site-packages`;
2. raw `ffi` / `lib` remain separately available;
3. host API identity/version is explicit;
4. deterministic ownership and local use-after-close rejection;
5. stable ABI status -> Python exception mapping;
6. non-finite candidate rejection remains atomic;
7. optimizer dimension comes from `GraphParameterBinding.total_len`;
8. Python owns the objective/evaluation schedule;
9. learned flat state reads back exactly;
10. stateful `ProgramBundle` import into a fresh registry preserves identities, state, and output replay.

## Candidate transport fast path

Research in #194 separated large-candidate host apply cost into Python/CFFI marshalling and raw ABI/core work. Research in #196 then showed that an already-contiguous standard-library `array('f')` candidate can be borrowed with `ffi.from_buffer` at negligible acquisition cost, while a persistent CFFI pointer/view provided essentially no benefit over acquiring a fresh view for each call.

The Python host therefore keeps the existing public method:

```python
binding.apply_flat(graph, registry, candidate)
```

and adds only an internal, stateless transport optimization.

A candidate is borrowed directly for one ABI call only when a `memoryview` proves:

```text
format == 'f'
itemsize == 4
ndim == 1
C-contiguous
writable
```

For that path:

```text
Python backing object + memoryview
        -> fresh ffi.from_buffer("float[]", view)
        -> existing br_v1_binding_apply_flat
        -> borrowed view discarded when the call returns
```

Nothing is stored on `GraphParameterBinding`, `Registry`, or another long-lived object.

Anything that cannot prove the buffer contract falls back to the historical generic path:

```text
Sequence[float]
    -> list[float]
    -> ffi.new("float[]", values)
    -> existing br_v1_binding_apply_flat
```

This keeps list, tuple, float64-array, non-contiguous sequence-like objects, and other compatible Python sequences working without raw reinterpretation.

### Error ownership is unchanged

The fast path deliberately does **not** call `binding.total_len` or create a new Python-side length validation rule. It forwards the actual eligible-buffer length to the same ABI function. Therefore structural length errors and finite-only validation remain owned by the existing ABI/core boundary and retain the same `BurnResearchError` mapping and atomicity guarantee.

The installed-wheel proof additionally requires:

- a deliberately non-iterable `array('f')` subclass to succeed, proving the buffer path was actually used;
- writable contiguous f32 `memoryview` success;
- list/tuple compatibility;
- float64-array fallback rather than f32 reinterpretation;
- non-contiguous f32 memoryview compatibility fallback;
- wrong-length f32 buffer rejection through the existing ABI/core error path with no mutation;
- non-finite f32 buffer rejection through the existing core finite-only path with no mutation;
- stable program and binding identities across all transport paths;
- exact stateful `ProgramBundle` replay after a fast-path candidate is applied.

## Supported matrix

The host promotion and candidate transport optimization do not widen the already-proven Python matrix. The current support claim remains limited to the matrix recorded in `docs/host-support.v1.json`.

## Other languages

Go and C++ are not current product priorities. Their absence does not change the language-neutral design of ABI v1, but no work should be scheduled merely to add consumers that are unlikely to be used.

## Performance interpretation

The f32 buffer path is an implementation optimization, not a semantic capability version and not an ABI change. Research timing remains descriptive evidence only; it is not a CI performance threshold or SLA.

No NumPy, DLPack, PyO3, persistent CFFI pointer cache, or binding-validation cache is introduced by this slice.
