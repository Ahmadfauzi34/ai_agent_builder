# Python Host Layer v1

Status: **proven first-class host surface on the installed-wheel support slice**

Related: #184, #186, #188, #190, #192, #194, #196, #198, #199, #200

## Purpose

The Python package has proven more than foreign-function access: an installed wheel can build and execute graphs, bind trainable state canonically, run host-owned objectives with the existing optimizer, export/import `ProgramBundle` with exact replay, and use a Python-side f32 candidate transport fast path without changing the native ABI.

The product boundary is:

```text
Rust reference machine
    -> burn-research.ffi.v1
    -> Python facade implementation
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

The namespace exposes the proven typed objects:

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

## Candidate transport optimization

The host namespace re-exports `GraphParameterBinding` from the typed facade, so the stateless f32 candidate fast path proven in #196 and implemented in #200 is available through the first-class host API without adding another layer or API family.

For a compatible native-f32 buffer, the transport path is:

```text
host.GraphParameterBinding.apply_flat(...)
    -> validate 1-D C-contiguous native f32 buffer
    -> fresh ffi.from_buffer view
    -> existing br_v1_binding_apply_flat
    -> discard borrowed view after call
```

Generic `Sequence[float]` remains the compatibility fallback. No persistent CFFI pointer, binding cache, ABI widening, NumPy dependency, or DLPack path is introduced.

This optimization belongs to Python transport only; atomic finite validation and mutation semantics remain owned by the Rust/core binding implementation.

## Installed-wheel proof

The supported Python claim requires a wheel installed into a fresh venv outside the repository checkout.

`scripts/audit_python_wheel.py` uses `burn_research_ffi.host` for the main workload and proves:

1. package and host imports come from `site-packages`;
2. raw `ffi` / `lib` remain separately available;
3. host API identity/version is explicit;
4. root facade exports remain aliases to host classes;
5. deterministic ownership and local use-after-close rejection;
6. stable ABI status -> Python exception mapping;
7. list/tuple and non-f32-buffer Sequence compatibility;
8. stateless compatible-f32 buffer fast-path use;
9. malformed native-f32 buffers fail locally;
10. non-finite f32 candidates remain atomically rejected by core semantics;
11. optimizer dimension comes from `GraphParameterBinding.total_len`;
12. Python owns the objective/evaluation schedule;
13. learned flat state reads back exactly;
14. stateful `ProgramBundle` import into a fresh registry preserves identities, state, and output replay;
15. large-candidate list-vs-buffer performance is recorded only as non-threshold evidence.

## Supported matrix

This promotion and transport optimization do not widen the already-proven Python matrix. The support claim remains limited to `docs/host-support.v1.json`.

## Other languages

Go and C++ are not current product priorities. Their absence does not change the language-neutral design of ABI v1, but no work should be scheduled merely to add consumers that are unlikely to be used.

## Next boundary

Do not automatically widen the ABI or add a core `GraphParameterBinding` cache. The current transport optimization addresses already-contiguous f32 candidates. Any further Python optimization must first measure a real remaining host workload, especially the current `EsOptimizer.ask()` list -> candidate slice -> `apply_flat()` path, before changing optimizer return types or adding another API.
