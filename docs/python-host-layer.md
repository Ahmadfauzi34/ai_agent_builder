# Python Host Layer v1

Status: **candidate first-class host surface pending installed-wheel proof**

Related: #184, #186, #188, #190, #192, #194, #196, #198

## Purpose

The Python package has already proven more than foreign-function access: an installed wheel can build and execute graphs, bind trainable state canonically, run host-owned objectives with the existing optimizer, and export/import `ProgramBundle` with exact replay.

This document promotes that existing capability into an explicit product boundary:

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
schema        = burn-research.python-host.v1
version       = 1
abi_version   = 1
abi_schema    = burn-research.ffi.v1
orchestration = host_owned
typed         = true
raw_ffi_primary = false
```

This is **not** a second native ABI. `br_v1_*` remains unchanged.

## Installed-wheel proof

The supported Python claim continues to require a wheel installed into a fresh venv outside the repository checkout.

The wheel proof must use `burn_research_ffi.host` for the main workload and still prove:

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

## Supported matrix

This promotion does not widen the already-proven Python matrix. The current support claim remains limited to the matrix recorded in `docs/host-support.v1.json`.

## Other languages

Go and C++ are not current product priorities. Their absence does not change the language-neutral design of ABI v1, but no work should be scheduled merely to add consumers that are unlikely to be used.

## Next optimization boundary

The standard-library f32 buffer research in #196 demonstrated a substantial Python transport opportunity without ABI widening. That optimization remains a separate slice after the host namespace itself is proven.
