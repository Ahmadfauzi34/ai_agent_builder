# Python Host Layer v1

Status: **proven first-class host surface on the installed-wheel support slice**

Related: #184, #186, #188, #190, #192, #194, #196, #198, #199, #200, #203, #204, #205, #206, #207

## Purpose

The Python package has proven more than foreign-function access: an installed wheel can build and execute graphs, bind trainable state canonically, run host-owned objectives with the existing optimizer, export/import `ProgramBundle` with exact replay, and use Python-side f32 transport paths without changing the native ABI.

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

## Parameter candidate transport

The host namespace re-exports `GraphParameterBinding` from the typed facade, so the stateless f32 candidate fast path proven in #196 and implemented in #200 is available through the first-class host API without adding another layer or API family.

For a compatible native-f32 buffer, the transport path is:

```text
host.GraphParameterBinding.apply_flat(...)
    -> prove writable 1-D C-contiguous native f32 storage
    -> fresh ffi.from_buffer view
    -> existing br_v1_binding_apply_flat
    -> discard borrowed view after call
```

If a buffer cannot prove that borrowing contract, the facade falls back to the historical `Sequence[float]` conversion path when the object is otherwise iterable. The fast path does not pre-read `binding.total_len`; actual candidate length and finite-only validation remain owned by the existing ABI/core apply boundary.

No persistent CFFI pointer, binding cache, ABI widening, NumPy dependency, or DLPack path is introduced.

This optimization belongs to Python transport only; structural length errors, atomic finite validation, and mutation semantics remain owned by the Rust/core binding implementation.

## Optimizer candidate materialization

The historical optimizer API remains unchanged:

```python
optimizer.ask() -> list[float]
```

For large parameter workloads, the host also exposes an additive typed-buffer path:

```python
optimizer.ask_f32() -> array('f')
```

Both methods invoke the same existing optimizer `ask` operation and therefore have the same core lifecycle semantics: each call replaces the pending candidate batch, and `batch_size` is authoritative after the call. The difference is only Python materialization.

`ask_f32()` uses the existing ABI v1 f32-buffer result directly:

```text
br_v1_es_ask
    -> owned ABI f32-buffer handle
    -> br_v1_f32_buffer_len
    -> allocate standard-library array('f')
    -> fresh ffi.from_buffer view of the destination
    -> br_v1_f32_buffer_copy
    -> free temporary ABI buffer handle
    -> return array('f')
```

No optimizer algorithm, ABI symbol, graph primitive, or core binding semantics change. `array('f')` is standard-library storage; callers may take contiguous `memoryview` candidate windows and pass them directly to the existing `GraphParameterBinding.apply_flat` f32 fast path.

The compatibility choice remains explicit:

```text
ask()      -> ordinary Python list compatibility
ask_f32()  -> native-f32 host transport for workloads that benefit from it
```

Research #205/#206 measured why the additive path is justified at representative scale. At 66,560 parameters and batch size 4, the installed-wheel prototype reduced median `ask + transport` from about 58.54 ms to 22.87 ms (~2.56x), with byte-identical candidates and unchanged objective/checkpoint semantics. Those measurements are evidence for the API choice, **not** a performance SLA.

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
8. stateless compatible-f32 binding fast-path use;
9. non-borrowable f32 layouts fall back to Sequence conversion when compatible;
10. wrong-length and non-finite borrowed f32 candidates remain atomically rejected by core semantics;
11. historical `EsOptimizer.ask()` still returns `list[float]`;
12. additive `EsOptimizer.ask_f32()` returns writable, one-dimensional, C-contiguous standard-library `array('f')` storage;
13. same strict optimizer config/seed produces byte-identical candidates between `ask()` and `ask_f32()`;
14. `ask_f32()` batch cardinality and `tell()` lifecycle match the existing optimizer contract;
15. candidate windows from `ask_f32()` feed the existing binding f32 fast path directly;
16. closed optimizer handles reject `ask_f32()` locally;
17. optimizer dimension comes from `GraphParameterBinding.total_len`;
18. Python owns the objective/evaluation schedule;
19. learned flat state reads back exactly;
20. stateful `ProgramBundle` import into a fresh registry preserves identities, state, and output replay;
21. large-candidate transport performance is recorded only as non-threshold evidence.

## Supported matrix

This additive optimizer transport method does not widen the already-proven Python platform matrix. The support claim remains limited to `docs/host-support.v1.json`.

## Other languages

Go and C++ are not current product priorities. Their absence does not change the language-neutral design of ABI v1, but no work should be scheduled merely to add consumers that are unlikely to be used.

## Next boundary

Do not widen ABI v1, add a core `GraphParameterBinding` cache, or add native batch graph/controller primitives merely because `ask_f32()` exists. Future Python optimization should continue from measured installed-wheel workloads and preserve the same separation:

```text
Python transport optimization
    != ABI widening
    != optimizer semantic change
    != controller ownership in core
```

The current additive `ask_f32()` path closes the measured optimizer candidate-materialization bottleneck without changing the reference-machine boundary.
