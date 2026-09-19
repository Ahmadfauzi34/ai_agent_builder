# Python facade v1 boundary

Status: **proven on the verified installed-wheel Python host slice**

Related: #184, #185, #186, #187, #188, #195, #196, #197, #199, #200

## Purpose

The language-neutral ABI, installed Python wheel, typed facade, and first-class Python host namespace are separate layers over the same Rust reference machine.

Keep the dependency direction explicit:

```text
burn-research Rust core
    -> public Rust package API
    -> burn-research.ffi.v1
    -> Python facade implementation
    -> burn_research_ffi.host
    -> Python-owned datasets / objectives / scheduling / applications
```

The facade is not a second reference machine. It owns Python handle lifetime, status-to-exception mapping, buffer transport/copies, and method naming over the existing `br_v1_*` surface.

## Public Python surface

The primary typed host namespace is:

```python
from burn_research_ffi import host
```

with API identity:

```text
burn-research.python-host.v1
```

The package root continues to expose the same facade classes as compatibility aliases. Raw CFFI remains separately available:

```python
from burn_research_ffi import ffi, lib
```

The package ships `py.typed`.

## Ownership contract

Every facade object owning a `br_v1_handle *` supports:

- deterministic `close()`;
- context-manager ownership;
- idempotent Python-level double-close;
- local `ClosedHandleError` on use-after-close before another FFI call;
- destructor cleanup only as a defensive fallback.

The ABI remains the ownership authority. Python never reinterprets Rust object layout.

## Error contract

Nonzero ABI status codes become `BurnResearchError` with stable integer `status_code`, a `Status` enum when known, contextual operation name, and non-contractual diagnostic text.

`BR_V1_PANIC` remains distinguishable from ordinary core errors and is never converted into success.

## Tensor contract

`Tensor` remains copy-based rank-4 f32:

```text
Python sequence + explicit [d0,d1,d2,d3]
    -> br_v1_tensor_new_f32
    -> opaque owned tensor
    -> explicit to_f32 copy
```

The candidate-buffer optimization described below does **not** change the tensor contract. No NumPy dependency, DLPack, device-memory, or general zero-copy tensor claim is introduced.

## Graph / optimizer boundary

The typed host surface composes only capabilities already owned by the ABI/reference machine:

```text
Registry
  -> LinearLayerSpec
  -> GraphBuilder / Graph
  -> GraphParameterBinding
  -> EsOptimizer.ask
  -> host-owned Python objective
  -> EsOptimizer.tell
  -> ProgramBundle bytes
```

`GraphParameterBinding.total_len` remains the sole source of optimizer dimension/order. Python does not reconstruct parameter offsets.

Python still owns dataset selection, objective calculation, evaluation scheduling, stopping/promotion policy, experiment bookkeeping, and application integration. There is intentionally no giant graph-owning controller object.

### In-place OpenES learning-rate control

`EsOptimizer.set_learning_rate(value)` is a narrow host-control operation over the additive ABI v1 setter.

The method:

- accepts only finite positive values;
- is valid only for OpenES;
- is valid only after a generation has completed and before the next `ask()`;
- rejects mutation while a candidate batch is pending;
- preserves the current optimizer trajectory instead of constructing a new optimizer.

The installed-wheel proof uses two identical seeded optimizers to verify that changing LR between generations leaves the next `ask_f32()` candidate bytes exactly unchanged. After those identical candidates receive identical fitness, their later trajectories diverge only because the `tell()` update uses different learning rates.

This operation does not implement an automatic learning-rate schedule. Python remains responsible for deciding whether and when to call it.

## GraphParameterBinding candidate transport

`GraphParameterBinding.apply_flat(...)` preserves one public method and two Python-side transport paths over the same `br_v1_binding_apply_flat` ABI call.

### Generic Sequence compatibility path

Existing callers remain supported:

```text
Sequence[float]
    -> Python float normalization
    -> CFFI float[] allocation/copy
    -> br_v1_binding_apply_flat
```

Lists, tuples, and non-f32 buffer-backed Sequences continue through this historical path.

### Stateless compatible-f32 buffer fast path

Research #196 demonstrated that already-contiguous native-f32 host storage can remove a material share of Python marshalling cost without ABI widening or core changes.

The facade borrows a buffer only when all of these are true:

```text
format == 'f'
itemsize == 4
ndim == 1
C-contiguous
writable
```

The path is:

```text
compatible Python f32 buffer
    -> memoryview validation
    -> fresh ffi.from_buffer("float[]", view)
    -> br_v1_binding_apply_flat
    -> discard view/cdata after call returns
```

Required invariants:

- no CFFI pointer/view is cached on a facade or host object;
- the backing Python object and memoryview remain strongly referenced for the complete ABI call;
- the fast path does not pre-read `binding.total_len`;
- wrong-length borrowed f32 buffers reach the existing ABI/core length validation and preserve its error mapping;
- native-f32 layouts that cannot be borrowed, including non-contiguous storage, fall back to the historical Sequence path when iterable;
- objects exposing a non-f32 buffer remain eligible for the historical Sequence fallback;
- NaN/Inf buffer candidates still reach the existing core finite-only atomic rejection;
- ABI v1, canonical parameter ordering, binding identity, graph identity, and mutation semantics are unchanged.

Research #196 found a persistent CFFI view gave negligible benefit over a fresh view, so no pointer/session cache is introduced.

## Identity / checkpoint boundary

Preserve exact separation:

```text
program identity
    != binding identity
    != mutable learned state
    != ProgramBundle bytes
```

`ProgramBundle` remains the canonical stateful checkpoint payload. Python adds no pickle/checkpoint identity of its own.

## Installed-wheel proof

`scripts/audit_python_wheel.py` installs the wheel into a fresh external venv and runs the main workload through `burn_research_ffi.host` while retaining root aliases and raw CFFI as compatibility/escape-hatch surfaces.

The proof verifies:

1. package and host imports come from `site-packages`, not checkout;
2. host API schema/version is explicit and raw `ffi` / `lib` remain separately available;
3. root facade exports are aliases to host classes;
4. `py.typed` is installed;
5. deterministic close/double-close and local use-after-close behavior;
6. list/tuple Sequence compatibility;
7. non-f32 buffer-backed Sequence fallback;
8. a compatible f32 buffer whose iterator raises still succeeds, proving the buffer fast path is actually used;
9. contiguous native-f32 `memoryview` uses the same stateless path;
10. wrong-length borrowed f32 buffers fail through ABI/core with `BurnResearchError(Status.CORE_ERROR)` and no mutation;
11. non-contiguous f32 memoryviews fall back to the Sequence path when iterable;
12. non-finite f32 buffer candidates become `BurnResearchError(Status.CORE_ERROR)` and leave state unchanged;
13. program/binding identities remain stable;
14. graph + ES + Python-owned objective executes through the host surface;
15. in-place OpenES learning-rate mutation preserves the next candidate batch, rejects pending-batch mutation, and changes only subsequent `tell()` update scale;
16. stateful `ProgramBundle` replay preserves identities, learned flat state, and output;
17. large-candidate list-vs-buffer timing is recorded as evidence only, never as a CI performance threshold.

On the first implementation proof for #200, the 66,560-parameter installed-wheel workload measured approximately:

```text
list fallback median   2.570 ms
f32 buffer median      1.273 ms
ratio                  2.02x
```

These are runner-specific evidence, not a performance SLA.

## Non-goals

This slice does not add:

- ABI version changes or breaking changes to existing `br_v1_*` symbols;
- PyO3;
- NumPy/DLPack tensor integration;
- persistent CFFI pointer/view caches;
- `GraphParameterBinding` cache/reusable core validation state;
- a training-loop/controller abstraction;
- Math Program changes;
- optimizer algorithm changes or automatic adaptive schedules;
- Python-specific checkpoint identity;
- broader Python/platform support;
- PyPI publication;
- C++/Go/engine consumers.

## Promotion rule

The typed facade/host is supported only within the matrix recorded in `docs/host-support.v1.json`. This transport optimization does not widen OS, architecture, or Python-version support and does not change the experimental/versioned status of ABI v1.
