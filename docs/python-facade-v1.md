# Python facade v1 boundary

Status: **proven on the verified installed-wheel Python slice**

Related: #184, #185, #186, #187, #188

## Purpose

The language-neutral ABI and installed Python wheel are already proven. This slice adds a thin typed Python usability layer without widening the ABI.

Keep the layers separate:

```text
burn-research Rust core
    != public Rust package API
    != burn-research.ffi.v1
    != Python facade
    != host objective/controller policy
```

The facade is not a second reference machine. It only owns Python-side handle lifetime, status-to-exception mapping, explicit buffer copies, and method naming over the existing `br_v1_*` surface.

## Public Python surface

The wheel continues to expose raw CFFI objects:

```python
from burn_research_ffi import ffi, lib
```

It additionally exposes the typed facade:

```python
from burn_research_ffi import (
    BurnResearchError,
    ClosedHandleError,
    EsOptimizer,
    GraphBuilder,
    GraphParameterBinding,
    LinearLayerSpec,
    ProgramBundle,
    Registry,
    Status,
    Tensor,
)
```

The package ships `py.typed`.

## Ownership contract

Every facade object owning a `br_v1_handle *` supports:

- deterministic `close()`;
- context-manager ownership;
- idempotent Python-level double-close;
- local `ClosedHandleError` on use-after-close before another FFI call is attempted;
- destructor cleanup only as a defensive fallback.

The underlying ABI remains the ownership authority. The facade does not share or reinterpret Rust object layout.

## Error contract

Nonzero ABI status codes become `BurnResearchError` with:

- `status_code`: stable integer ABI status;
- `status`: `Status` enum when known;
- `diagnostic`: current thread-local ABI diagnostic text;
- `context`: facade operation name.

Consumer logic may branch on `Status`; diagnostic strings remain non-contractual.

`BR_V1_PANIC` remains distinguishable from ordinary core errors and is never converted into success.

## Tensor contract

`Tensor` remains copy-based rank-4 f32:

```text
Python sequence + explicit [d0,d1,d2,d3]
    -> br_v1_tensor_new_f32
    -> opaque owned tensor
    -> explicit to_f32 copy
```

No NumPy dependency, zero-copy buffer protocol, DLPack, or device-memory claim is introduced.

## Graph / optimizer boundary

The facade covers only the workflow already proven by the ABI:

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

`GraphParameterBinding.total_len` remains the sole source of optimizer dimension/order. The facade must not reconstruct parameter offsets.

Python still owns dataset selection, objective calculation, evaluation scheduling, stopping/promotion policy, and experiment bookkeeping. No graph-owning controller is introduced.

## Identity / checkpoint boundary

Preserve exact separation:

```text
program identity
    != binding identity
    != mutable learned state
    != ProgramBundle bytes
```

`ProgramBundle` remains the canonical stateful checkpoint payload. The facade does not add pickle or another Python-specific state format.

## Installed-wheel proof

`scripts/audit_python_wheel.py` runs the real workload through the facade after installing the wheel into a fresh external venv.

The proof verifies:

1. import comes from `site-packages`, not checkout;
2. raw `ffi` / `lib` remain available;
3. `py.typed` is present in the installed package;
4. double-close is harmless;
5. use-after-close fails locally;
6. non-finite parameter candidate becomes `BurnResearchError(Status.CORE_ERROR)` and leaves parameters unchanged;
7. graph + binding + ES + Python objective executes through facade methods;
8. stateful `ProgramBundle` replay preserves program identity, binding identity, learned flat state, and output.

Initial proof on PR #188 head `d7ab7b066324cedd62059d3e4d418c1fe8ccd6fb`:

- Python Wheel Proof #9: green;
- Rust AI CI #460: full green;
- no core `src/` or ABI header/implementation changes were required.

Because this documentation update changes the PR head, the same installed-wheel and full regression workflows must also be green on the final documentation head before merge.

## Non-goals

This slice does not add:

- new `br_v1_*` ABI symbols;
- PyO3;
- NumPy/DLPack/zero-copy;
- a training-loop/controller abstraction;
- Math Program changes;
- optimizer changes;
- Python-specific checkpoint identity;
- broader Python/platform support;
- PyPI publication;
- C++/Go/engine bindings.

## Promotion rule

The facade is considered supported only within the Python wheel support matrix already recorded in `docs/host-support.v1.json`. This proof does not widen the OS, architecture, or Python-version matrix and does not change the experimental/versioned status of the underlying ABI.
