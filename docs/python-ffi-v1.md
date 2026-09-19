# Python-first FFI v1 proof boundary

Status: **semantic ABI proof complete; packaged Python support is verified only for the narrow matrix recorded in `docs/python-wheel-support.md`**

Related: #182, #183, #184, #185, #186, #258

## Purpose

This slice proves the smallest language-neutral foreign boundary needed to exercise the already-supported Rust orchestration path from Python.

It deliberately preserves:

```text
burn-research Rust core
    != public Rust package API
    != C ABI v1
    != Python package/glue
    != host objective/controller policy
```

No Math Program semantics, graph identity schema, checkpoint schema, Resolution/Authorization ownership, or optimizer algorithm changes are introduced.

## Crate boundary

`ffi/` is a separate `cdylib` crate and depends only on the supported public `burn-research` package surface.

The root crate does not gain a Python dependency.

The first ABI is explicitly versioned through `br_v1_*` symbols and `br_v1_abi_version() == 1`.

`br_v1_capabilities_json()` exposes the machine-readable boundary contract used by the Python semantic proof.

## Opaque ownership

Foreign callers receive only `br_v1_handle *` values. Rust structs and compiler-generated layout are not part of the ABI.

One handle representation carries a tagged Rust enum internally, so passing a valid handle of the wrong object family returns `BR_V1_INVALID_HANDLE_TYPE` rather than reinterpreting another Rust layout.

Every successfully returned owned handle must be released with `br_v1_handle_free`.

As with ordinary C APIs, callers must not use a handle after freeing it or pass arbitrary non-library pointers as handles.

## Error and panic boundary

ABI calls return stable integer status codes. Diagnostic text is thread-local and can be copied with:

```text
br_v1_last_error_len
br_v1_last_error_copy
```

Callers must branch on status codes, not diagnostic message text.

Core `Result<..., String>` failures are translated to `BR_V1_CORE_ERROR`. Invalid FFI arguments and wrong handle families have separate status codes.

Every semantic entrypoint is wrapped in `catch_unwind`. A Rust panic is converted to `BR_V1_PANIC`; panic text is diagnostic only.

This guard is the final defensive boundary. Normal invalid input should be rejected before a panic path is reached.

## Tensor v1

Do not expose `WasmTensor` or `TensorView` as the Python contract.

ABI v1 uses explicit copy-based rank-4 f32 transfer:

```text
const float * + len + [d0,d1,d2,d3]
    -> owned opaque tensor handle
    -> graph execution
    -> explicit copy_to_f32
```

The ABI checks shape multiplication and exact element count before calling the existing tensor constructor. This prevents malformed Python input from reaching the native `WasmTensor::new` panic path.

`TensorView` remains a WASM/SharedArrayBuffer concept and is not used by Python.

This proof makes no zero-copy, NumPy buffer, DLPack, or device-memory interoperability claim.

Tensor payloads are not globally constrained to finite values by this v1 bridge. The existing finite-only invariants remain where they already belong: graph parameter candidates and ES fitness boundaries.

## Graph / binding / optimizer ownership

The Python proof uses only the already-proven orchestration path:

```text
registry + layer spec
    -> graph builder
    -> compiled graph
    -> GraphParameterBinding
    -> EsOptimizer.ask
    -> Python-owned objective
    -> GraphParameterBinding.apply_flat
    -> graph.run
    -> EsOptimizer.tell
```

Python never derives trainable offsets itself. `GraphParameterBinding` remains the sole source of parameter dimension/order.

Python owns dataset selection, objective calculation, evaluation scheduling, stopping/promotion policy, and experiment bookkeeping.

No graph-owning controller is introduced.

### In-place OpenES learning-rate control

ABI v1 includes the additive symbol:

```text
br_v1_es_set_learning_rate(optimizer, learning_rate)
```

Its contract is deliberately narrow:

- OpenES only; `mu_lambda` rejects the operation;
- learning rate must be finite and strictly positive;
- mutation is allowed only between completed `ask -> tell` generations;
- if an `ask()` batch is pending, mutation fails without consuming or replacing that batch;
- successful mutation changes only the OpenES learning-rate scalar and preserves search mean, RNG state, generation, lifetime best, stagnation, sigma, dimension, and population.

The symbol is additive, so `br_v1_abi_version() == 1` remains unchanged. This is a host-controlled optimizer setting, not an adaptive schedule implemented by the core.

## Checkpoint / replay

The foreign boundary reuses the existing stateful `ProgramBundle` bytes.

The proof exports a learned graph, imports the bytes into a fresh registry, rebuilds `GraphParameterBinding`, and verifies:

- program identity is unchanged;
- binding identity is unchanged;
- learned flat parameters are restored exactly;
- graph output replay matches the pre-export output.

No Python-specific checkpoint schema or pickle artifact becomes canonical state.

## Semantic proof

`scripts/audit_python_ffi.py` remains the direct CFFI semantic proof. It builds `ffi/Cargo.toml`, loads the produced Linux `cdylib` with CFFI, and checks:

1. ABI version/capabilities;
2. wrong-handle rejection;
3. invalid tensor-shape rejection before native tensor construction;
4. canonical graph parameter dimension;
5. non-finite candidate rejection without mutation;
6. in-place OpenES learning-rate validation, pending-batch rejection, and exact next-ask continuity;
7. ES ask/tell with objective calculated in Python;
8. stateful ProgramBundle replay into fresh objects.

The separate installed-wheel proof in `scripts/audit_python_wheel.py` verifies that the same ABI works after real wheel build/install from a fresh virtual environment outside the repository.

## Supported and unsupported scope

Packaged Python support is intentionally narrower than the ABI itself. The currently verified installed-wheel matrix is recorded in `docs/python-wheel-support.md` and `docs/host-support.v1.json`.

The current proof does not establish macOS, Windows, non-x86_64, PyPy, free-threaded Python, zero-copy NumPy/DLPack, PyO3 ergonomics, C++/Go/Unity support, or long-term ABI stability beyond the explicit versioned v1 contract.

Any broader support claim requires its own external-consumer proof before the manifest is widened.
