# Python installed-wheel support proof

Status: **supported for the verified initial matrix only**

Related: #183, #184, #185, #186

## Purpose

This slice advances the Python-first foreign boundary from the semantic CFFI proof in #185 to a real installed-wheel distribution proof.

It does not change ABI semantics. It packages the existing `br_v1_*` ABI into a wheel, installs that wheel into a fresh virtual environment outside the repository, and runs the established graph/binding/ES/checkpoint path through the installed package.

Keep these layers distinct:

```text
language-neutral ABI v1
    != Python CFFI semantic proof
    != installed Python wheel support
    != broad Python/platform support matrix
```

## Packaging mechanism

`ffi/pyproject.toml` uses Maturin with `bindings = "cffi"`.

The canonical ABI contract remains `ffi/include/burn_research_ffi.h`. Maturin feeds `target/header.h` directly to `cffi.FFI().cdef()`, which does not run a C preprocessor. Therefore `ffi/build.rs` derives a declaration-only CFFI view from the canonical header by removing preprocessor directives and the C++ `extern "C"` wrapper. The versioned declarations themselves are not rewritten.

This preserves one reviewed foreign contract while adapting only its packaging representation for CFFI.

The Python package exposes Maturin's generated CFFI `ffi` and `lib` objects:

```python
from burn_research_ffi import ffi, lib
```

No Python-specific graph/controller semantics are introduced.

## Installed-consumer proof

`scripts/audit_python_wheel.py` performs:

```text
maturin build
    -> wheel artifact
    -> fresh temp virtualenv outside repository
    -> pip install wheel
    -> remove PYTHONPATH / disable user-site
    -> import from site-packages
    -> ABI capabilities/version
    -> graph + GraphParameterBinding
    -> EsOptimizer.ask
    -> Python-owned objective
    -> EsOptimizer.tell
    -> apply canonical best vector
    -> ProgramBundle export(include_state=true)
    -> fresh registry import
    -> rebuild binding
    -> exact learned vector / identity / output replay
```

The consumer asserts that the imported module path is not inside the repository checkout.

It does not rely on repository-relative Python imports, `PYTHONPATH` pointing at the checkout, `ffi/target` at runtime, Rust `pub(crate)` APIs, or an editable/develop install.

## Verified support matrix

The support claim is deliberately narrow:

```text
OS:            Linux (GitHub ubuntu-latest)
architecture:  x86_64
Python:        CPython 3.12
binding:       CFFI
ABI:           burn-research.ffi.v1
package:       burn-research-ffi wheel
orchestration: host-owned
```

The installed-wheel proof passed on this matrix. `docs/host-support.v1.json` records only this verified slice.

This must not be interpreted as proof for macOS, Windows, other architectures, PyPy, free-threaded Python, or every CPython version accepted by package metadata.

## Inherited boundaries

The wheel preserves the existing boundaries:

- Rust core remains Python-independent;
- graph structure/identity remains core-owned;
- parameter dimension/order comes only from `GraphParameterBinding`;
- candidate application remains finite-only and fail-closed;
- Python owns objective/dataset/evaluation policy;
- `ProgramBundle` remains the canonical stateful checkpoint payload;
- program identity, binding identity, mutable state, bundle SHA, and authorization/proof remain distinct;
- no Rust panic is intended to unwind across the foreign ABI;
- tensor transfer remains explicit copy-based rank-4 f32 in v1.

## Non-goals / unverified scope

This support proof does not add or verify:

- PyO3;
- a graph-owning controller;
- NumPy zero-copy or DLPack;
- Math Program changes;
- new optimizer algorithms;
- Python-specific checkpoint/state identity;
- C++/Go/engine support;
- macOS or Windows wheels;
- non-x86_64 wheels;
- PyPy or free-threaded Python;
- a broad CPython version matrix;
- PyPI publication.

## Support-status rule

Python is `supported` only for the explicit matrix above. Any broader support claim requires its own installed-consumer proof and CI matrix expansion before the manifest is widened.
