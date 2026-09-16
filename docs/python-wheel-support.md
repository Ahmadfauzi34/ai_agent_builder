# Python installed-wheel support proof

Status: **candidate support contract pending installed-wheel CI proof**

Related: #183, #184, #185

## Purpose

This slice advances the Python-first foreign boundary from **semantic CFFI proof** (#185) to a real Python distribution proof.

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

The existing `ffi/include/burn_research_ffi.h` remains the ABI contract. `ffi/build.rs` copies that explicit versioned header to Cargo's target root as `header.h`, which is the supported Maturin override path for CFFI header generation.

This prevents wheel packaging from silently deriving a second foreign contract from Rust implementation details.

The Python package exposes Maturin's generated CFFI `ffi` and `lib` objects only:

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
    -> ES ask
    -> Python-owned objective
    -> ES tell
    -> apply canonical best vector
    -> ProgramBundle export(include_state=true)
    -> fresh registry import
    -> rebuild binding
    -> exact learned vector / identity / output replay
```

The consumer asserts that the imported module path is not inside the repository checkout.

It must not rely on:

- repository-relative Python imports;
- `PYTHONPATH` pointing at the checkout;
- `ffi/target` at runtime;
- Rust `pub(crate)` APIs;
- an in-tree editable/develop install.

## Initial support matrix

This proof is deliberately narrow:

```text
OS:            Linux (GitHub ubuntu-latest)
architecture:  x86_64 runner
Python:        CPython 3.12
binding:       CFFI
ABI:           burn-research.ffi.v1
wheel tag:     native Linux wheel produced by Maturin
```

A green result proves this installed-wheel slice only. It must not be interpreted as proof for macOS, Windows, other architectures, PyPy, free-threaded Python, or every CPython version accepted by package metadata.

## Inherited boundaries

The wheel must preserve all existing boundaries:

- Rust core remains Python-independent;
- graph structure/identity remains core-owned;
- parameter dimension/order comes only from `GraphParameterBinding`;
- candidate application remains finite-only and fail-closed;
- Python owns objective/dataset/evaluation policy;
- `ProgramBundle` remains the canonical stateful checkpoint payload;
- program identity, binding identity, mutable state, bundle SHA, and authorization/proof remain distinct;
- no Rust panic is intended to unwind across the foreign ABI;
- tensor transfer remains explicit copy-based rank-4 f32 in v1.

## Non-goals

This proof does not add:

- PyO3;
- a graph-owning controller;
- NumPy zero-copy or DLPack;
- Math Program changes;
- new optimizer algorithms;
- Python-specific checkpoint/state identity;
- C++/Go/engine support;
- a broad multi-platform wheel matrix;
- a PyPI publication claim.

## Support-status rule

Do not change `docs/host-support.v1.json` from Python `planned` to `supported` merely because the packaging files exist.

First require the installed-wheel workflow to pass on the exact PR head. After that proof is green, record only the narrow verified Python support slice and keep all untested platforms/interpreters explicitly unverified.
