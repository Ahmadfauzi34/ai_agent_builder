# Native Rust package support contract

Status: **supported native Rust package contract**

Related work: #179, #181, #182, #183

## Purpose

This document defines what it means for `burn-research` to be a **supported native Rust package** without conflating that claim with a stable cross-language ABI.

The repository proved semantic host portability in #181: an external Rust integration test can compose `CompiledGraph`, `GraphParameterBinding`, `EsOptimizer`, host-owned objective policy, and `ProgramBundle` replay through the public crate surface.

The packaged-consumer audit was then completed and merged. Current `docs/host-support.v1.json` records the native Rust package as `supported` with `scripts/audit_rust_package.py` as its consumer proof. The package contract therefore describes an established support boundary, not a pending promotion.

## Support layers

Keep these layers distinct:

```text
semantic host portability
        !=
packaged Rust dependency support
        !=
stable cross-language ABI
        !=
Python binding support
```

This slice addresses only the second layer.

## Supported Rust package contract

Rust package support means all of the following are true:

1. `Cargo.toml` continues to produce an `rlib` suitable for native Rust dependency use;
2. `cargo package` succeeds for the repository package;
3. a temporary Cargo project can depend on the extracted packaged crate rather than repository source paths;
4. that project can import only public `burn_research` modules/types;
5. the consumer can build a graph, derive canonical parameter coordinates, apply a finite candidate, and execute the graph;
6. the consumer can export/import `ProgramBundle(include_state=true)` and replay the learned state;
7. existing WASM/Node and native host tests remain green.

The package audit intentionally uses a temporary project outside the repository tree so it cannot rely on `pub(crate)`, test-only modules, repository-relative source imports, or workspace-local implementation details.

## Public entrypoint expectations

The supported native composition currently relies on public APIs including:

- `burn_research::agent::{AgentGraphBuilder, AgentLayerSpec}`;
- `burn_research::graph::CompiledGraph`;
- `burn_research::graph_parameters::GraphParameterBinding`;
- `burn_research::registry::LayerRegistry`;
- `burn_research::es::optimizer::EsOptimizer`;
- `burn_research::program_bundle::{export_program_bundle, import_program_bundle}`;
- `burn_research::WasmTensor` as the current rank-4 tensor bridge type, despite its historical WASM-oriented name.

Package support does not freeze every Rust type or module forever. Until a stronger Rust semver/public-API policy is adopted, consumers should treat version `0.x` changes as potentially breaking and should follow the repository's documented architecture boundaries.

## Ownership remains unchanged

Native package support does not move orchestration policy into the core.

Core owns:

- graph structure and structural identity;
- canonical trainable-owner ordering/layout;
- finite/fail-closed candidate application;
- optimizer lifecycle/cardinality/finite-fitness checks;
- stateful checkpoint import/export semantics.

The Rust host owns:

- dataset selection;
- objective/reward/penalty policy;
- evaluation scheduling;
- stopping and promotion policy;
- experiment bookkeeping.

The accepted architecture remains `KEEP_HOST_ORCHESTRATION`.

## Package proof

`scripts/audit_rust_package.py` is the package-level proof.

It performs:

```text
cargo package
    -> locate .crate artifact
    -> extract into a temporary directory
    -> create a separate Cargo consumer project
    -> depend on the extracted package by path
    -> resolve a fresh consumer dependency lockfile
    -> cargo run --locked
    -> graph + binding + candidate apply + run
    -> stateful ProgramBundle export/import
    -> fresh replay
```

The consumer deliberately resolves its own dependency graph instead of inheriting the repository `Cargo.lock`. This keeps the proof representative of a real external Cargo consumer while `--locked` keeps the resolved graph fixed for the actual build/run within the audit.

The proof is deliberately separate from `tests/native_es_graph_host_orchestration.rs`:

- the integration test proves semantic portability through the crate's public surface;
- the package audit proves the **packaged source artifact** remains consumable as an external dependency.

Both proofs remain required support gates. Current main satisfies them, and `docs/host-support.v1.json` records Rust as a supported native package surface.

## What Rust support does not mean

It does **not** mean:

- Rust symbols form a stable C ABI;
- Rust struct layout is stable across compiler/crate versions;
- `cdylib` exports are suitable for foreign-language consumers;
- Python can safely call Rust internals directly;
- C++, Go, Unity, or other engines are supported native consumers;
- the Rust module tree is permanently frozen;
- host policy belongs in Rust core.

In particular:

```text
public Rust API != stable ABI
```

Foreign-language consumers must not bind directly to compiler-generated Rust symbols or assume Rust memory layout.

## ABI handoff: verified Python consumer

The initial ABI/Python handoff is now complete.

The repository followed this ordering:

```text
supported Rust package
    -> language-neutral versioned C ABI v1
    -> Python CFFI semantic proof
    -> installed Python wheel proof
    -> typed verified Python host layer
```

Python was the first cross-language ABI consumer verified in repository history. That records implementation provenance only; it is not a host ranking, future prioritization rule, or permission to contaminate the core with Python-specific policy.

The current foreign boundary is `burn-research.ffi.v1`, consumed by the supported Python package through CFFI. The ABI remains explicitly versioned and experimentally stable rather than a promise that all future foreign-language consumers or Rust layouts are frozen.

No C++/Go/engine consumer is implied by the Rust package support contract. Additional consumers require their own concrete lifecycle, ownership, error, buffer, distribution, and external-consumer proof before becoming supported.

## ABI design constraints inherited from current architecture

Future ABI work must preserve at least these separations:

```text
Rust internal representation
    != public Rust package API
    != stable cross-language ABI
    != host objective/controller policy
```

It should also preserve:

- canonical graph-parameter coordinates from `GraphParameterBinding`;
- finite-only candidate boundaries;
- staged/fail-closed checkpoint import;
- separation of program identity, binding identity, mutable learned state, and authorization/proof;
- host-owned objective policy;
- no reliance on `ProgramBundle` byte SHA as semantic state identity.

## Continuation rule

The Rust package, ABI v1, installed Python wheel, and typed Python host have all crossed their initial support proofs. Future work should therefore preserve the existing separation instead of replaying the old promotion sequence.

For native Rust package changes:

1. keep `scripts/audit_rust_package.py` green against the packaged `.crate` artifact;
2. preserve public-package consumption without repository-internal access;
3. keep host objective/reward/scheduling policy outside the reference machine;
4. do not infer a stable C ABI from Rust public API stability;
5. keep `docs/host-support.v1.json` authoritative for actual supported hosts and matrices.

For foreign-host changes, treat `burn-research.ffi.v1` and the Python host contract as separate layers. Broader platform, Python-version, or language support requires new external-consumer proof before the support manifest is widened.

## Communication-path clarification

Native Rust consumers use the packaged Rust API directly and do not route through WASM. Python uses CFFI over ABI v1 and also does not route through WASM. The verified Node host is the supported consumer that reaches the generated wasm-bindgen surface. See [WASM host communication](wasm-host-communication.md).
