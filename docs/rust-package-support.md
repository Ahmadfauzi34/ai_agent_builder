# Native Rust package support contract

Status: **candidate support contract pending CI proof**

Related work: #179, #181, #182

## Purpose

This document defines what it means for `burn-research` to be a **supported native Rust package** without conflating that claim with a stable cross-language ABI.

The repository already proved semantic host portability in #181: an external Rust integration test can compose `CompiledGraph`, `GraphParameterBinding`, `EsOptimizer`, host-owned objective policy, and `ProgramBundle` replay through the public crate surface.

That proof is necessary but not sufficient for a package-support claim. A package contract must also prove that the actual output of `cargo package` can be consumed by a separate Cargo project without repository-internal access.

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
    -> compile/run offline against already resolved dependencies
    -> graph + binding + candidate apply + run
    -> stateful ProgramBundle export/import
    -> fresh replay
```

The proof is deliberately separate from `tests/native_es_graph_host_orchestration.rs`:

- the integration test proves semantic portability through the crate's public surface;
- the package audit proves the **packaged source artifact** remains consumable as an external dependency.

Both proofs are required before Rust is marked supported in the host-support contract.

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

## ABI handoff: Python first

When the Rust package contract is proven and cross-language ABI work begins, **Python is the first-priority consumer**.

The required ordering is:

```text
supported Rust package
    -> design ABI/FFI v1
    -> prove Python consumer/binding FIRST
    -> then consider C/C++/Go/engine consumers
```

Python-first is a prioritization rule, not permission to contaminate the core with Python-specific policy.

The eventual low-level ABI should remain language-neutral and versioned. Python is the first consumer used to validate that boundary because it is the highest-priority external host.

No decision is made here between PyO3/maturin, a thin C ABI consumed from Python, or another packaging mechanism. That choice belongs to the ABI/Python design slice and should be made from concrete lifecycle, ownership, error, buffer, and distribution requirements.

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

Do not begin Python implementation merely because this document exists. First require the external package audit to pass in CI and update `docs/host-support.v1.json` to record Rust as a native package surface rather than a WASM host.

Once that proof is green, open a separate ABI/Python issue/PR whose description explicitly covers:

- Python-first use cases;
- handle/lifetime ownership;
- errors across the boundary;
- tensor/buffer transfer semantics;
- graph/binding/checkpoint lifecycle;
- versioning and compatibility;
- wheel/distribution strategy;
- what remains host policy;
- proof gates required before calling Python supported.
