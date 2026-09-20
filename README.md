# ai_agent_builder

Reference execution and research infrastructure for deterministic math/graph workloads, trainable state, external host consumption, and proof-oriented runtime experiments.

This README is a navigation hub. Detailed semantics and support claims live in the documents below.

## Start here

- [Burn contract baseline](docs/burn-contract-baseline.md) — numerical/backend authority and current non-autodiff execution baseline.
- [Agent contracts v1](docs/agent-contracts.v1.json) — machine-readable agent capability/contract surface.
- [Agent layout contracts v1](docs/agent-layout-contracts.v1.json) — machine-readable layout/structure contracts.
- [Agent workspace](docs/agent-workspace.md) — workspace metadata/control memory and its separation from execution truth.
- [ES + graph host orchestration](docs/es-graph-host-orchestration.md) — graph/binding/optimizer lifecycle and host-owned policy boundary.

## Host and distribution contracts

- [Host support manifest](docs/host-support.v1.json) — **authoritative source for currently supported hosts and verified matrices**.
- [Native Rust package support](docs/rust-package-support.md) — packaged `.crate` consumer boundary.
- [Python host layer](docs/python-host-layer.md) — first-class typed Python host role and orchestration boundary.
- [Python FFI v1](docs/python-ffi-v1.md) — versioned C ABI/CFFI boundary.
- [Python facade v1](docs/python-facade-v1.md) — typed Python facade over ABI v1.
- [Python wheel support](docs/python-wheel-support.md) — installed-wheel consumer proof and supported matrix.
- [WASM surface v1](docs/wasm-surface.v1.json) — machine-readable WASM surface contract.

## Runtime proof and evidence

- [Integrated runtime architecture artifact proof](docs/runtime-architecture-artifact-proof.md) — external-consumer proof for identity/state boundaries, ProgramBundle behavior, optimizer lifecycle/control, and Math v9 replay.
- [Research and evidence documents](docs/) — performance, rollout, OpenES, graph-runtime, and other historical research slices.

Research documents are evidence records, not automatic support or production-policy claims.

## Repository mapper

The repository includes a deterministic Python mapper:

```bash
# Frozen Rust-focused v1 profile
python3 scripts/repo_map.py --format summary

# Opt-in cross-surface v2 profile
python3 scripts/repo_map.py --profile cross-surface --format summary

# Machine-readable or Graphviz output
python3 scripts/repo_map.py --profile cross-surface --format json
python3 scripts/repo_map.py --profile cross-surface --format dot
```

The mapper inventories repository structure and dependency/evidence edges. It does not infer host support or policy ownership from file names; [`docs/host-support.v1.json`](docs/host-support.v1.json) remains authoritative for support status.

## Architectural separation

The current contracts intentionally keep these concepts separate:

```text
structure != mutable learned state
state != program/binding identity
ProgramBundle != optimizer search state
ProgramBundle != AgentWorkspace state
optimizer control capability != host schedule policy
checkpoint != authorization/proof authority
```

For the exact runtime evidence behind these boundaries, see the [runtime architecture artifact proof](docs/runtime-architecture-artifact-proof.md).
