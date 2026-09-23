# WASM host communication contract

Status: **contracted communication guidance for the pinned WASM surface v1**

Related contracts:

- `docs/wasm-surface.v1.json`
- `docs/host-support.v1.json`
- `docs/agent-facade.md`

## Purpose

This document explains how each supported consumer reaches the reference machine and how to classify failures at the correct boundary.

The repository has multiple supported consumer surfaces. They are not a ranking and they do not all communicate through WASM.

```text
Rust reference machine
├─ native Rust package consumer
├─ generated wasm-bindgen surface -> verified Node adapter
└─ language-neutral C ABI v1 -> verified Python/CFFI host
```

A support claim means that the named consumer path has been proven on its recorded matrix. It does **not** mean the interpreter, toolchain, or every host runtime is bundled in an artifact.

## Reference-machine ownership

The numerical/execution semantics live in the Rust reference machine.

The following distinctions are intentional:

```text
Rust reference machine
    != generated WASM bindings
    != Node host adapter
    != C ABI v1
    != Python host facade
    != host-owned objective/scheduling policy
```

A failure in one consumer path must not automatically be diagnosed as a failure of the Rust/WASM execution design.

## Pinned WASM surface v1

The declared generated surface is:

`docs/wasm-surface.v1.json`

`wasm-pack` produces `pkg/burn_research.d.ts`. The export checker parses the generated TypeScript declaration into:

`pkg/wasm-surface.bindings.actual.json`

CI requires exact equality between the pinned contract and this declaration projection for:

- exported class names;
- members of every exported class;
- exported free functions;
- the default wasm-bindgen initializer.

Therefore:

```text
declared wasm-surface.v1
        == generated declaration projection
```

is a real conformance gate. This projection describes declared names and members; it does not claim to recover raw WebAssembly import/export signatures or runtime behavior.

After packaging the Node adapter, `scripts/generate_wasm_surface_actual.mjs` loads that exact package and creates `pkg/wasm-surface.actual.json`. The generated runtime surface includes:

- SHA-256 and byte length for the exact WASM, generated JS, TypeScript declarations, and export projection;
- raw WebAssembly import/export names and kinds read from the packaged binary (the WebAssembly reflection API does not expose function signatures);
- capability JSON queried from the loaded WASM for multi-input ingress, input contracts/provenance, and Math interaction;
- separate `math_program_surface_version` and `math_interaction_protocol_version` values;
- Node host contract definitions and hashes for provenance, durable replay, execution receipts, and state handoff;
- a canonical SHA-256 fingerprint over the complete generated description.

CI rebuilds this description from the packaged runtime and checks it again before artifact upload. Host definitions are embedded from their packaged contract files so an agent can inspect the full authority boundary from this snapshot; the file hashes bind those definitions to the sibling artifacts. The fingerprint proves consistency between the packaged bytes, queried capabilities, and included contracts; it is not a signature, provenance proof, or authorization token. `docs/runtime-surface.v1.json` declares the capability inventory and keeps host-owned features distinct from WASM exports.

Adding or removing an exported member without intentionally updating the v1 contract is a WASM surface drift failure.

The v1 name means the surface is contracted, not frozen forever. A future incompatible public surface should be treated as an explicit contract-version decision rather than accidental drift.

## Node communication path

Node is the verified WASM host.

The supported local-package path is:

```text
Node application
    -> hosts/node/node.mjs
    -> import generated burn_research.js
    -> read burn_research_bg.wasm from the filesystem
    -> initSync({ module: wasmBytes })
    -> generated wasm-bindgen exports
    -> Rust/WASM reference-machine execution
```

This is intentional.

The package is generated with the wasm-bindgen `web` target, whose default initializer is async/browser-oriented. The Node adapter deliberately avoids relying on default `file://` fetch behavior for a local package. It reads the sibling WASM bytes itself and supplies them to `initSync`.

Do **not** classify this adapter as a workaround that should be removed merely because `__wbg_init` also exists. The Node adapter is the verified communication boundary for the packaged Node host.

A Node failure with WASM surface conformance still green should first be investigated as a Node adapter/package/distribution problem, not as a missing WASM primitive.

## Browser status

`burn_research.js` contains the generated browser-oriented wasm-bindgen entry and the default async initializer.

That does not currently imply verified browser-host support.

The support manifest deliberately separates:

```text
generated browser entry
    !=
verified browser host
```

An agent must not promote browser support merely because the generated web artifact exists.

## Native Rust communication path

A native Rust consumer does not communicate through WASM.

Its path is:

```text
external Cargo project
    -> packaged burn-research .crate
    -> public Rust package API
    -> Rust reference machine
```

The native package proof is `scripts/audit_rust_package.py`.

A failure in this path belongs first to the packaged Rust/public-API boundary. It is not a WASM host failure unless an independent WASM proof also fails.

Historical names such as `WasmTensor` may still appear in the public Rust API. That name does not change the communication path: native Rust package consumption remains native.

## Python communication path

Python does not communicate through the WASM surface.

Its verified path is:

```text
Python application
    -> burn_research_ffi.host
    -> typed Python facade
    -> CFFI
    -> burn-research.ffi.v1
    -> Rust reference machine
```

Python support therefore means:

- the installed wheel consumer is proven;
- the C ABI/CFFI path is proven;
- the recorded Python matrix is supported.

It does **not** mean:

- Python is the primary runtime;
- Python owns the reference-machine semantics;
- a Python interpreter is bundled with WASM or Rust artifacts;
- Python should be used to diagnose a WASM surface mismatch;
- every host must copy the Python integration model.

Python was the first cross-language ABI consumer proven in repository history. That is provenance, not architectural rank or future prioritization.

## Agent-facing WASM facade

The typed WASM facade exists so WASM callers do not need to construct protocol bytes manually.

Typical WASM-host flow:

```text
agentCapabilities()
    -> AgentLayerSpec typed constructor
    -> LayerRegistry.initAgentLayer(...)
    -> AgentGraphBuilder
    -> CompiledGraph
    -> run / verify
```

The facade is an ergonomics layer. Registry/execution contracts and Burn-backed semantics remain authoritative below it.

For multiple external tensor inputs, `SemanticIngressManifestV2` maps logical port IDs to the exact `MultiInputGraphPlan.v1` slots. Its read-only `status`, `inputPortStatus`, and `consumerCompatibility` explain the current binding. Its `run` and `verifyFlat` methods recheck bridge coverage, then delegate to `CompiledMultiInputGraph`; they do not introduce another numerical execution engine. The packaged `interactive_multi_input_ingress.mjs` exposes this flow as a persistent JSON Lines session through the verified Node adapter. See `docs/interactive-multi-input-ingress.md` in the repository for a reproducible transcript.

`CompiledMultiInputGraph.explainPlan(registry)` returns `burn-research.multi-input-plan-explain.v1` before a bundle is bound. It reports the exact compiled program identity, declared input shapes, ordered graph steps, current registry binding, and partial static output shape projections. Known shape mismatches are labeled `incompatible`; operators without a sound static rule are labeled `unknown`. The reported `output_f32_payload_bytes` covers only the projected output tensor and is not a runtime allocation estimate. See `docs/multi-input-plan-explain.v1.json` for the proof boundary; call `preflight` with the actual bundle and `run` for Burn's execution verdict.

Raw compatibility surfaces may remain present even when the typed facade is preferred. Their existence is not evidence of duplicated execution engines.

## State-bound signed input claims

The Node host supports the additive `burn-research.signed-input-claim.v2` schema when an issuer explicitly lists it in `claim_schemas`. A trust policy that omits `claim_schemas` remains v1-only. Claim v2 signs `active_state_checkpoint_bytes_sha256`, the SHA-256 of the current stateful multi-input ProgramBundle bytes.

When a configured issuer allows v2, the host returns the active digest in `host_provenance` after graph creation, inspection, and restore. This exposes only the digest, not raw checkpoint bytes. The host checks the signed digest at bind, then recomputes it before run, verify, and checkpoint operations. A restore or other state change therefore requires a new v2 claim for the resulting state.

`--require-state-bound-inputs` is a trusted runner startup option. It requires every configured issuer to allow v2 and every runtime input to carry a current v2 claim. Without it, v1 remains accepted according to issuer policy. The v2 execution receipt records each input's signed state digest and, when all inputs share one digest, `input_state_checkpoint_bytes_sha256`; `state_checkpoint_bytes_sha256` continues to identify the post-execution checkpoint.

These hashes establish byte-level state binding and freshness. They do not authenticate who created a checkpoint, prove semantic equivalence, or grant action authority. Direct WASM callers remain outside this Node host gate. See `docs/ingress-provenance.v2.json` for the machine-readable contract.

## Failure classification

Use this order when an agent investigates a failure:

| Observed failure | Primary boundary to inspect |
| --- | --- |
| `wasm-surface.bindings.actual.json` differs from `wasm-surface.v1.json` | WASM declared export-surface drift |
| `wasm-surface.actual.json` fingerprint or artifact digest differs from the loaded package | stale or mismatched runtime-surface artifact |
| runtime-surface capability entrypoint is absent | runtime capability-contract drift |
| generated surface conforms, but `node.mjs` cannot initialize/load/run | Node adapter or packaged WASM distribution |
| external Cargo package consumer fails | native Rust package/public API |
| Python installed-wheel/CFFI proof fails | C ABI / CFFI / Python host |
| browser-generated entry exists but browser scenario is unproven | no support claim; browser verification is missing |
| objective/reward/schedule behavior differs | host policy/orchestration |
| graph/math execution disagrees across proven consumers | reference-machine semantic investigation |
| signed ingress proof fails before `bind` | Node host trust policy, issuer signature, subject, nonce, revision, or current manifest |
| state-bound signed claim fails before `bind` or `run` | Node host issuer schema allowlist or active checkpoint digest mismatch |

Do not jump across these boundaries without evidence.

## Intentional design choices that are not bugs

The following are deliberate:

- wasm-bindgen `web` artifact target plus a Node-specific filesystem/`initSync` adapter;
- Node, native Rust, and Python using different host-appropriate communication paths;
- Python using C ABI/CFFI instead of WASM;
- browser code being generated without a verified browser-host claim;
- host-owned objective/reward/scheduling policy;
- support claims not implying bundled interpreters or toolchains;
- exact declared-vs-generated WASM surface conformance.

## Standalone artifact copy

The verified Node package carries this document as:

`pkg/wasm-host-communication.md`

and the packaged `host-support.v1.json` exposes the artifact-local pointer:

```text
support_semantics.packaged_communication_contract
    = wasm-host-communication.md
```

This is intentional so an agent that receives only the packaged artifact can still recover the communication and failure-classification contract without access to the repository checkout.

The package also carries `wasm-surface.v1.json`, `runtime-surface.v1.json`, `wasm-surface.bindings.actual.json`, and the generated `wasm-surface.actual.json`. The Node package audit fails if these descriptions, their package allowlist entries, the manifest pointer, or the core communication statements are missing or inconsistent.

## Authority

For support status and verified matrices, use `docs/host-support.v1.json`.

For the pinned WASM export contract, use `docs/wasm-surface.v1.json` and its generated declaration projection `pkg/wasm-surface.bindings.actual.json`. For the self-describing snapshot of the exact packaged runtime, use `pkg/wasm-surface.actual.json` with `docs/runtime-surface.v1.json`.

For execution/identity semantics, follow the underlying graph, binding, Math Program, ProgramBundle, and Burn contracts rather than inferring semantics from host adapter names.

The optional signed ingress policy is enforced by the Node JSON Lines runner before it calls `MultiInputInputBundle.bindInput`. Its `host_provenance` state identifies issuer verification; the WASM ingress `status` continues to report tensor and metadata contract readiness. A direct WASM caller does not pass through this host gate. The packaged `ingress-provenance.v1.json` records the exact signed-claim and trust-policy boundary.

The runner can also pin one host subject and keep a private durable replay ledger, passed as trusted startup arguments. The host explicitly initializes it once with `init_ingress_replay_ledger.mjs`; runner startup rejects missing ledgers. It commits each accepted nonce and revision before returning a successful `bind` and checks the current ledger again during `run` and `verify`. A failed commit clears the affected input. The packaged `ingress-replay-ledger.v1.json` describes persistence, operator recovery of a crash lock, and the host filesystem trust boundary.

In durable mode, a successful `run` also returns a host execution receipt committed under the same lock as the input freshness check. It records the exact `programIdentity`, input claim digests, observed f32 output digest, and SHA-256 of the exact stateful multi-input ProgramBundle bytes. `op=checkpoint` exports those bytes; compare its `checkpoint_bytes_sha256` with the run receipt before storing the checkpoint. This byte digest is correlation only: it is not a signature, semantic state identity, or authorization. `op=receipt` retrieves the receipt by ID after restart and can compare proposed output values. See `host-execution-receipt.v2.json` for the authority and limits; historical v1 receipts remain readable.

The host retains checkpoint bytes privately for receipt-bound restore. A trusted startup flag `--allow-checkpoint-restore` enables `op=restore` by receipt ID independently of `--allow-state-checkpoint-export`. The ledger commits a restore event before the new graph session replaces the old one. The first later run receipt names its restored state parent; subsequent receipts name their immediately prior successful run while preserving the restore event ID. A run that started numerical execution but failed to commit a receipt discards its session. These fields record causal internal state ancestry, while signed input claim v1 still binds input to the structural manifest only. See `host-checkpoint-restore.v1.json` for proof state, valid transitions, and limits.

Durable mode also supports state handoff between graph ticks. Preserve the run's `output_f32_le_base64`, then bind it to a child port with `role: "state"`, the matching rank-4 shape, and `handoff_receipt_id` set to the parent receipt ID. The child input must still pass the configured issuer signature for its own manifest. The host compares exact f32 bytes, atomically records the signed child claim and its lineage, and includes `handoff_id` in the later execution receipt. A lane identified by host subject, input source, logical port ID, and branch only accepts a strictly newer parent receipt. This host record demonstrates byte continuity; it does not grant action authority or prove causal meaning. See `host-state-handoff.v1.json`.
