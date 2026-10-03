# Registry Kontrak Kanonis

Indeks kanonis seluruh kontrak JSON di `docs/contracts/`. Setiap kontrak yang
di-embed Rust (`include_str!`) atau dipaketkan ke `pkg/docs/` harus tercatat di sini.
`scripts/check_contract_registry.py` (CI) menegakkan: (1) setiap file
`docs/contracts/*.json` terdaftar di tabel ini; (2) setiap
`include_str!("../docs/contracts/...")` di `src/` merujuk ke file yang ada dan
terdaftar; (3) tidak ada entri basi (file yang sudah tidak ada).

Layout paket tidak berubah: packager menyalin `docs/contracts/<nama>.json` →
`pkg/docs/<nama>.json`, sehingga `artifact_identity.files` dan audit yang
membaca kontrak dari paket tetap stabil.

| Kontrak | Dipaketkan | Di-embed oleh | Penegak |
|---|---|---|---|
| `agent-contracts.v1.json` | ya | `src/contracts.rs` | scripts/check_wasm_surface.py, `src/contracts.rs` |
| `agent-fault-contract.v1.json` | ya | — | scripts/audit_operation_contract_registry.mjs |
| `agent-input-contract.v1.json` | tidak | `src/ingress/input_contract.rs` | `src/ingress/input_contract.rs` |
| `agent-input-port-consumer.v1.json` | tidak | `src/ingress/input_port_consumer.rs` | `src/ingress/input_port_consumer.rs` |
| `agent-input-port-edge-binding.v1.json` | tidak | `src/ingress/input_port_edge_binding.rs` | `src/ingress/input_port_edge_binding.rs` |
| `agent-input-port-routing.v1.json` | tidak | `src/ingress/input_port_routing.rs` | `src/ingress/input_port_routing.rs` |
| `agent-input-port.v1.json` | ya | `src/ingress/input_port.rs` | scripts/audit_operation_contract_registry.mjs, `src/ingress/input_port.rs` |
| `agent-interaction-contract.v1.json` | tidak | — | — |
| `agent-introspection-contract.v1.json` | tidak | `src/introspection.rs` | `src/introspection.rs` |
| `agent-layer-catalog.v1.json` | ya | `src/introspection.rs` | scripts/audit_operation_contract_registry.mjs, `src/introspection.rs` |
| `agent-layout-contracts.v1.json` | ya | `src/contracts.rs` | scripts/audit_operation_contract_registry.mjs, scripts/check_wasm_surface.py, `src/contracts.rs` |
| `agent-response-intent.v1.json` | tidak | `src/dispatch/agent_response_intent.rs` | `src/bin/agent_response_intent_probe.rs`, `src/dispatch/agent_response_intent.rs` |
| `agent-semantic-execution-context.v1.json` | tidak | `src/semantic/semantic_execution_context.rs` | `src/semantic/semantic_execution_context.rs` |
| `agent-semantic-lifecycle.v1.json` | tidak | `src/semantic/semantic_lifecycle.rs` | `src/semantic/semantic_lifecycle.rs` |
| `baseline-candidate-verification.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `graph-reverify-execution-adapter.v1.json` | tidak | `src/graph/graph_reverify_execution_adapter.rs` | `src/bin/graph_reverify_execution_adapter_probe.rs`, `src/graph/graph_reverify_execution_adapter.rs` |
| `graph-reverify-runtime-binding.v1.json` | tidak | `src/graph/graph_reverify_runtime_binding.rs` | `src/bin/graph_reverify_runtime_binding_probe.rs`, `src/graph/graph_reverify_runtime_binding.rs` |
| `host-branch-promotion-lineage.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `host-checkpoint-restore.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `host-execution-receipt.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `host-execution-receipt.v2.json` | ya | — | scripts/audit_node_host.mjs |
| `host-operation-contract-registry.v1.json` | ya | — | scripts/audit_operation_contract_registry.mjs |
| `host-state-handoff.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `host-support.v1.json` | ya | — | scripts/audit_node_host.mjs, scripts/audit_operation_contract_registry.mjs |
| `ingress-provenance.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `ingress-provenance.v2.json` | ya | — | scripts/audit_node_host.mjs |
| `ingress-replay-ledger.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `layer-registry-inventory.v1.json` | tidak | `src/registry/runtime_contract/inventory.rs` | `src/registry/runtime_contract/inventory.rs` |
| `layer-registry-operation-binding.v1.json` | tidak | `src/registry/runtime_contract/binding.rs` | `src/registry/runtime_contract/binding.rs` |
| `math-proof.v1.json` | tidak | `src/evidence/proof_provenance.rs` | `src/evidence/proof_provenance.rs` |
| `multi-input-execution-trace.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `multi-input-graph-plan.v1.json` | ya | — | scripts/audit_operation_contract_registry.mjs |
| `multi-input-plan-explain.v1.json` | ya | — | scripts/audit_operation_contract_registry.mjs |
| `proof-provenance.v1.json` | tidak | `src/evidence/proof_provenance.rs` | `src/evidence/proof_provenance.rs` |
| `resolution-runtime-bridge.v1.json` | tidak | `src/resolution/resolution_runtime_bridge.rs` | `src/bin/resolution_runtime_bridge_probe.rs`, `src/resolution/resolution_runtime_bridge.rs` |
| `response-dispatch-executor-preflight.v1.json` | tidak | `src/dispatch/response_dispatch_executor_preflight.rs` | `src/bin/response_dispatch_executor_preflight_probe.rs`, `src/dispatch/response_dispatch_executor_preflight.rs` |
| `response-dispatch-request.v1.json` | tidak | `src/dispatch/response_dispatch_request.rs` | `src/bin/response_dispatch_request_probe.rs`, `src/dispatch/response_dispatch_request.rs` |
| `response-dispatch-requirements.v1.json` | tidak | `src/dispatch/response_dispatch_requirements.rs` | `src/bin/response_dispatch_requirements_probe.rs`, `src/dispatch/response_dispatch_requirements.rs` |
| `response-intent-execution-gate.v1.json` | tidak | `src/dispatch/response_intent_execution_gate.rs` | `src/bin/response_intent_execution_gate_probe.rs`, `src/dispatch/response_intent_execution_gate.rs` |
| `revision-dispatch-execution-adapter.v1.json` | tidak | `src/dispatch/revision_dispatch_execution_adapter.rs` | `src/bin/revision_dispatch_execution_adapter_probe.rs`, `src/dispatch/revision_dispatch_execution_adapter.rs` |
| `revision-execution-evidence-rejoin.v1.json` | tidak | `src/dispatch/revision_execution_evidence_rejoin.rs` | `src/bin/revision_execution_evidence_rejoin_probe.rs`, `src/dispatch/revision_execution_evidence_rejoin.rs` |
| `runtime-evidence-interpretation.v1.json` | tidak | `src/evidence/runtime_evidence_interpretation.rs` | `src/bin/runtime_evidence_interpretation_probe.rs`, `src/evidence/runtime_evidence_interpretation.rs` |
| `runtime-resolution-evidence.v1.json` | tidak | `src/evidence/runtime_resolution_evidence.rs` | `src/bin/runtime_resolution_evidence_probe.rs`, `src/evidence/runtime_resolution_evidence.rs` |
| `runtime-surface.v1.json` | ya | — | scripts/audit_node_host.mjs, scripts/audit_operation_contract_registry.mjs |
| `semantic-ingress-manifest.v1.json` | tidak | `src/semantic/semantic_ingress_manifest.rs` | `src/semantic/semantic_ingress_manifest.rs` |
| `semantic-ingress-manifest.v2.json` | ya | `src/semantic/semantic_ingress_manifest_v2.rs` | `src/semantic/semantic_ingress_manifest_v2.rs` |
| `transactional-graph-checkpoint-branch.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `transactional-graph-mutation.v1.json` | ya | — | scripts/audit_node_host.mjs |
| `unified-math-interaction.v1.json` | tidak | — | — |
| `wasm-surface.v1.json` | ya | — | scripts/audit_node_host.mjs, scripts/check_wasm_surface.py |

## Kontrak tanpa konsumen (kandidat review)

Kontrak berikut saat ini tidak di-embed Rust mana pun dan tidak dipaketkan.
Dibiarkan tercatat di sini sampai ada keputusan: dihapus atau dipakai.

- `agent-interaction-contract.v1.json`
- `unified-math-interaction.v1.json`
