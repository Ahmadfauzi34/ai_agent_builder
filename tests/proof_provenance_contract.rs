use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
use burn_research::introspection::describe_workspace;
use burn_research::proof_provenance::{
    proof_provenance_capabilities, workspace_proof_ledger, workspace_record_attestation,
    workspace_verify_graph_receipt, workspace_verify_vector_receipt,
};
use burn_research::registry::LayerRegistry;
use burn_research::workspace::AgentWorkspace;
use burn_research::workspace_ops::{workspace_compile, workspace_init_unary};
use burn_research::WasmTensor;

#[test]
fn proof_provenance_contract_is_artifact_discoverable() {
    let caps: serde_json::Value =
        serde_json::from_str(&proof_provenance_capabilities()).unwrap();
    assert_eq!(caps["schema_id"], "burn-research.proof-provenance.v1");
    assert_eq!(
        caps["authorities"]["caller_attestation"],
        "explicit caller claim only"
    );
    assert_eq!(
        caps["authorities"]["graph_receipt"],
        "WASM verifier authority using Burn CompiledGraph as the reference execution"
    );
}

#[test]
fn legacy_claim_attestation_and_verifier_receipt_remain_separate() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let registry = LayerRegistry::new();

    workspace
        .record_proof("legacy-self-claim".into(), true, 999.0, "caller supplied".into())
        .unwrap();
    workspace_record_attestation(
        &mut workspace,
        "explicit-claim".into(),
        true,
        "caller only".into(),
    )
    .unwrap();
    workspace_verify_vector_receipt(
        &mut workspace,
        &[1.0, 2.0],
        &[1.0, 2.0],
        0.0,
        0.0,
        "vector-check".into(),
    )
    .unwrap();

    let ledger: serde_json::Value =
        serde_json::from_str(&workspace_proof_ledger(&workspace)).unwrap();

    assert_eq!(
        ledger["legacy_recordProof_authority"],
        "caller_controlled_legacy"
    );
    assert_eq!(ledger["legacy_proofs"]["rows"].as_array().unwrap().len(), 1);
    assert_eq!(ledger["attestations"]["rows"].as_array().unwrap().len(), 1);
    assert_eq!(
        ledger["verifier_receipts"]["rows"].as_array().unwrap().len(),
        1
    );

    let view: serde_json::Value =
        serde_json::from_str(&describe_workspace(&workspace, &registry)).unwrap();
    assert_eq!(
        view["proof_summary"]["legacy_recordProof_authority"],
        "caller_controlled_legacy"
    );
    assert_eq!(view["proof_summary"]["legacy_passed"], 1);
    assert_eq!(view["proof_summary"]["attestations"], 1);
    assert_eq!(view["proof_summary"]["verifier_receipts"]["passed"], 1);
}

#[test]
fn graph_receipt_binds_burn_reference_and_program_identity() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let mut builder = AgentGraphBuilder::new(2).unwrap();
    let mut registry = LayerRegistry::new();

    let id = workspace.reserve_layer_id(&registry, "relu".into()).unwrap();
    let spec = AgentLayerSpec::relu(id);
    let out = workspace_init_unary(
        &mut workspace,
        &mut builder,
        &mut registry,
        &spec,
        0,
        "relu".into(),
    )
    .unwrap();
    let graph = workspace_compile(&builder, &registry, out).unwrap();
    let input = WasmTensor::new(&[-2.0, 3.0], &[1, 2, 1, 1]);

    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_graph_receipt(
            &mut workspace,
            &graph,
            &registry,
            &input,
            &[0.0, 3.0],
            0.0,
            0.0,
            "graph-reference".into(),
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(receipt["authority"], "wasm_verifier");
    assert_eq!(receipt["verifier"], "CompiledGraph.verifyFlat");
    assert_eq!(receipt["reference_authority"], "burn_compiled_graph");
    assert_eq!(receipt["mutable_state_in_program_identity"], false);
    assert_eq!(receipt["result"]["passed"], true);
    assert_eq!(
        receipt["program_identity"]["schema"],
        "burn-research.program-identity.v1"
    );
    assert!(receipt["reference_fingerprint"]
        .as_str()
        .unwrap()
        .starts_with("fnv1a64:"));
}

#[test]
fn numerical_failure_is_failed_receipt_not_transport_failure() {
    let mut workspace = AgentWorkspace::new(2).unwrap();

    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0, 2.0],
            &[1.0, 2.5],
            0.01,
            0.0,
            "mismatch".into(),
        )
        .unwrap(),
    )
    .unwrap();

    assert_eq!(receipt["result"]["passed"], false);
    assert_eq!(receipt["reference_authority"], "caller_supplied");

    let ledger: serde_json::Value =
        serde_json::from_str(&workspace_proof_ledger(&workspace)).unwrap();
    assert_eq!(
        ledger["verifier_receipts"]["rows"][0]["state"],
        "failed"
    );
}

#[test]
fn invalid_candidate_does_not_mutate_ledger_or_consume_receipt_id() {
    let mut workspace = AgentWorkspace::new(2).unwrap();
    let before = workspace.snapshot();

    assert!(workspace_verify_vector_receipt(
        &mut workspace,
        &[1.0],
        &[f32::NAN],
        0.0,
        0.0,
        "invalid".into(),
    )
    .is_err());
    assert_eq!(workspace.snapshot(), before);

    let receipt: serde_json::Value = serde_json::from_str(
        &workspace_verify_vector_receipt(
            &mut workspace,
            &[1.0],
            &[1.0],
            0.0,
            0.0,
            "valid".into(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["receipt_id"], 1);
}
