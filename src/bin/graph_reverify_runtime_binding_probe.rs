#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::graph_reverify_runtime_binding::{
    create_graph_reverify_runtime_binding, preflight_graph_reverify_runtime_binding,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::proof_provenance::workspace_verify_graph_receipt;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::registry::LayerRegistry;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_review::ResolutionReviewSession;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::{
    bind_runtime_subject_projection, RuntimeSubjectProjection,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::EvidenceResponseAction;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{
    ResolutionEvidenceInbox, RuntimeEvidence,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::workspace::AgentWorkspace;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::workspace_ops::workspace_compile_for_runtime_subject;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::WasmTensor;

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.graph-reverify-runtime-binding-probe.v1",
            "status": "failed",
            "message": message.as_ref(),
        })
    );
    std::process::exit(1);
}

#[cfg(not(target_arch = "wasm32"))]
fn ensure(condition: bool, message: &str) {
    if !condition {
        fail(message);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    if let Err(error) = run() {
        fail(error);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run() -> Result<(), String> {
    let mut review = ResolutionReviewSession::new("intent-graph-reverify-runtime-binding-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let resolution = review.snapshot().workflow;

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id,
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-graph-reverify-runtime-binding-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-graph-reverify-runtime-binding-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-graph-reverify-runtime-binding-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "customer".to_string(),
        fields: Vec::new(),
    };

    let mut workspace = AgentWorkspace::new(2)?;
    bind_runtime_subject_projection(&mut workspace, &projection)?;

    let mut registry = LayerRegistry::new();
    let spec = AgentLayerSpec::relu(7);
    registry.init_agent_layer(&spec)?;
    let mut builder = AgentGraphBuilder::new(2)?;
    builder.add_unary(&spec, 0, 1)?;
    builder.set_output(1)?;
    let graph =
        workspace_compile_for_runtime_subject(&mut workspace, &builder, &registry, 1)?;

    let input = WasmTensor::new(&[-1.0, 2.0], &[1, 2, 1, 1]);
    let candidate = vec![0.0, 3.0];

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let source_receipt = workspace_verify_graph_receipt(
        &mut workspace,
        &graph,
        &registry,
        &input,
        &candidate,
        0.0,
        0.0,
        "source-failed".to_string(),
    )?;
    let source_evidence = RuntimeEvidence::from_graph_verifier_receipt_json(&source_receipt)?;
    ensure(
        inbox.record(source_evidence)?,
        "source graph evidence was not recorded",
    );

    let intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent",
    )?;

    let workspace_before_binding = workspace.snapshot();
    let inbox_before_binding = inbox.len();

    let binding = create_graph_reverify_runtime_binding(
        &inbox,
        &intent,
        &workspace,
        &graph,
        &registry,
        &input,
        &candidate,
        0.0,
        0.0,
        "repeat-check",
    )?;

    let preflight = preflight_graph_reverify_runtime_binding(
        &inbox,
        &intent,
        &binding,
        &workspace,
        &graph,
        &registry,
        &input,
        &candidate,
        0.0,
        0.0,
        "repeat-check",
    );
    ensure(preflight.ready, "real-handle graph reverify preflight was not ready");
    ensure(
        !preflight.execution_authorized,
        "graph reverify binding unexpectedly authorized execution",
    );
    ensure(
        workspace.snapshot() == workspace_before_binding,
        "graph reverify binding/preflight mutated workspace",
    );
    ensure(
        inbox.len() == inbox_before_binding,
        "graph reverify binding/preflight mutated evidence inbox",
    );

    let drifted_candidate = vec![0.0, 4.0];
    let drifted = preflight_graph_reverify_runtime_binding(
        &inbox,
        &intent,
        &binding,
        &workspace,
        &graph,
        &registry,
        &input,
        &drifted_candidate,
        0.0,
        0.0,
        "repeat-check",
    );
    ensure(!drifted.ready, "candidate drift did not close preflight");
    ensure(
        !drifted.candidate_matches,
        "candidate drift was not detected",
    );

    let mut foreign_registry = LayerRegistry::new();
    let foreign_spec = AgentLayerSpec::gelu(9);
    foreign_registry.init_agent_layer(&foreign_spec)?;
    let mut foreign_builder = AgentGraphBuilder::new(2)?;
    foreign_builder.add_unary(&foreign_spec, 0, 1)?;
    foreign_builder.set_output(1)?;
    let foreign_graph = foreign_builder.compile(&foreign_registry)?;
    let foreign = create_graph_reverify_runtime_binding(
        &inbox,
        &intent,
        &workspace,
        &foreign_graph,
        &foreign_registry,
        &input,
        &candidate,
        0.0,
        0.0,
        "repeat-check",
    );
    ensure(
        foreign.is_err(),
        "foreign graph was rebound to selected graph evidence",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.graph-reverify-runtime-binding-probe.v1",
            "status": "passed",
            "binding": {
                "source_entry_index": binding.source_entry_index(),
                "response_intent_fingerprint": binding.response_intent_fingerprint(),
                "dispatch_fingerprint": binding.dispatch_fingerprint(),
                "requirements_fingerprint": binding.requirements_fingerprint(),
                "input_fingerprint": binding.input_fingerprint(),
                "candidate_fingerprint": binding.candidate_fingerprint(),
                "candidate_len": binding.candidate_len(),
                "binding_fingerprint": binding.binding_fingerprint(),
            },
            "preflight": {
                "ready": preflight.ready,
                "workspace_subject_matches": preflight.workspace_subject_matches,
                "source_evidence_matches": preflight.source_evidence_matches,
                "program_identity_matches": preflight.program_identity_matches,
                "registry_binding_valid": preflight.registry_binding_valid,
                "runtime_program_bound": preflight.runtime_program_bound,
                "input_matches": preflight.input_matches,
                "candidate_matches": preflight.candidate_matches,
                "execution_authorized": preflight.execution_authorized,
            },
            "drift": {
                "candidate_closed": !drifted.ready,
                "candidate_matches": drifted.candidate_matches,
            },
            "foreign_graph_rejected": foreign.is_err(),
            "workspace_unchanged": workspace.snapshot() == workspace_before_binding,
            "inbox_unchanged": inbox.len() == inbox_before_binding,
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
