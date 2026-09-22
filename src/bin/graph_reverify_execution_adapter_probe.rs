#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent::{AgentGraphBuilder, AgentLayerSpec};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::graph_reverify_execution_adapter::{
    execute_graph_reverify_binding, rejoin_graph_reverify_execution_result,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::graph_reverify_runtime_binding::create_graph_reverify_runtime_binding;
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
            "schema": "burn-research.graph-reverify-execution-adapter-probe.v1",
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
    let mut review = ResolutionReviewSession::new("intent-graph-reverify-execution-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let resolution = review.snapshot().workflow;

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id,
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-graph-reverify-execution-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-graph-reverify-execution-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-graph-reverify-execution-probe".to_string(),
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

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let source_receipt = workspace_verify_graph_receipt(
        &mut workspace,
        &graph,
        &registry,
        &input,
        &[0.0, 3.0],
        0.0,
        0.0,
        "source-failed".to_string(),
    )?;
    let source = RuntimeEvidence::from_graph_verifier_receipt_json(&source_receipt)?;
    ensure(source.outcome() == "failed", "source verifier receipt did not fail");
    ensure(inbox.record(source)?, "source evidence was not recorded");

    let intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent",
    )?;

    let corrected = vec![0.0, 2.0];
    let binding = create_graph_reverify_runtime_binding(
        &inbox,
        &intent,
        &workspace,
        &graph,
        &registry,
        &input,
        &corrected,
        0.0,
        0.0,
        "reverify-corrected",
    )?;

    let drifted = vec![0.0, 2.5];
    let workspace_before_drift = workspace.snapshot();
    let drift_attempt = execute_graph_reverify_binding(
        &inbox,
        &intent,
        &binding,
        &mut workspace,
        &graph,
        &registry,
        &input,
        &drifted,
        0.0,
        0.0,
        "reverify-corrected",
    );
    ensure(drift_attempt.is_err(), "drifted candidate entered canonical verifier");
    ensure(
        workspace.snapshot() == workspace_before_drift,
        "drifted execution attempt mutated workspace",
    );

    let inbox_before_execution = inbox.len();
    let result = execute_graph_reverify_binding(
        &inbox,
        &intent,
        &binding,
        &mut workspace,
        &graph,
        &registry,
        &input,
        &corrected,
        0.0,
        0.0,
        "reverify-corrected",
    )?;
    ensure(result.verifier_outcome() == "passed", "corrected reverify did not pass");
    ensure(
        result.after_verifier_receipt_count()
            == result.before_verifier_receipt_count().saturating_add(1),
        "execution did not append exactly one verifier receipt",
    );
    ensure(
        inbox.len() == inbox_before_execution,
        "execution mutated evidence inbox before explicit rejoin",
    );
    ensure(
        result
            .verifier_receipt_json()
            .contains("\"authority\":\"wasm_verifier\""),
        "canonical verifier authority missing",
    );
    ensure(
        result
            .verifier_receipt_json()
            .contains("\"verifier\":\"CompiledGraph.verifyFlat\""),
        "canonical graph verifier identity missing",
    );

    ensure(
        rejoin_graph_reverify_execution_result(
            &mut inbox,
            &intent,
            &binding,
            &result,
        )?,
        "canonical graph verifier evidence did not rejoin",
    );
    ensure(
        inbox.len() == inbox_before_execution + 1,
        "explicit rejoin did not append exactly one evidence observation",
    );

    let inbox_json = inbox.to_json();
    ensure(
        inbox_json.contains("\"graph_verifier_passed\":1"),
        "rejoined passing graph verifier evidence missing",
    );
    ensure(
        inbox_json.contains("\"graph_verifier_failed\":1"),
        "original failed graph verifier evidence missing",
    );
    ensure(
        inbox_json.contains("\"action_selected\":false"),
        "rejoin selected an action automatically",
    );
    ensure(
        inbox_json.contains("\"state_transition\":\"none\""),
        "rejoin mutated Resolution state",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.graph-reverify-execution-adapter-probe.v1",
            "status": "passed",
            "drift_rejected_without_workspace_mutation": true,
            "execution": {
                "binding_fingerprint": result.binding_fingerprint(),
                "execution_fingerprint": result.execution_fingerprint(),
                "verifier_receipt_id": result.verifier_receipt_id(),
                "verifier_outcome": result.verifier_outcome(),
                "before_verifier_receipt_count": result.before_verifier_receipt_count(),
                "after_verifier_receipt_count": result.after_verifier_receipt_count(),
                "canonical_authority": "wasm_verifier",
                "canonical_verifier": "CompiledGraph.verifyFlat"
            },
            "rejoin": {
                "evidence_authority": "observation_only",
                "inbox_entry_count": inbox.len(),
                "automatic_action_selection": false,
                "resolution_state_transition": "none"
            }
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
