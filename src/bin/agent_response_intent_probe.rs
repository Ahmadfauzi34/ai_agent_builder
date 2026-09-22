#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::{
    create_agent_response_intent, preflight_response_intent,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution::ResolutionWorkflow;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::EvidenceResponseAction;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{
    ResolutionEvidenceInbox, RuntimeEvidence,
};

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.agent-response-intent-probe.v1",
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
fn projection(
    intent_id: &str,
    workflow_revision: u64,
    subject_identity: &str,
) -> RuntimeSubjectProjection {
    RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: intent_id.to_string(),
        workflow_revision,
        approval_id: "approval-response-intent-probe".to_string(),
        subject_kind: "effective-spec".to_string(),
        subject_identity: subject_identity.to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: subject_identity.to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-response-intent-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "probe".to_string(),
        fields: Vec::new(),
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
    let mut workflow = ResolutionWorkflow::new("intent-response-intent-probe")?;
    workflow.submit()?;
    workflow.finalize_resolution()?;
    let resolution = workflow.snapshot();
    let before_resolution = resolution.clone();

    let projection = projection(
        &resolution.intent_id,
        resolution.revision,
        "spec-response-intent-probe",
    );
    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;

    let failed = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        1,
        "failed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
        false,
        "candidate mismatch",
    )?;
    let passed = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        2,
        "passed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"passed\"}",
        true,
        "candidate match",
    )?;

    ensure(inbox.record(failed)?, "failed evidence was not recorded");
    ensure(inbox.record(passed)?, "passed evidence was not recorded");
    let before_len = inbox.len();

    let revision_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent-probe",
    )?;
    let revision_preflight = preflight_response_intent(&inbox, &revision_intent);
    ensure(revision_preflight.ready, "revision response intent was not preflight-ready");
    ensure(
        revision_preflight.status == "ready_nonexecuting",
        "unexpected revision response intent status",
    );
    ensure(
        !revision_preflight.execution_authorized,
        "preflight must not authorize execution",
    );

    let reverify_first = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent-probe",
    )?;
    let reverify_second = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent-probe",
    )?;
    ensure(
        reverify_first.response_intent_fingerprint()
            == reverify_second.response_intent_fingerprint(),
        "identical response selections produced different fingerprints",
    );
    ensure(
        reverify_first.response_intent_fingerprint()
            != revision_intent.response_intent_fingerprint(),
        "different selected actions produced the same response intent fingerprint",
    );

    let unavailable = create_agent_response_intent(
        &inbox,
        1,
        EvidenceResponseAction::ProposeRevision,
        "agent-probe",
    )
    .unwrap_err();
    ensure(
        unavailable.contains("not available from recorded evidence"),
        "passed-evidence unavailable candidate did not fail closed",
    );

    ensure(inbox.len() == before_len, "response intent creation mutated inbox");
    ensure(
        resolution == before_resolution,
        "response intent creation mutated Resolution state",
    );

    let intent_json: serde_json::Value = serde_json::from_str(&revision_intent.to_json())
        .map_err(|error| format!("response intent JSON parse failed: {error}"))?;
    ensure(
        intent_json["selection"]["selected_action"] == "propose_revision",
        "response intent lost explicit selected action",
    );
    ensure(
        intent_json["selection"]["authority"] == "explicit_caller_agent",
        "response intent selection authority drifted",
    );
    ensure(
        intent_json["execution_authorized"] == false
            && intent_json["execution_effect"] == "none",
        "response intent overclaimed execution authority",
    );
    ensure(
        intent_json["mutation"]["resolution"] == "none"
            && intent_json["mutation"]["revision"] == "none"
            && intent_json["mutation"]["diagnostic"] == "none"
            && intent_json["mutation"]["verifier"] == "none",
        "response intent overclaimed mutation",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.agent-response-intent-probe.v1",
            "status": "passed",
            "entry_count": inbox.len(),
            "selected_action": revision_intent.selected_action().as_str(),
            "selector": revision_intent.selector(),
            "evidence_kind": revision_intent.evidence_kind(),
            "evidence_outcome": revision_intent.evidence_outcome(),
            "evidence_fingerprint": revision_intent.evidence_fingerprint(),
            "response_intent_fingerprint": revision_intent.response_intent_fingerprint(),
            "preflight": serde_json::from_str::<serde_json::Value>(&revision_preflight.to_json())
                .map_err(|error| format!("preflight JSON parse failed: {error}"))?,
            "deterministic_repeat": true,
            "action_specific_identity": true,
            "unavailable_candidate_rejected": true,
            "inbox_unchanged": true,
            "resolution_unchanged": true,
            "execution_authorized": false
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
