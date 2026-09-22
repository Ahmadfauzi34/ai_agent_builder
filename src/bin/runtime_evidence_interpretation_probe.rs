#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution::ResolutionWorkflow;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::{
    interpret_recorded_evidence, EvidenceResponseAction,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{
    ResolutionEvidenceInbox, RuntimeEvidence,
};

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.runtime-evidence-interpretation-probe.v1",
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
        approval_id: "approval-interpretation-probe".to_string(),
        subject_kind: "effective-spec".to_string(),
        subject_identity: subject_identity.to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: subject_identity.to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-interpretation-probe".to_string(),
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
    let mut workflow = ResolutionWorkflow::new("intent-interpretation-probe")?;
    workflow.submit()?;
    workflow.finalize_resolution()?;
    let resolution = workflow.snapshot();
    let before_resolution = resolution.clone();

    let projection = projection(
        &resolution.intent_id,
        resolution.revision,
        "spec-interpretation-probe",
    );
    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;

    let fault = RuntimeEvidence::bound_agent_fault(
        &projection,
        "E_RUNTIME_CONTEXT",
        "semantic_precondition",
        "workspaceInitUnary",
        "layout.compatible",
        true,
        "runtime context requires attention",
    )?;
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

    ensure(inbox.record(fault)?, "fault evidence was not recorded");
    ensure(inbox.record(failed)?, "failed verifier evidence was not recorded");
    ensure(inbox.record(passed)?, "passed verifier evidence was not recorded");
    ensure(inbox.len() == 3, "unexpected inbox length");

    let before_len = inbox.len();

    let fault_view = interpret_recorded_evidence(&inbox, 0)?;
    let failed_view = interpret_recorded_evidence(&inbox, 1)?;
    let passed_view = interpret_recorded_evidence(&inbox, 2)?;

    ensure(
        fault_view.available(EvidenceResponseAction::Ignore)
            && !fault_view.available(EvidenceResponseAction::Reverify)
            && fault_view.available(EvidenceResponseAction::RequestInformation)
            && fault_view.available(EvidenceResponseAction::ProposeRevision),
        "fault candidate matrix drifted",
    );
    ensure(
        failed_view.available(EvidenceResponseAction::Ignore)
            && failed_view.available(EvidenceResponseAction::Reverify)
            && failed_view.available(EvidenceResponseAction::RequestInformation)
            && failed_view.available(EvidenceResponseAction::ProposeRevision),
        "failed verifier candidate matrix drifted",
    );
    ensure(
        passed_view.available(EvidenceResponseAction::Ignore)
            && passed_view.available(EvidenceResponseAction::Reverify)
            && !passed_view.available(EvidenceResponseAction::RequestInformation)
            && !passed_view.available(EvidenceResponseAction::ProposeRevision),
        "passed verifier candidate matrix drifted",
    );

    let failed_json: serde_json::Value = serde_json::from_str(&failed_view.to_json())
        .map_err(|error| format!("failed interpretation JSON parse failed: {error}"))?;
    ensure(
        failed_json["selection"]["selected_action"].is_null()
            && failed_json["selection"]["default_action"].is_null()
            && failed_json["selection"]["ranking"] == "none",
        "interpreter selected or ranked an action",
    );
    ensure(
        failed_json["mutation"]["interpreter"] == "none"
            && failed_json["mutation"]["inbox"] == "none"
            && failed_json["mutation"]["resolution"] == "none"
            && failed_json["mutation"]["revision"] == "none"
            && failed_json["mutation"]["diagnostic"] == "none"
            && failed_json["mutation"]["verifier_execution"] == "none",
        "interpreter overclaimed a mutation",
    );

    ensure(inbox.len() == before_len, "interpretation mutated inbox");
    ensure(
        resolution == before_resolution,
        "interpretation mutated Resolution snapshot",
    );
    ensure(
        interpret_recorded_evidence(&inbox, 3).is_err(),
        "interpreter read an unrecorded entry",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.runtime-evidence-interpretation-probe.v1",
            "status": "passed",
            "entry_count": inbox.len(),
            "fault_matrix": {
                "ignore": true,
                "reverify": false,
                "request_information": true,
                "propose_revision": true
            },
            "failed_verifier_matrix": {
                "ignore": true,
                "reverify": true,
                "request_information": true,
                "propose_revision": true
            },
            "passed_verifier_matrix": {
                "ignore": true,
                "reverify": true,
                "request_information": false,
                "propose_revision": false
            },
            "selected_action": null,
            "default_action": null,
            "ranking": "none",
            "inbox_unchanged": true,
            "resolution_unchanged": true,
            "unrecorded_entry_rejected": true
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
