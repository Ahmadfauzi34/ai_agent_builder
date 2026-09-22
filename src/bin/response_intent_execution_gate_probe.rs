#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution::ResolutionWorkflow;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_intent_execution_gate::response_intent_execution_gate;
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
            "schema": "burn-research.response-intent-execution-gate-probe.v1",
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
        approval_id: "approval-execution-gate-probe".to_string(),
        subject_kind: "effective-spec".to_string(),
        subject_identity: subject_identity.to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: subject_identity.to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-execution-gate-probe".to_string(),
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
    let mut workflow = ResolutionWorkflow::new("intent-execution-gate-probe")?;
    workflow.submit()?;
    workflow.finalize_resolution()?;
    let resolution = workflow.snapshot();
    let before_resolution = resolution.clone();

    let projection = projection(
        &resolution.intent_id,
        resolution.revision,
        "spec-execution-gate-probe",
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
    ensure(inbox.record(failed)?, "failed graph evidence was not recorded");
    let before_len = inbox.len();

    let revision_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent-gate-probe",
    )?;
    let revision_gate = response_intent_execution_gate(&inbox, &revision_intent);
    ensure(revision_gate.dispatchable, "revision gate was not dispatchable");
    ensure(
        revision_gate.gate_status == "dispatchable_nonexecuting",
        "unexpected revision gate status",
    );
    ensure(
        !revision_gate.execution_authorized,
        "execution gate must not authorize revision execution",
    );
    let revision_route = revision_gate
        .route
        .as_ref()
        .ok_or_else(|| "revision gate route missing".to_string())?;
    ensure(
        revision_route.authority == "ResolutionRevisionChain"
            && revision_route.operation == "open_revision"
            && revision_route.route_mode == "authority_handoff"
            && revision_route.explicit_executor_required,
        "revision route drifted",
    );

    let revision_repeat = response_intent_execution_gate(&inbox, &revision_intent);
    ensure(
        revision_gate.dispatch_fingerprint == revision_repeat.dispatch_fingerprint,
        "identical dispatch projection was not deterministic",
    );

    let reverify_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent-gate-probe",
    )?;
    let reverify_gate = response_intent_execution_gate(&inbox, &reverify_intent);
    let reverify_route = reverify_gate
        .route
        .as_ref()
        .ok_or_else(|| "reverify route missing".to_string())?;
    ensure(
        reverify_gate.dispatchable
            && reverify_route.authority == "runtime_verifier"
            && reverify_route.operation == "CompiledGraph.verifyFlat"
            && reverify_route.explicit_executor_required,
        "reverify route drifted",
    );
    ensure(
        revision_gate.dispatch_fingerprint != reverify_gate.dispatch_fingerprint,
        "different response actions produced the same dispatch fingerprint",
    );

    let ignore_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Ignore,
        "agent-gate-probe",
    )?;
    let ignore_gate = response_intent_execution_gate(&inbox, &ignore_intent);
    let ignore_route = ignore_gate
        .route
        .as_ref()
        .ok_or_else(|| "ignore route missing".to_string())?;
    ensure(
        ignore_gate.dispatchable
            && ignore_route.authority == "none"
            && ignore_route.operation == "no_op"
            && ignore_route.route_mode == "terminal_noop"
            && !ignore_route.explicit_executor_required,
        "ignore no-op route drifted",
    );

    let request_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::RequestInformation,
        "agent-gate-probe",
    )?;
    let request_gate = response_intent_execution_gate(&inbox, &request_intent);
    let request_route = request_gate
        .route
        .as_ref()
        .ok_or_else(|| "request-information route missing".to_string())?;
    ensure(
        request_gate.dispatchable
            && request_route.authority == "external_resolution_review"
            && request_route.operation == "request_information"
            && request_route.explicit_executor_required,
        "request-information route drifted",
    );

    let mut different_inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let different = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        2,
        "passed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"passed\"}",
        true,
        "candidate match",
    )?;
    ensure(
        different_inbox.record(different)?,
        "different evidence was not recorded",
    );
    let closed_gate = response_intent_execution_gate(&different_inbox, &revision_intent);
    ensure(
        !closed_gate.dispatchable
            && closed_gate.route.is_none()
            && closed_gate.dispatch_fingerprint.is_none()
            && closed_gate
                .gate_status
                .contains("evidence_fingerprint_mismatch")
            && !closed_gate.execution_authorized,
        "mismatched evidence did not close gate",
    );

    ensure(inbox.len() == before_len, "execution gate mutated inbox");
    ensure(
        resolution == before_resolution,
        "execution gate mutated Resolution state",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-intent-execution-gate-probe.v1",
            "status": "passed",
            "revision_route": {
                "dispatchable": revision_gate.dispatchable,
                "authority": revision_route.authority,
                "operation": revision_route.operation,
                "dispatch_fingerprint": revision_gate.dispatch_fingerprint,
                "execution_authorized": revision_gate.execution_authorized
            },
            "reverify_route": {
                "dispatchable": reverify_gate.dispatchable,
                "authority": reverify_route.authority,
                "operation": reverify_route.operation,
                "execution_authorized": reverify_gate.execution_authorized
            },
            "request_information_route": {
                "dispatchable": request_gate.dispatchable,
                "authority": request_route.authority,
                "operation": request_route.operation,
                "execution_authorized": request_gate.execution_authorized
            },
            "ignore_route": {
                "dispatchable": ignore_gate.dispatchable,
                "authority": ignore_route.authority,
                "operation": ignore_route.operation,
                "route_mode": ignore_route.route_mode,
                "execution_authorized": ignore_gate.execution_authorized
            },
            "deterministic_repeat": true,
            "action_specific_dispatch_identity": true,
            "mismatched_evidence_gate_closed": true,
            "inbox_unchanged": true,
            "resolution_unchanged": true
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
