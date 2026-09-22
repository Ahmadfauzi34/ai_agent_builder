#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_review::ResolutionReviewSession;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_revision::ResolutionRevisionChain;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_dispatch_request::{
    create_ignore_dispatch_request, create_information_dispatch_request,
    create_revision_dispatch_request, preflight_response_dispatch_request,
    response_dispatch_request_bindability,
};
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
            "schema": "burn-research.response-dispatch-request-probe.v1",
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
    let mut review = ResolutionReviewSession::new("intent-dispatch-request-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let review_snapshot = review.snapshot();
    let resolution = review_snapshot.workflow.clone();
    let before_resolution = resolution.clone();

    let subject = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id.clone(),
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-dispatch-request-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-dispatch-request-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-dispatch-request-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "customer".to_string(),
        fields: Vec::new(),
    };

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &subject)?;
    let failed = RuntimeEvidence::bound_graph_verifier_receipt(
        &subject,
        1,
        "failed-graph",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
        false,
        "candidate mismatch",
    )?;
    ensure(inbox.record(failed)?, "failed graph evidence was not recorded");
    let before_inbox_len = inbox.len();

    let reverify_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Reverify,
        "agent-request-probe",
    )?;
    let reverify_bindability =
        response_dispatch_request_bindability(&inbox, &reverify_intent);
    ensure(
        !reverify_bindability.bindable
            && reverify_bindability.status == "runtime_handle_binding_deferred",
        "reverify must remain deferred until real runtime handles are bound",
    );
    ensure(
        reverify_bindability
            .unresolved_required
            .iter()
            .any(|name| name == "compiled_graph"),
        "reverify unresolved runtime handle requirements missing",
    );

    let info_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::RequestInformation,
        "agent-request-probe",
    )?;
    let info_request = create_information_dispatch_request(
        &inbox,
        &info_intent,
        "Provide the missing tensor rank",
        Some("reviewer".to_string()),
    )?;
    ensure(
        preflight_response_dispatch_request(&inbox, &info_intent, &info_request).ready,
        "information dispatch request preflight was not ready",
    );
    ensure(
        info_request.executor_contract() == "external_resolution_review"
            && info_request.operation() == "request_information",
        "information dispatch target drift",
    );

    let info_request_same = create_information_dispatch_request(
        &inbox,
        &info_intent,
        "Provide the missing tensor rank",
        Some("reviewer".to_string()),
    )?;
    let info_request_different = create_information_dispatch_request(
        &inbox,
        &info_intent,
        "Provide the missing dtype",
        Some("reviewer".to_string()),
    )?;
    ensure(
        info_request.request_fingerprint() == info_request_same.request_fingerprint(),
        "identical information request fingerprint drift",
    );
    ensure(
        info_request.request_fingerprint() != info_request_different.request_fingerprint(),
        "different information requests share request fingerprint",
    );

    let revision_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent-request-probe",
    )?;
    let chain = ResolutionRevisionChain::from_approved(review_snapshot)?;
    let before_chain = chain.snapshot();
    let revision_request = create_revision_dispatch_request(
        &inbox,
        &revision_intent,
        &chain,
        "fix-shape",
        None,
    )?;
    ensure(
        preflight_response_dispatch_request(&inbox, &revision_intent, &revision_request).ready,
        "revision dispatch request preflight was not ready",
    );
    ensure(
        revision_request.executor_contract() == "ResolutionRevisionChain"
            && revision_request.operation() == "open_revision",
        "revision dispatch target drift",
    );
    ensure(
        chain.snapshot() == before_chain,
        "creating revision dispatch request mutated revision chain",
    );

    let ignore_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::Ignore,
        "agent-request-probe",
    )?;
    let ignore_request = create_ignore_dispatch_request(&inbox, &ignore_intent)?;
    ensure(
        preflight_response_dispatch_request(&inbox, &ignore_intent, &ignore_request).ready,
        "ignore dispatch request preflight was not ready",
    );
    ensure(
        ignore_request.executor_contract() == "none"
            && ignore_request.operation() == "no_op",
        "ignore request must remain explicit no-op",
    );

    let mut changed_inbox = ResolutionEvidenceInbox::new(&resolution, &subject)?;
    let changed = RuntimeEvidence::bound_graph_verifier_receipt(
        &subject,
        2,
        "changed",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"changed\"}",
        true,
        "match",
    )?;
    ensure(changed_inbox.record(changed)?, "changed evidence was not recorded");
    let stale_preflight =
        preflight_response_dispatch_request(&changed_inbox, &info_intent, &info_request);
    ensure(
        !stale_preflight.ready,
        "dispatch request remained ready against changed evidence",
    );

    ensure(
        inbox.len() == before_inbox_len,
        "dispatch request creation mutated evidence inbox",
    );
    ensure(
        resolution == before_resolution,
        "dispatch request creation mutated Resolution snapshot",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-dispatch-request-probe.v1",
            "status": "passed",
            "reverify": {
                "bindable": reverify_bindability.bindable,
                "status": reverify_bindability.status,
                "unresolved_required": reverify_bindability.unresolved_required,
            },
            "request_information": {
                "executor_contract": info_request.executor_contract(),
                "operation": info_request.operation(),
                "request_fingerprint": info_request.request_fingerprint(),
                "preflight_ready": true,
            },
            "propose_revision": {
                "executor_contract": revision_request.executor_contract(),
                "operation": revision_request.operation(),
                "request_fingerprint": revision_request.request_fingerprint(),
                "revision_chain_unchanged": chain.snapshot() == before_chain,
            },
            "ignore": {
                "executor_contract": ignore_request.executor_contract(),
                "operation": ignore_request.operation(),
            },
            "stale_request_closed": !stale_preflight.ready,
            "inbox_unchanged": inbox.len() == before_inbox_len,
            "resolution_unchanged": resolution == before_resolution,
            "execution_authorized": false,
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
