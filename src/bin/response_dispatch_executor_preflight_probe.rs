#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_review::ResolutionReviewSession;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_revision::ResolutionRevisionChain;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_dispatch_executor_preflight::{
    preflight_revision_dispatch_executor, response_dispatch_executor_preflight,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_dispatch_request::{
    create_ignore_dispatch_request, create_information_dispatch_request,
    create_revision_dispatch_request,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::EvidenceResponseAction;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-dispatch-executor-preflight-probe.v1",
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
    let mut review = ResolutionReviewSession::new("intent-executor-preflight-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let resolution = review.snapshot().workflow;

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id,
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-executor-preflight-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-executor-preflight-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-executor-preflight-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "customer".to_string(),
        fields: Vec::new(),
    };

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        1,
        "failed",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
        false,
        "mismatch",
    )?;
    ensure(inbox.record(evidence)?, "evidence not recorded");
    let inbox_len_before = inbox.len();

    let ignore_intent =
        create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")?;
    let ignore_request = create_ignore_dispatch_request(&inbox, &ignore_intent)?;
    let ignore =
        response_dispatch_executor_preflight(&inbox, &ignore_intent, &ignore_request);
    ensure(ignore.request_ready, "ignore request not ready");
    ensure(ignore.executor_ready, "ignore no-op executor should be ready");
    ensure(ignore.status == "ready_noop", "ignore status drift");
    ensure(!ignore.execution_authorized, "ignore preflight authorized execution");

    let info_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::RequestInformation,
        "agent",
    )?;
    let info_request = create_information_dispatch_request(
        &inbox,
        &info_intent,
        "provide authoritative source",
        Some("agent".to_string()),
    )?;
    let info = response_dispatch_executor_preflight(&inbox, &info_intent, &info_request);
    ensure(info.request_ready, "information request not ready");
    ensure(!info.executor_ready, "external authority should remain deferred");
    ensure(
        info.status == "deferred:external_resolution_review_authority",
        "external review status drift",
    );

    let chain = ResolutionRevisionChain::from_approved(review.snapshot())?;
    let revision_intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent",
    )?;
    let revision_request =
        create_revision_dispatch_request(&inbox, &revision_intent, &chain, "r1", None)?;
    let generic =
        response_dispatch_executor_preflight(&inbox, &revision_intent, &revision_request);
    ensure(generic.request_ready, "revision request not ready");
    ensure(
        !generic.executor_ready,
        "generic revision preflight must require chain context",
    );

    let chain_before = chain.snapshot();
    let revision =
        preflight_revision_dispatch_executor(&inbox, &revision_intent, &revision_request, &chain);
    ensure(revision.request_ready, "typed revision request not ready");
    ensure(revision.executor_ready, "revision clone preflight did not accept");
    ensure(
        revision.status == "ready_revision_clone_preflight",
        "revision preflight status drift",
    );
    ensure(
        revision.simulated_effect.starts_with("would_open_revision:"),
        "revision simulated effect missing",
    );
    ensure(chain.snapshot() == chain_before, "caller-owned revision chain mutated");
    ensure(
        !revision.execution_authorized,
        "revision preflight authorized execution",
    );

    ensure(inbox.len() == inbox_len_before, "executor preflight mutated inbox");

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.response-dispatch-executor-preflight-probe.v1",
            "status": "passed",
            "ignore": {
                "request_ready": ignore.request_ready,
                "executor_ready": ignore.executor_ready,
                "status": ignore.status,
            },
            "request_information": {
                "request_ready": info.request_ready,
                "executor_ready": info.executor_ready,
                "status": info.status,
            },
            "propose_revision": {
                "request_ready": revision.request_ready,
                "executor_ready": revision.executor_ready,
                "status": revision.status,
                "simulated_effect": revision.simulated_effect,
                "chain_unchanged": chain.snapshot() == chain_before,
            },
            "inbox_unchanged": inbox.len() == inbox_len_before,
            "execution_authorized": false,
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
