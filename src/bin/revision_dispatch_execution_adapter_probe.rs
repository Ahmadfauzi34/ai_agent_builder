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
    create_ignore_dispatch_request, create_revision_dispatch_request,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_research::revision_dispatch_execution_adapter::execute_revision_dispatch_request;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_evidence_interpretation::EvidenceResponseAction;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

#[cfg(not(target_arch = "wasm32"))]
fn fail(message: impl AsRef<str>) -> ! {
    eprintln!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.revision-dispatch-execution-adapter-probe.v1",
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
    let mut review = ResolutionReviewSession::new("intent-revision-execution-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let resolution = review.snapshot().workflow;

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id,
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-revision-execution-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-revision-execution-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-revision-execution-probe".to_string(),
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
    ensure(inbox.record(evidence)?, "evidence was not recorded");
    let inbox_len_before = inbox.len();

    let mut chain = ResolutionRevisionChain::from_approved(review.snapshot())?;
    let intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent",
    )?;
    let request = create_revision_dispatch_request(&inbox, &intent, &chain, "r1", None)?;

    let before = chain.snapshot();
    let receipt = execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain)?;
    let after = chain.snapshot();

    ensure(before.revisions.is_empty(), "chain was not empty before execution");
    ensure(after.revisions.len() == 1, "execution did not create exactly one revision");
    ensure(
        receipt.request_fingerprint == request.request_fingerprint(),
        "receipt/request fingerprint correlation drift",
    );
    ensure(
        receipt.before_revision_count == 0 && receipt.after_revision_count == 1,
        "revision count receipt drift",
    );
    ensure(
        receipt.execution_trigger == "explicit_caller_invocation",
        "execution trigger drift",
    );
    ensure(receipt.authorization_claim == "none", "authorization claim drift");
    ensure(receipt.mutation == "committed", "mutation state drift");
    ensure(
        after.revision(&receipt.revision_id).is_some(),
        "receipt revision missing from committed chain",
    );

    let once = chain.snapshot();
    let replay = execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain);
    ensure(replay.is_err(), "same request replay unexpectedly executed");
    ensure(chain.snapshot() == once, "replay mutated chain");

    let ignore_intent =
        create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")?;
    let ignore_request = create_ignore_dispatch_request(&inbox, &ignore_intent)?;
    let before_wrong_route = chain.snapshot();
    let wrong_route =
        execute_revision_dispatch_request(&inbox, &ignore_intent, &ignore_request, &mut chain);
    ensure(
        wrong_route.is_err(),
        "non-revision request entered revision mutation adapter",
    );
    ensure(
        chain.snapshot() == before_wrong_route,
        "wrong-route request mutated chain",
    );

    ensure(inbox.len() == inbox_len_before, "execution adapter mutated evidence inbox");

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.revision-dispatch-execution-adapter-probe.v1",
            "status": "passed",
            "receipt": serde_json::from_str::<serde_json::Value>(&receipt.to_json())
                .map_err(|error| error.to_string())?,
            "single_commit": after.revisions.len() == 1,
            "replay_rejected": replay.is_err(),
            "wrong_route_rejected": wrong_route.is_err(),
            "inbox_unchanged": inbox.len() == inbox_len_before,
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
