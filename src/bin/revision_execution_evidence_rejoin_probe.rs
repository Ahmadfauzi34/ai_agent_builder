#[cfg(not(target_arch = "wasm32"))]
use burn_research::agent_response_intent::create_agent_response_intent;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_review::ResolutionReviewSession;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_revision::ResolutionRevisionChain;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::resolution_runtime_bridge::RuntimeSubjectProjection;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::response_dispatch_request::create_revision_dispatch_request;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::revision_dispatch_execution_adapter::execute_revision_dispatch_request;
#[cfg(not(target_arch = "wasm32"))]
use burn_research::revision_execution_evidence_rejoin::record_revision_dispatch_execution_receipt;
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
            "schema": "burn-research.revision-execution-evidence-rejoin-probe.v1",
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
    let mut review = ResolutionReviewSession::new("intent-execution-evidence-rejoin-probe")?;
    review.submit("agent")?;
    let approval = review.approve("customer")?;
    let resolution = review.snapshot().workflow;

    let projection = RuntimeSubjectProjection {
        schema: "burn-research.runtime-subject-projection.v1".to_string(),
        intent_id: resolution.intent_id.clone(),
        workflow_revision: resolution.revision,
        approval_id: approval.approval_id,
        subject_kind: "effective-spec".to_string(),
        subject_identity: "spec-execution-evidence-rejoin-probe".to_string(),
        effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
        effective_spec_identity: "spec-execution-evidence-rejoin-probe".to_string(),
        authorization_schema: "burn-research.authorization.v1".to_string(),
        authorization_policy_id: "policy-execution-evidence-rejoin-probe".to_string(),
        authorization_policy_revision: 1,
        authorization_is_revision: false,
        approver: "customer".to_string(),
        fields: Vec::new(),
    };

    let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection)?;
    let failed = RuntimeEvidence::bound_graph_verifier_receipt(
        &projection,
        1,
        "failed",
        "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
        false,
        "mismatch",
    )?;
    ensure(inbox.record(failed)?, "source failed evidence was not recorded");

    let mut chain = ResolutionRevisionChain::from_approved(review.snapshot())?;
    let intent = create_agent_response_intent(
        &inbox,
        0,
        EvidenceResponseAction::ProposeRevision,
        "agent",
    )?;
    let request = create_revision_dispatch_request(&inbox, &intent, &chain, "r1", None)?;
    let receipt = execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain)?;
    let before_rejoin = inbox.len();

    ensure(
        record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )?,
        "committed execution receipt was not recorded",
    );
    ensure(
        inbox.len() == before_rejoin + 1,
        "execution receipt did not add exactly one observation",
    );

    let (
        evidence_kind,
        evidence_outcome,
        evidence_source_authority,
        evidence_authority,
        evidence_transport_integrity,
    ) = {
        let evidence = inbox
            .observations()
            .last()
            .ok_or_else(|| "execution evidence missing after rejoin".to_string())?;
        (
            evidence.kind().to_string(),
            evidence.outcome().to_string(),
            evidence.source_authority().to_string(),
            evidence.evidence_authority().to_string(),
            evidence.transport_integrity().to_string(),
        )
    };
    ensure(
        evidence_kind == "revision_dispatch_execution_receipt",
        "execution evidence kind mismatch",
    );
    ensure(
        evidence_outcome == "committed",
        "execution evidence outcome mismatch",
    );
    ensure(
        evidence_source_authority == "ResolutionRevisionChain",
        "execution source authority mismatch",
    );
    ensure(
        evidence_authority == "observation_only",
        "execution evidence authority escalated",
    );
    ensure(
        evidence_transport_integrity == "native_typed_correlated",
        "execution evidence transport classification mismatch",
    );

    let interpretation = interpret_recorded_evidence(&inbox, inbox.len() - 1)?;
    ensure(
        interpretation.available(EvidenceResponseAction::Ignore),
        "committed execution evidence must remain ignorable",
    );
    ensure(
        !interpretation.available(EvidenceResponseAction::Reverify),
        "committed execution evidence unexpectedly exposed reverify",
    );
    ensure(
        !interpretation.available(EvidenceResponseAction::RequestInformation),
        "committed execution evidence unexpectedly requested information",
    );
    ensure(
        !interpretation.available(EvidenceResponseAction::ProposeRevision),
        "committed execution evidence recursively proposed another revision",
    );

    ensure(
        !record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )?,
        "duplicate execution receipt was not idempotent",
    );

    let mut tampered = receipt.clone();
    tampered.receipt_fingerprint = "fnv1a64:0000000000000000".to_string();
    let before_tamper = inbox.len();
    ensure(
        record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &tampered,
        )
        .is_err(),
        "tampered execution receipt was accepted",
    );
    ensure(
        inbox.len() == before_tamper,
        "tampered execution receipt mutated inbox",
    );

    println!(
        "{}",
        serde_json::json!({
            "schema": "burn-research.revision-execution-evidence-rejoin-probe.v1",
            "status": "passed",
            "receipt": {
                "revision_id": receipt.revision_id,
                "request_fingerprint": receipt.request_fingerprint,
                "receipt_fingerprint": receipt.receipt_fingerprint,
            },
            "evidence": {
                "kind": evidence_kind,
                "outcome": evidence_outcome,
                "source_authority": evidence_source_authority,
                "evidence_authority": evidence_authority,
                "transport_integrity": evidence_transport_integrity,
            },
            "interpretation": {
                "ignore": interpretation.available(EvidenceResponseAction::Ignore),
                "reverify": interpretation.available(EvidenceResponseAction::Reverify),
                "request_information": interpretation.available(EvidenceResponseAction::RequestInformation),
                "propose_revision": interpretation.available(EvidenceResponseAction::ProposeRevision),
            },
            "duplicate_idempotent": true,
            "tamper_rejected": true,
            "inbox_entry_count": inbox.len(),
        })
    );

    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}
