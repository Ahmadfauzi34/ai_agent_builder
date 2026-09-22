use crate::agent_response_intent::AgentResponseIntent;
use crate::resolution_revision::ResolutionRevisionChain;
use crate::response_dispatch_request::{
    preflight_response_dispatch_request, ResponseDispatchRequest, ResponseDispatchRequestPayload,
};
use crate::revision_dispatch_execution_adapter::RevisionDispatchExecutionReceipt;
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

const REVISION_EXECUTION_EVIDENCE_REJOIN_V1: &str =
    include_str!("../docs/revision-execution-evidence-rejoin.v1.json");

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn expected_receipt_fingerprint(receipt: &RevisionDispatchExecutionReceipt) -> String {
    let canonical = format!(
        concat!(
            "v1|request={}|intent={}|dispatch={}|lineage={}|revision={}|",
            "before={}|after={}|"
        ),
        receipt.request_fingerprint,
        receipt.response_intent_fingerprint,
        receipt.dispatch_fingerprint,
        receipt.lineage_id,
        receipt.revision_id,
        receipt.before_revision_count,
        receipt.after_revision_count,
    );
    fnv1a64(canonical.bytes())
}

pub fn revision_execution_evidence_rejoin_capabilities() -> &'static str {
    REVISION_EXECUTION_EVIDENCE_REJOIN_V1
}

pub fn record_revision_dispatch_execution_receipt(
    inbox: &mut ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
    chain: &ResolutionRevisionChain,
    receipt: &RevisionDispatchExecutionReceipt,
) -> Result<bool, String> {
    const CONTEXT: &str = "RevisionExecutionEvidenceRejoin";

    if intent.selected_action() != EvidenceResponseAction::ProposeRevision
        || request.selected_action() != EvidenceResponseAction::ProposeRevision
    {
        return Err(format!(
            "{CONTEXT}: only propose_revision execution receipts can rejoin through this adapter"
        ));
    }

    let request_preflight = preflight_response_dispatch_request(inbox, intent, request);
    if !request_preflight.ready {
        return Err(format!(
            "{CONTEXT}: originating dispatch request is no longer current: {}",
            request_preflight.status
        ));
    }

    let ResponseDispatchRequestPayload::ProposeRevision {
        lineage_id,
        revision_key,
        parent_revision_id,
    } = request.payload()
    else {
        return Err(format!("{CONTEXT}: request payload is not propose_revision"));
    };

    if receipt.request_fingerprint != request.request_fingerprint() {
        return Err(format!("{CONTEXT}: receipt/request fingerprint mismatch"));
    }
    if receipt.response_intent_fingerprint != intent.response_intent_fingerprint()
        || receipt.response_intent_fingerprint != request.response_intent_fingerprint()
    {
        return Err(format!("{CONTEXT}: response-intent fingerprint mismatch"));
    }
    if receipt.dispatch_fingerprint != request.dispatch_fingerprint() {
        return Err(format!("{CONTEXT}: dispatch fingerprint mismatch"));
    }
    if receipt.lineage_id.as_str() != lineage_id.as_str()
        || receipt.lineage_id.as_str() != chain.lineage_id()
    {
        return Err(format!("{CONTEXT}: revision lineage mismatch"));
    }
    if receipt.revision_key.as_str() != revision_key.as_str()
        || receipt.parent_revision_id.as_deref() != parent_revision_id.as_deref()
    {
        return Err(format!("{CONTEXT}: committed revision payload mismatch"));
    }
    if receipt.execution_trigger != "explicit_caller_invocation"
        || receipt.authorization_claim != "none"
        || receipt.mutation != "committed"
    {
        return Err(format!("{CONTEXT}: execution authority tuple mismatch"));
    }
    if receipt.after_revision_count != receipt.before_revision_count.saturating_add(1) {
        return Err(format!(
            "{CONTEXT}: receipt revision-count delta must be exactly one"
        ));
    }

    let expected_fingerprint = expected_receipt_fingerprint(receipt);
    if receipt.receipt_fingerprint != expected_fingerprint {
        return Err(format!("{CONTEXT}: receipt fingerprint mismatch"));
    }

    let chain_snapshot = chain.snapshot();
    let committed = chain_snapshot
        .revision(&receipt.revision_id)
        .ok_or_else(|| format!("{CONTEXT}: committed revision is absent from current chain"))?;
    if committed.revision_key.as_str() != receipt.revision_key.as_str()
        || committed.parent_revision_id.as_deref() != receipt.parent_revision_id.as_deref()
    {
        return Err(format!("{CONTEXT}: current chain revision does not match receipt"));
    }
    if chain_snapshot.revisions.len() < receipt.after_revision_count {
        return Err(format!(
            "{CONTEXT}: current chain is older than receipt post-state"
        ));
    }

    let evidence = RuntimeEvidence::bound_revision_dispatch_execution_receipt_for_inbox(
        inbox,
        receipt.receipt_fingerprint.clone(),
        receipt.request_fingerprint.clone(),
        receipt.response_intent_fingerprint.clone(),
        receipt.dispatch_fingerprint.clone(),
        receipt.lineage_id.clone(),
        receipt.revision_id.clone(),
        receipt.revision_key.clone(),
        receipt.parent_revision_id.clone(),
        receipt.before_revision_count,
        receipt.after_revision_count,
    )?;

    inbox.record(evidence)
}

#[cfg(test)]
mod tests {
    use super::{
        record_revision_dispatch_execution_receipt,
        revision_execution_evidence_rejoin_capabilities,
    };
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_revision::ResolutionRevisionChain;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::response_dispatch_request::create_revision_dispatch_request;
    use crate::revision_dispatch_execution_adapter::execute_revision_dispatch_request;
    use crate::runtime_evidence_interpretation::{
        interpret_recorded_evidence, EvidenceResponseAction,
    };
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn fixture() -> (
        ResolutionEvidenceInbox,
        crate::agent_response_intent::AgentResponseIntent,
        crate::response_dispatch_request::ResponseDispatchRequest,
        ResolutionRevisionChain,
        crate::revision_dispatch_execution_adapter::RevisionDispatchExecutionReceipt,
    ) {
        let mut review = ResolutionReviewSession::new("intent-execution-rejoin").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-execution-rejoin".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-execution-rejoin".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-execution-rejoin".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "customer".to_string(),
            fields: Vec::new(),
        };

        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let failed = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "failed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
            false,
            "mismatch",
        )
        .unwrap();
        inbox.record(failed).unwrap();

        let mut chain = ResolutionRevisionChain::from_approved(review.snapshot()).unwrap();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent",
        )
        .unwrap();
        let request =
            create_revision_dispatch_request(&inbox, &intent, &chain, "r1", None).unwrap();
        let receipt =
            execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain).unwrap();

        (inbox, intent, request, chain, receipt)
    }

    #[test]
    fn committed_revision_receipt_rejoins_as_observation_only_evidence() {
        let (mut inbox, intent, request, chain, receipt) = fixture();
        let before = inbox.len();

        assert!(record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )
        .unwrap());
        assert_eq!(inbox.len(), before + 1);

        let evidence = inbox.observations().last().unwrap();
        assert_eq!(evidence.kind(), "revision_dispatch_execution_receipt");
        assert_eq!(evidence.outcome(), "committed");
        assert_eq!(evidence.source_authority(), "ResolutionRevisionChain");
        assert_eq!(evidence.evidence_authority(), "observation_only");
        assert_eq!(evidence.transport_integrity(), "native_typed_correlated");

        let view = interpret_recorded_evidence(&inbox, inbox.len() - 1).unwrap();
        assert!(view.available(EvidenceResponseAction::Ignore));
        assert!(!view.available(EvidenceResponseAction::Reverify));
        assert!(!view.available(EvidenceResponseAction::RequestInformation));
        assert!(!view.available(EvidenceResponseAction::ProposeRevision));

        let inbox_json = inbox.to_json();
        assert!(inbox_json.contains("\"revision_dispatch_committed\":1"));
        assert!(inbox_json.contains("\"revision_created\":false"));
    }

    #[test]
    fn duplicate_execution_receipt_recording_is_idempotent() {
        let (mut inbox, intent, request, chain, receipt) = fixture();
        assert!(record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )
        .unwrap());
        assert!(!record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )
        .unwrap());
    }

    #[test]
    fn tampered_receipt_fails_before_inbox_mutation() {
        let (mut inbox, intent, request, chain, mut receipt) = fixture();
        let before = inbox.len();
        receipt.receipt_fingerprint = "fnv1a64:0000000000000000".to_string();

        let error = record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )
        .unwrap_err();
        assert!(error.contains("receipt fingerprint mismatch"));
        assert_eq!(inbox.len(), before);
    }

    #[test]
    fn receipt_cannot_be_rebound_to_different_request_lineage() {
        let (mut inbox, intent, request, chain, mut receipt) = fixture();
        receipt.lineage_id = "foreign-lineage".to_string();
        let before = inbox.len();

        assert!(record_revision_dispatch_execution_receipt(
            &mut inbox,
            &intent,
            &request,
            &chain,
            &receipt,
        )
        .is_err());
        assert_eq!(inbox.len(), before);
    }

    #[test]
    fn capability_contract_keeps_rejoin_observation_only() {
        let contract: serde_json::Value =
            serde_json::from_str(revision_execution_evidence_rejoin_capabilities()).unwrap();
        assert_eq!(contract["evidence_authority"], "observation_only");
        assert_eq!(contract["automatic_action_selection"], false);
        assert_eq!(contract["source_authority"], "ResolutionRevisionChain");
    }
}
