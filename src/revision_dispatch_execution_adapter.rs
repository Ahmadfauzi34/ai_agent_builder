use crate::agent_response_intent::AgentResponseIntent;
use crate::resolution_revision::ResolutionRevisionChain;
use crate::response_dispatch_executor_preflight::preflight_revision_dispatch_executor;
use crate::response_dispatch_request::{
    ResponseDispatchRequest, ResponseDispatchRequestPayload,
};
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const REVISION_DISPATCH_EXECUTION_ADAPTER_V1: &str =
    include_str!("../docs/revision-dispatch-execution-adapter.v1.json");

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn optional_json(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevisionDispatchExecutionReceipt {
    pub receipt_fingerprint: String,
    pub request_fingerprint: String,
    pub response_intent_fingerprint: String,
    pub dispatch_fingerprint: String,
    pub lineage_id: String,
    pub revision_id: String,
    pub revision_key: String,
    pub parent_revision_id: Option<String>,
    pub before_revision_count: usize,
    pub after_revision_count: usize,
    pub execution_trigger: String,
    pub authorization_claim: String,
    pub mutation: String,
}

impl RevisionDispatchExecutionReceipt {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.revision-dispatch-execution-receipt.v1\",",
                "\"role\":\"committed_revision_dispatch_execution_receipt\",",
                "\"authority\":\"ResolutionRevisionChain\",",
                "\"operation\":\"open_revision\",",
                "\"receipt_fingerprint\":\"{}\",",
                "\"request_fingerprint\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"dispatch_fingerprint\":\"{}\",",
                "\"lineage_id\":\"{}\",",
                "\"revision_id\":\"{}\",",
                "\"revision_key\":\"{}\",",
                "\"parent_revision_id\":{},",
                "\"before_revision_count\":{},",
                "\"after_revision_count\":{},",
                "\"execution_trigger\":\"{}\",",
                "\"authorization_claim\":\"{}\",",
                "\"mutation\":\"{}\",",
                "\"execution_performed\":true",
                "}}"
            ),
            json_escape(&self.receipt_fingerprint),
            json_escape(&self.request_fingerprint),
            json_escape(&self.response_intent_fingerprint),
            json_escape(&self.dispatch_fingerprint),
            json_escape(&self.lineage_id),
            json_escape(&self.revision_id),
            json_escape(&self.revision_key),
            optional_json(self.parent_revision_id.as_deref()),
            self.before_revision_count,
            self.after_revision_count,
            json_escape(&self.execution_trigger),
            json_escape(&self.authorization_claim),
            json_escape(&self.mutation),
        )
    }
}

pub fn revision_dispatch_execution_adapter_capabilities() -> &'static str {
    REVISION_DISPATCH_EXECUTION_ADAPTER_V1
}

fn receipt_fingerprint(
    request: &ResponseDispatchRequest,
    lineage_id: &str,
    revision_id: &str,
    before_revision_count: usize,
    after_revision_count: usize,
) -> String {
    let canonical = format!(
        concat!(
            "v1|request={}|intent={}|dispatch={}|lineage={}|revision={}|",
            "before={}|after={}|"
        ),
        request.request_fingerprint(),
        request.response_intent_fingerprint(),
        request.dispatch_fingerprint(),
        lineage_id,
        revision_id,
        before_revision_count,
        after_revision_count,
    );
    fnv1a64(canonical.bytes())
}

pub fn execute_revision_dispatch_request(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
    chain: &mut ResolutionRevisionChain,
) -> Result<RevisionDispatchExecutionReceipt, String> {
    if request.selected_action() != EvidenceResponseAction::ProposeRevision {
        return Err(
            "RevisionDispatchExecutionAdapter: only propose_revision requests are supported"
                .to_string(),
        );
    }

    let preflight = preflight_revision_dispatch_executor(inbox, intent, request, chain);
    if !preflight.request_ready || !preflight.executor_ready {
        return Err(format!(
            "RevisionDispatchExecutionAdapter: executor preflight not ready: {}",
            preflight.status
        ));
    }

    let ResponseDispatchRequestPayload::ProposeRevision {
        lineage_id,
        revision_key,
        parent_revision_id,
    } = request.payload()
    else {
        return Err(
            "RevisionDispatchExecutionAdapter: revision route payload mismatch".to_string(),
        );
    };

    let expected_revision_id = preflight
        .simulated_effect
        .strip_prefix("would_open_revision:")
        .ok_or_else(|| {
            "RevisionDispatchExecutionAdapter: preflight simulated effect missing revision id"
                .to_string()
        })?
        .to_string();

    let rollback = chain.clone();
    let before = chain.snapshot();
    let before_revision_count = before.revisions.len();

    let revision_id = match chain.open_revision(parent_revision_id.as_deref(), revision_key.clone()) {
        Ok(revision_id) => revision_id,
        Err(error) => {
            *chain = rollback;
            return Err(format!(
                "RevisionDispatchExecutionAdapter: canonical open_revision rejected after preflight: {error}"
            ));
        }
    };

    let after = chain.snapshot();
    let after_revision_count = after.revisions.len();
    let committed_revision = after.revision(&revision_id);

    let postcondition_ok = revision_id == expected_revision_id
        && after_revision_count == before_revision_count + 1
        && committed_revision.is_some_and(|revision| {
            revision.revision_key == *revision_key
                && revision.parent_revision_id.as_deref() == parent_revision_id.as_deref()
        });

    if !postcondition_ok {
        *chain = rollback;
        return Err(
            "RevisionDispatchExecutionAdapter: postcondition mismatch; mutation rolled back"
                .to_string(),
        );
    }

    let receipt_fingerprint = receipt_fingerprint(
        request,
        lineage_id,
        &revision_id,
        before_revision_count,
        after_revision_count,
    );

    Ok(RevisionDispatchExecutionReceipt {
        receipt_fingerprint,
        request_fingerprint: request.request_fingerprint().to_string(),
        response_intent_fingerprint: request.response_intent_fingerprint().to_string(),
        dispatch_fingerprint: request.dispatch_fingerprint().to_string(),
        lineage_id: lineage_id.clone(),
        revision_id,
        revision_key: revision_key.clone(),
        parent_revision_id: parent_revision_id.clone(),
        before_revision_count,
        after_revision_count,
        execution_trigger: "explicit_caller_invocation".to_string(),
        authorization_claim: "none".to_string(),
        mutation: "committed".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::{
        execute_revision_dispatch_request, revision_dispatch_execution_adapter_capabilities,
    };
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_revision::ResolutionRevisionChain;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::response_dispatch_request::{
        create_ignore_dispatch_request, create_revision_dispatch_request,
    };
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn fixture() -> (
        ResolutionEvidenceInbox,
        ResolutionReviewSession,
    ) {
        let mut review = ResolutionReviewSession::new("intent-revision-execution").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-revision-execution".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-revision-execution".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-revision-execution".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "customer".to_string(),
            fields: Vec::new(),
        };
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "failed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"failed\"}",
            false,
            "mismatch",
        )
        .unwrap();
        inbox.record(evidence).unwrap();
        (inbox, review)
    }

    #[test]
    fn revision_request_executes_once_and_receipt_correlates_request() {
        let (inbox, review) = fixture();
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

        let before = chain.snapshot();
        let receipt =
            execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain).unwrap();
        let after = chain.snapshot();

        assert_eq!(before.revisions.len(), 0);
        assert_eq!(after.revisions.len(), 1);
        assert_eq!(receipt.request_fingerprint, request.request_fingerprint());
        assert_eq!(receipt.before_revision_count, 0);
        assert_eq!(receipt.after_revision_count, 1);
        assert_eq!(receipt.mutation, "committed");
        assert_eq!(receipt.execution_trigger, "explicit_caller_invocation");
        assert_eq!(receipt.authorization_claim, "none");
        assert!(after.revision(&receipt.revision_id).is_some());
    }

    #[test]
    fn replay_of_same_request_is_rejected_without_second_mutation() {
        let (inbox, review) = fixture();
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

        execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain).unwrap();
        let once = chain.snapshot();

        let replay =
            execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain);
        assert!(replay.is_err());
        assert_eq!(chain.snapshot(), once);
    }

    #[test]
    fn non_revision_request_cannot_enter_mutation_adapter() {
        let (inbox, review) = fixture();
        let mut chain = ResolutionRevisionChain::from_approved(review.snapshot()).unwrap();
        let intent =
            create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")
                .unwrap();
        let request = create_ignore_dispatch_request(&inbox, &intent).unwrap();
        let before = chain.snapshot();

        let result =
            execute_revision_dispatch_request(&inbox, &intent, &request, &mut chain);
        assert!(result.is_err());
        assert_eq!(chain.snapshot(), before);
    }

    #[test]
    fn capability_contract_declares_explicit_mutation_boundary() {
        let contract: serde_json::Value =
            serde_json::from_str(revision_dispatch_execution_adapter_capabilities()).unwrap();
        assert_eq!(contract["role"], "explicit_revision_dispatch_execution_adapter");
        assert_eq!(contract["semantics"]["mutation"], "committed_on_success");
        assert_eq!(
            contract["semantics"]["execution_trigger"],
            "explicit_caller_invocation"
        );
        assert_eq!(contract["semantics"]["authorization_claim"], "none");
    }
}
