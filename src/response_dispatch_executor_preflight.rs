use crate::agent_response_intent::AgentResponseIntent;
use crate::resolution_revision::ResolutionRevisionChain;
use crate::response_dispatch_request::{
    preflight_response_dispatch_request, ResponseDispatchRequest, ResponseDispatchRequestPayload,
};
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const RESPONSE_DISPATCH_EXECUTOR_PREFLIGHT_V1: &str =
    include_str!("../docs/response-dispatch-executor-preflight.v1.json");

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseDispatchExecutorPreflight {
    pub request_ready: bool,
    pub executor_ready: bool,
    pub status: String,
    pub selected_action: EvidenceResponseAction,
    pub authority: String,
    pub operation: String,
    pub executor_contract: String,
    pub request_fingerprint: String,
    pub authority_context: String,
    pub simulated_effect: String,
    pub execution_authorized: bool,
    pub mutation: String,
}

impl ResponseDispatchExecutorPreflight {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-dispatch-executor-preflight.v1\",",
                "\"role\":\"executor_acceptance_preflight_nonexecuting\",",
                "\"request_ready\":{},",
                "\"executor_ready\":{},",
                "\"status\":\"{}\",",
                "\"selected_action\":\"{}\",",
                "\"authority\":\"{}\",",
                "\"operation\":\"{}\",",
                "\"executor_contract\":\"{}\",",
                "\"request_fingerprint\":\"{}\",",
                "\"authority_context\":\"{}\",",
                "\"simulated_effect\":\"{}\",",
                "\"execution_authorized\":{},",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"{}\"" ,
                "}}"
            ),
            self.request_ready,
            self.executor_ready,
            json_escape(&self.status),
            self.selected_action.as_str(),
            json_escape(&self.authority),
            json_escape(&self.operation),
            json_escape(&self.executor_contract),
            json_escape(&self.request_fingerprint),
            json_escape(&self.authority_context),
            json_escape(&self.simulated_effect),
            self.execution_authorized,
            json_escape(&self.mutation),
        )
    }
}

fn closed(
    request: &ResponseDispatchRequest,
    status: impl Into<String>,
    authority_context: impl Into<String>,
) -> ResponseDispatchExecutorPreflight {
    ResponseDispatchExecutorPreflight {
        request_ready: false,
        executor_ready: false,
        status: status.into(),
        selected_action: request.selected_action(),
        authority: request.authority().to_string(),
        operation: request.operation().to_string(),
        executor_contract: request.executor_contract().to_string(),
        request_fingerprint: request.request_fingerprint().to_string(),
        authority_context: authority_context.into(),
        simulated_effect: "none".to_string(),
        execution_authorized: false,
        mutation: "none".to_string(),
    }
}

fn request_base_ready(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
) -> Result<(), ResponseDispatchExecutorPreflight> {
    let preflight = preflight_response_dispatch_request(inbox, intent, request);
    if preflight.ready {
        Ok(())
    } else {
        Err(closed(
            request,
            format!("closed:request_preflight:{}", preflight.status),
            "none",
        ))
    }
}

pub fn response_dispatch_executor_preflight_capabilities() -> &'static str {
    RESPONSE_DISPATCH_EXECUTOR_PREFLIGHT_V1
}

pub fn response_dispatch_executor_preflight(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
) -> ResponseDispatchExecutorPreflight {
    if let Err(closed) = request_base_ready(inbox, intent, request) {
        return closed;
    }

    match request.payload() {
        ResponseDispatchRequestPayload::Ignore => ResponseDispatchExecutorPreflight {
            request_ready: true,
            executor_ready: true,
            status: "ready_noop".to_string(),
            selected_action: request.selected_action(),
            authority: request.authority().to_string(),
            operation: request.operation().to_string(),
            executor_contract: request.executor_contract().to_string(),
            request_fingerprint: request.request_fingerprint().to_string(),
            authority_context: "none".to_string(),
            simulated_effect: "no_op".to_string(),
            execution_authorized: false,
            mutation: "none".to_string(),
        },
        ResponseDispatchRequestPayload::RequestInformation { .. } => {
            ResponseDispatchExecutorPreflight {
                request_ready: true,
                executor_ready: false,
                status: "deferred:external_resolution_review_authority".to_string(),
                selected_action: request.selected_action(),
                authority: request.authority().to_string(),
                operation: request.operation().to_string(),
                executor_contract: request.executor_contract().to_string(),
                request_fingerprint: request.request_fingerprint().to_string(),
                authority_context: "external_resolution_review_not_modeled".to_string(),
                simulated_effect: "none".to_string(),
                execution_authorized: false,
                mutation: "none".to_string(),
            }
        }
        ResponseDispatchRequestPayload::ProposeRevision { .. } => {
            ResponseDispatchExecutorPreflight {
                request_ready: true,
                executor_ready: false,
                status: "deferred:resolution_revision_chain_context_required".to_string(),
                selected_action: request.selected_action(),
                authority: request.authority().to_string(),
                operation: request.operation().to_string(),
                executor_contract: request.executor_contract().to_string(),
                request_fingerprint: request.request_fingerprint().to_string(),
                authority_context: "ResolutionRevisionChain required".to_string(),
                simulated_effect: "none".to_string(),
                execution_authorized: false,
                mutation: "none".to_string(),
            }
        }
    }
}

pub fn preflight_revision_dispatch_executor(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
    chain: &ResolutionRevisionChain,
) -> ResponseDispatchExecutorPreflight {
    if let Err(closed) = request_base_ready(inbox, intent, request) {
        return closed;
    }

    if request.selected_action() != EvidenceResponseAction::ProposeRevision
        || request.authority() != "ResolutionRevisionChain"
        || request.operation() != "open_revision"
        || request.executor_contract() != "ResolutionRevisionChain"
    {
        return closed(
            request,
            "closed:not_revision_dispatch_route",
            "ResolutionRevisionChain",
        );
    }

    let ResponseDispatchRequestPayload::ProposeRevision {
        lineage_id,
        revision_key,
        parent_revision_id,
    } = request.payload()
    else {
        return closed(
            request,
            "closed:revision_route_payload_mismatch",
            "ResolutionRevisionChain",
        );
    };

    if lineage_id != chain.lineage_id() {
        return closed(
            request,
            "closed:revision_lineage_mismatch",
            "ResolutionRevisionChain",
        );
    }

    let before = chain.snapshot();
    let mut probe = chain.clone();
    let result = probe.open_revision(parent_revision_id.as_deref(), revision_key.clone());
    let original_unchanged = chain.snapshot() == before;

    match result {
        Ok(revision_id) if original_unchanged => ResponseDispatchExecutorPreflight {
            request_ready: true,
            executor_ready: true,
            status: "ready_revision_clone_preflight".to_string(),
            selected_action: request.selected_action(),
            authority: request.authority().to_string(),
            operation: request.operation().to_string(),
            executor_contract: request.executor_contract().to_string(),
            request_fingerprint: request.request_fingerprint().to_string(),
            authority_context: format!("ResolutionRevisionChain:{}", chain.lineage_id()),
            simulated_effect: format!("would_open_revision:{revision_id}"),
            execution_authorized: false,
            mutation: "none".to_string(),
        },
        Ok(_) => closed(
            request,
            "closed:unexpected_original_chain_mutation",
            "ResolutionRevisionChain",
        ),
        Err(error) => closed(
            request,
            format!("closed:revision_executor_rejected:{error}"),
            "ResolutionRevisionChain",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preflight_revision_dispatch_executor, response_dispatch_executor_preflight,
        response_dispatch_executor_preflight_capabilities,
    };
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_revision::ResolutionRevisionChain;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::response_dispatch_request::{
        create_ignore_dispatch_request, create_information_dispatch_request,
        create_revision_dispatch_request,
    };
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn fixture() -> (
        ResolutionEvidenceInbox,
        ResolutionReviewSession,
        RuntimeSubjectProjection,
    ) {
        let mut review = ResolutionReviewSession::new("intent-executor-preflight").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-executor-preflight".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-executor-preflight".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-executor-preflight".to_string(),
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
        (inbox, review, projection)
    }

    #[test]
    fn ignore_is_ready_noop_without_execution_authority() {
        let (inbox, _, _) = fixture();
        let intent =
            create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")
                .unwrap();
        let request = create_ignore_dispatch_request(&inbox, &intent).unwrap();
        let preflight = response_dispatch_executor_preflight(&inbox, &intent, &request);

        assert!(preflight.request_ready);
        assert!(preflight.executor_ready);
        assert_eq!(preflight.status, "ready_noop");
        assert_eq!(preflight.simulated_effect, "no_op");
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn external_information_review_stays_deferred() {
        let (inbox, _, _) = fixture();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::RequestInformation,
            "agent",
        )
        .unwrap();
        let request = create_information_dispatch_request(
            &inbox,
            &intent,
            "provide authoritative source",
            Some("agent".to_string()),
        )
        .unwrap();
        let preflight = response_dispatch_executor_preflight(&inbox, &intent, &request);

        assert!(preflight.request_ready);
        assert!(!preflight.executor_ready);
        assert!(preflight.status.contains("external_resolution_review"));
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn revision_clone_preflight_proves_acceptance_without_mutating_chain() {
        let (inbox, review, _) = fixture();
        let chain = ResolutionRevisionChain::from_approved(review.snapshot()).unwrap();
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
        let generic = response_dispatch_executor_preflight(&inbox, &intent, &request);
        assert!(generic.request_ready);
        assert!(!generic.executor_ready);

        let preflight =
            preflight_revision_dispatch_executor(&inbox, &intent, &request, &chain);
        assert!(preflight.request_ready);
        assert!(preflight.executor_ready);
        assert_eq!(preflight.status, "ready_revision_clone_preflight");
        assert!(preflight.simulated_effect.starts_with("would_open_revision:"));
        assert_eq!(chain.snapshot(), before);
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn stale_request_closes_before_authority_preflight() {
        let (inbox, _, projection) = fixture();
        let intent =
            create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")
                .unwrap();
        let request = create_ignore_dispatch_request(&inbox, &intent).unwrap();

        let resolution = {
            let mut review = ResolutionReviewSession::new("intent-executor-preflight").unwrap();
            review.submit("agent").unwrap();
            review.approve("customer").unwrap();
            review.snapshot().workflow
        };
        let mut different = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            2,
            "different",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"different\"}",
            true,
            "match",
        )
        .unwrap();
        different.record(evidence).unwrap();

        let closed = response_dispatch_executor_preflight(&different, &intent, &request);
        assert!(!closed.request_ready);
        assert!(!closed.executor_ready);
        assert!(closed.status.contains("request_preflight"));
    }

    #[test]
    fn capability_contract_keeps_preflight_nonexecuting() {
        let contract: serde_json::Value =
            serde_json::from_str(response_dispatch_executor_preflight_capabilities()).unwrap();
        assert_eq!(contract["role"], "executor_acceptance_preflight_nonexecuting");
        assert_eq!(contract["semantics"]["execution_authorized"], false);
        assert_eq!(contract["semantics"]["mutation"], "none");
    }
}
