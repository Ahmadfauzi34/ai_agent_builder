use crate::agent_response_intent::AgentResponseIntent;
use crate::resolution_review::validate_actor;
use crate::resolution_revision::{validate_revision_key, ResolutionRevisionChain};
use crate::response_dispatch_requirements::{
    response_dispatch_requirements, ResponseDispatchRequirements,
};
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const RESPONSE_DISPATCH_REQUEST_V1: &str =
    include_str!("../docs/response-dispatch-request.v1.json");
const MAX_INFORMATION_REQUEST_BYTES: usize = 4096;

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

fn length_prefixed(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn json_string_or_null(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{}\"", json_escape(value)))
        .unwrap_or_else(|| "null".to_string())
}

fn json_string_array(values: &[String]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .collect::<Vec<_>>()
            .join(",")
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResponseDispatchRequestPayload {
    Ignore,
    RequestInformation {
        information_request: String,
        actor: Option<String>,
    },
    ProposeRevision {
        lineage_id: String,
        revision_key: String,
        parent_revision_id: Option<String>,
    },
}

impl ResponseDispatchRequestPayload {
    fn canonical(&self) -> String {
        match self {
            Self::Ignore => "ignore".to_string(),
            Self::RequestInformation {
                information_request,
                actor,
            } => format!(
                "request_information|request={}|actor={}|",
                length_prefixed(information_request),
                length_prefixed(actor.as_deref().unwrap_or(""))
            ),
            Self::ProposeRevision {
                lineage_id,
                revision_key,
                parent_revision_id,
            } => format!(
                "propose_revision|lineage={}|key={}|parent={}|",
                length_prefixed(lineage_id),
                length_prefixed(revision_key),
                length_prefixed(parent_revision_id.as_deref().unwrap_or(""))
            ),
        }
    }

    fn to_json(&self) -> String {
        match self {
            Self::Ignore => "{\"kind\":\"ignore\"}".to_string(),
            Self::RequestInformation {
                information_request,
                actor,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"request_information\",",
                    "\"information_request\":\"{}\",",
                    "\"actor\":{}",
                    "}}"
                ),
                json_escape(information_request),
                json_string_or_null(actor.as_deref()),
            ),
            Self::ProposeRevision {
                lineage_id,
                revision_key,
                parent_revision_id,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"propose_revision\",",
                    "\"lineage_id\":\"{}\",",
                    "\"revision_key\":\"{}\",",
                    "\"parent_revision_id\":{}",
                    "}}"
                ),
                json_escape(lineage_id),
                json_escape(revision_key),
                json_string_or_null(parent_revision_id.as_deref()),
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseDispatchRequest {
    selected_action: EvidenceResponseAction,
    evidence_kind: String,
    response_intent_fingerprint: String,
    dispatch_fingerprint: String,
    requirements_fingerprint: String,
    authority: String,
    operation: String,
    executor_contract: String,
    payload_mode: String,
    payload: ResponseDispatchRequestPayload,
    bound_requirements: Vec<String>,
    unbound_optional_requirements: Vec<String>,
    request_fingerprint: String,
}

impl ResponseDispatchRequest {
    pub fn selected_action(&self) -> EvidenceResponseAction {
        self.selected_action
    }

    pub fn evidence_kind(&self) -> &str {
        &self.evidence_kind
    }

    pub fn response_intent_fingerprint(&self) -> &str {
        &self.response_intent_fingerprint
    }

    pub fn dispatch_fingerprint(&self) -> &str {
        &self.dispatch_fingerprint
    }

    pub fn requirements_fingerprint(&self) -> &str {
        &self.requirements_fingerprint
    }

    pub fn authority(&self) -> &str {
        &self.authority
    }

    pub fn operation(&self) -> &str {
        &self.operation
    }

    pub fn executor_contract(&self) -> &str {
        &self.executor_contract
    }

    pub fn payload_mode(&self) -> &str {
        &self.payload_mode
    }

    pub fn payload(&self) -> &ResponseDispatchRequestPayload {
        &self.payload
    }

    pub fn request_fingerprint(&self) -> &str {
        &self.request_fingerprint
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-dispatch-request.v1\",",
                "\"role\":\"bound_serializable_dispatch_request_nonexecuting\",",
                "\"selected_action\":\"{}\",",
                "\"evidence_kind\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"dispatch_fingerprint\":\"{}\",",
                "\"requirements_fingerprint\":\"{}\",",
                "\"authority\":\"{}\",",
                "\"operation\":\"{}\",",
                "\"executor_contract\":\"{}\",",
                "\"payload_mode\":\"{}\",",
                "\"payload\":{},",
                "\"bound_requirements\":{},",
                "\"unbound_optional_requirements\":{},",
                "\"request_fingerprint\":\"{}\",",
                "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
                "\"execution_authorized\":false,",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"none\"",
                "}}"
            ),
            self.selected_action.as_str(),
            json_escape(&self.evidence_kind),
            json_escape(&self.response_intent_fingerprint),
            json_escape(&self.dispatch_fingerprint),
            json_escape(&self.requirements_fingerprint),
            json_escape(&self.authority),
            json_escape(&self.operation),
            json_escape(&self.executor_contract),
            json_escape(&self.payload_mode),
            self.payload.to_json(),
            json_string_array(&self.bound_requirements),
            json_string_array(&self.unbound_optional_requirements),
            json_escape(&self.request_fingerprint),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseDispatchRequestBindability {
    pub bindable: bool,
    pub status: String,
    pub selected_action: EvidenceResponseAction,
    pub payload_mode: Option<String>,
    pub unresolved_required: Vec<String>,
    pub execution_authorized: bool,
    pub mutation: String,
}

impl ResponseDispatchRequestBindability {
    pub fn to_json(&self) -> String {
        let payload_mode = self
            .payload_mode
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-dispatch-request-bindability.v1\",",
                "\"bindable\":{},",
                "\"status\":\"{}\",",
                "\"selected_action\":\"{}\",",
                "\"payload_mode\":{},",
                "\"unresolved_required\":{},",
                "\"execution_authorized\":{},",
                "\"mutation\":\"{}\"" ,
                "}}"
            ),
            self.bindable,
            json_escape(&self.status),
            self.selected_action.as_str(),
            payload_mode,
            json_string_array(&self.unresolved_required),
            self.execution_authorized,
            json_escape(&self.mutation),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseDispatchRequestPreflight {
    pub ready: bool,
    pub status: String,
    pub response_intent_matches: bool,
    pub dispatch_matches: bool,
    pub requirements_match: bool,
    pub request_fingerprint_matches: bool,
    pub execution_authorized: bool,
    pub mutation: String,
}

impl ResponseDispatchRequestPreflight {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-dispatch-request-preflight.v1\",",
                "\"ready\":{},",
                "\"status\":\"{}\",",
                "\"checks\":{{",
                    "\"response_intent_matches\":{},",
                    "\"dispatch_matches\":{},",
                    "\"requirements_match\":{},",
                    "\"request_fingerprint_matches\":{}",
                "}},",
                "\"execution_authorized\":{},",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"{}\"" ,
                "}}"
            ),
            self.ready,
            json_escape(&self.status),
            self.response_intent_matches,
            self.dispatch_matches,
            self.requirements_match,
            self.request_fingerprint_matches,
            self.execution_authorized,
            json_escape(&self.mutation),
        )
    }
}

fn current_requirements(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    expected_action: EvidenceResponseAction,
) -> Result<ResponseDispatchRequirements, String> {
    if intent.selected_action() != expected_action {
        return Err(format!(
            "ResponseDispatchRequest: expected selected action {}, got {}",
            expected_action.as_str(),
            intent.selected_action().as_str()
        ));
    }
    let requirements = response_dispatch_requirements(inbox, intent);
    if !requirements.ready {
        return Err(format!(
            "ResponseDispatchRequest: requirements are not ready: {}",
            requirements.status
        ));
    }
    Ok(requirements)
}

fn request_fingerprint(
    requirements: &ResponseDispatchRequirements,
    payload: &ResponseDispatchRequestPayload,
) -> Result<String, String> {
    let dispatch = requirements
        .dispatch_fingerprint
        .as_deref()
        .ok_or_else(|| "ResponseDispatchRequest: dispatch fingerprint missing".to_string())?;
    let requirement_fingerprint = requirements
        .requirements_fingerprint
        .as_deref()
        .ok_or_else(|| "ResponseDispatchRequest: requirements fingerprint missing".to_string())?;
    let canonical = format!(
        "v1|intent={}|dispatch={dispatch}|requirements={requirement_fingerprint}|action={}|payload={}|",
        requirements.response_intent_fingerprint,
        requirements.selected_action.as_str(),
        payload.canonical(),
    );
    Ok(fnv1a64(canonical.bytes()))
}

fn validate_requirement_accounting(
    requirements: &ResponseDispatchRequirements,
    bound_requirements: &[String],
    unbound_optional_requirements: &[String],
) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for name in bound_requirements {
        if !seen.insert(name.as_str()) {
            return Err(format!(
                "ResponseDispatchRequest: duplicate bound requirement {name}"
            ));
        }
        if !requirements.requirements.iter().any(|item| item.name == *name) {
            return Err(format!(
                "ResponseDispatchRequest: unknown bound requirement {name}"
            ));
        }
    }

    for name in unbound_optional_requirements {
        if !seen.insert(name.as_str()) {
            return Err(format!(
                "ResponseDispatchRequest: requirement {name} cannot be both bound and unbound"
            ));
        }
        let requirement = requirements
            .requirements
            .iter()
            .find(|item| item.name == *name)
            .ok_or_else(|| {
                format!("ResponseDispatchRequest: unknown unbound requirement {name}")
            })?;
        if requirement.required {
            return Err(format!(
                "ResponseDispatchRequest: required requirement {name} cannot remain unbound"
            ));
        }
    }

    for requirement in &requirements.requirements {
        if requirement.required
            && !bound_requirements
                .iter()
                .any(|name| name == &requirement.name)
        {
            return Err(format!(
                "ResponseDispatchRequest: required requirement {} is not bound",
                requirement.name
            ));
        }
        if !requirement.required
            && !bound_requirements
                .iter()
                .any(|name| name == &requirement.name)
            && !unbound_optional_requirements
                .iter()
                .any(|name| name == &requirement.name)
        {
            return Err(format!(
                "ResponseDispatchRequest: optional requirement {} is not accounted for",
                requirement.name
            ));
        }
    }
    Ok(())
}

fn build_request(
    requirements: ResponseDispatchRequirements,
    payload: ResponseDispatchRequestPayload,
    bound_requirements: Vec<String>,
    unbound_optional_requirements: Vec<String>,
) -> Result<ResponseDispatchRequest, String> {
    validate_requirement_accounting(
        &requirements,
        &bound_requirements,
        &unbound_optional_requirements,
    )?;
    let request_fingerprint = request_fingerprint(&requirements, &payload)?;
    let dispatch_fingerprint = requirements
        .dispatch_fingerprint
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: dispatch fingerprint missing".to_string())?;
    let requirements_fingerprint = requirements
        .requirements_fingerprint
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: requirements fingerprint missing".to_string())?;
    let authority = requirements
        .authority
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: authority missing".to_string())?;
    let operation = requirements
        .operation
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: operation missing".to_string())?;
    let executor_contract = requirements
        .executor_contract
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: executor contract missing".to_string())?;
    let payload_mode = requirements
        .payload_mode
        .clone()
        .ok_or_else(|| "ResponseDispatchRequest: payload mode missing".to_string())?;

    Ok(ResponseDispatchRequest {
        selected_action: requirements.selected_action,
        evidence_kind: requirements.evidence_kind,
        response_intent_fingerprint: requirements.response_intent_fingerprint,
        dispatch_fingerprint,
        requirements_fingerprint,
        authority,
        operation,
        executor_contract,
        payload_mode,
        payload,
        bound_requirements,
        unbound_optional_requirements,
        request_fingerprint,
    })
}

pub fn response_dispatch_request_capabilities() -> &'static str {
    RESPONSE_DISPATCH_REQUEST_V1
}

pub fn response_dispatch_request_bindability(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> ResponseDispatchRequestBindability {
    let requirements = response_dispatch_requirements(inbox, intent);
    if !requirements.ready {
        return ResponseDispatchRequestBindability {
            bindable: false,
            status: format!("closed:{}", requirements.status),
            selected_action: intent.selected_action(),
            payload_mode: requirements.payload_mode,
            unresolved_required: Vec::new(),
            execution_authorized: false,
            mutation: "none".to_string(),
        };
    }

    if intent.selected_action() == EvidenceResponseAction::Reverify {
        return ResponseDispatchRequestBindability {
            bindable: false,
            status: "runtime_handle_binding_deferred".to_string(),
            selected_action: intent.selected_action(),
            payload_mode: requirements.payload_mode,
            unresolved_required: requirements
                .requirements
                .iter()
                .filter(|item| item.required)
                .map(|item| item.name.clone())
                .collect(),
            execution_authorized: false,
            mutation: "none".to_string(),
        };
    }

    ResponseDispatchRequestBindability {
        bindable: true,
        status: "serializable_payload_bindable".to_string(),
        selected_action: intent.selected_action(),
        payload_mode: requirements.payload_mode,
        unresolved_required: Vec::new(),
        execution_authorized: false,
        mutation: "none".to_string(),
    }
}

pub fn create_ignore_dispatch_request(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> Result<ResponseDispatchRequest, String> {
    let requirements = current_requirements(inbox, intent, EvidenceResponseAction::Ignore)?;
    build_request(
        requirements,
        ResponseDispatchRequestPayload::Ignore,
        Vec::new(),
        Vec::new(),
    )
}

pub fn create_information_dispatch_request(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    information_request: impl Into<String>,
    actor: Option<String>,
) -> Result<ResponseDispatchRequest, String> {
    let requirements =
        current_requirements(inbox, intent, EvidenceResponseAction::RequestInformation)?;
    let information_request = information_request.into();
    if information_request.trim().is_empty() {
        return Err(
            "ResponseDispatchRequest: information_request must not be empty".to_string(),
        );
    }
    if information_request.len() > MAX_INFORMATION_REQUEST_BYTES {
        return Err(format!(
            "ResponseDispatchRequest: information_request exceeds {MAX_INFORMATION_REQUEST_BYTES} bytes"
        ));
    }
    let actor = match actor {
        Some(actor) => Some(validate_actor(actor)?),
        None => None,
    };

    let mut bound = vec!["information_request".to_string()];
    let mut optional = vec!["diagnostic_materialization".to_string()];
    if actor.is_some() {
        bound.push("actor".to_string());
    } else {
        optional.push("actor".to_string());
    }

    build_request(
        requirements,
        ResponseDispatchRequestPayload::RequestInformation {
            information_request,
            actor,
        },
        bound,
        optional,
    )
}

pub fn create_revision_dispatch_request(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    chain: &ResolutionRevisionChain,
    revision_key: impl Into<String>,
    parent_revision_id: Option<String>,
) -> Result<ResponseDispatchRequest, String> {
    let requirements =
        current_requirements(inbox, intent, EvidenceResponseAction::ProposeRevision)?;

    let root = chain.root_snapshot();
    let root_approval = root
        .approval
        .as_ref()
        .ok_or_else(|| "ResponseDispatchRequest: revision chain root approval missing".to_string())?;
    if root.workflow.intent_id != inbox.target_intent_id()
        || root.workflow.revision != inbox.target_workflow_revision()
        || root_approval.approval_id != inbox.target_approval_id()
    {
        return Err(
            "ResponseDispatchRequest: revision chain root does not match evidence inbox target"
                .to_string(),
        );
    }

    let revision_key = validate_revision_key(revision_key)?;
    if let Some(parent) = parent_revision_id.as_deref() {
        if !chain.revision_compile_eligible(parent)? {
            return Err(format!(
                "ResponseDispatchRequest: parent revision is not approved/compile-eligible: {parent}"
            ));
        }
    }

    let mut bound = vec!["revision_key".to_string()];
    let mut optional = Vec::new();
    if parent_revision_id.is_some() {
        bound.push("parent_revision_id".to_string());
    } else {
        optional.push("parent_revision_id".to_string());
    }

    build_request(
        requirements,
        ResponseDispatchRequestPayload::ProposeRevision {
            lineage_id: chain.lineage_id().to_string(),
            revision_key,
            parent_revision_id,
        },
        bound,
        optional,
    )
}

pub fn preflight_response_dispatch_request(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
    request: &ResponseDispatchRequest,
) -> ResponseDispatchRequestPreflight {
    let requirements = response_dispatch_requirements(inbox, intent);
    let response_intent_matches =
        request.response_intent_fingerprint == intent.response_intent_fingerprint()
            && request.selected_action == intent.selected_action()
            && request.evidence_kind == intent.evidence_kind();

    let dispatch_matches = requirements.ready
        && requirements.dispatch_fingerprint.as_deref() == Some(&request.dispatch_fingerprint)
        && requirements.authority.as_deref() == Some(&request.authority)
        && requirements.operation.as_deref() == Some(&request.operation)
        && requirements.executor_contract.as_deref() == Some(&request.executor_contract)
        && requirements.payload_mode.as_deref() == Some(&request.payload_mode);

    let requirements_match = requirements.ready
        && requirements.requirements_fingerprint.as_deref()
            == Some(&request.requirements_fingerprint);

    let request_fingerprint_matches = if requirements.ready {
        request_fingerprint(&requirements, &request.payload)
            .is_ok_and(|fingerprint| fingerprint == request.request_fingerprint)
    } else {
        false
    };

    let ready = response_intent_matches
        && dispatch_matches
        && requirements_match
        && request_fingerprint_matches;

    ResponseDispatchRequestPreflight {
        ready,
        status: if ready {
            "ready_nonexecuting".to_string()
        } else {
            "closed:request_or_upstream_drift".to_string()
        },
        response_intent_matches,
        dispatch_matches,
        requirements_match,
        request_fingerprint_matches,
        execution_authorized: false,
        mutation: "none".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        create_ignore_dispatch_request, create_information_dispatch_request,
        create_revision_dispatch_request, preflight_response_dispatch_request,
        response_dispatch_request_bindability, response_dispatch_request_capabilities,
        ResponseDispatchRequestPayload,
    };
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution_review::ResolutionReviewSession;
    use crate::resolution_revision::ResolutionRevisionChain;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn fixture() -> (
        crate::resolution::ResolutionSnapshot,
        RuntimeSubjectProjection,
        ResolutionEvidenceInbox,
        ResolutionReviewSession,
    ) {
        let mut review =
            ResolutionReviewSession::new("intent-dispatch-request").unwrap();
        review.submit("agent").unwrap();
        let approval = review.approve("customer").unwrap();
        let resolution = review.snapshot().workflow;
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: approval.approval_id,
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-dispatch-request".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-dispatch-request".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-dispatch-request".to_string(),
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
        (resolution, projection, inbox, review)
    }

    #[test]
    fn reverify_remains_deferred_until_real_runtime_handles_exist() {
        let (_, _, inbox, _) = fixture();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        let projection = response_dispatch_request_bindability(&inbox, &intent);
        assert!(!projection.bindable);
        assert_eq!(projection.status, "runtime_handle_binding_deferred");
        assert!(projection
            .unresolved_required
            .iter()
            .any(|name| name == "compiled_graph"));
        assert!(!projection.execution_authorized);
    }

    #[test]
    fn ignore_request_is_empty_and_nonexecuting() {
        let (_, _, inbox, _) = fixture();
        let intent =
            create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")
                .unwrap();
        let request = create_ignore_dispatch_request(&inbox, &intent).unwrap();

        assert!(matches!(request.payload(), ResponseDispatchRequestPayload::Ignore));
        assert_eq!(request.executor_contract(), "none");
        assert_eq!(request.operation(), "no_op");
        assert!(preflight_response_dispatch_request(&inbox, &intent, &request).ready);
        assert!(request.to_json().contains("\"execution_authorized\":false"));
    }

    #[test]
    fn information_request_binds_serializable_payload_without_diagnostic_creation() {
        let (resolution, _, inbox, _) = fixture();
        let before = resolution.clone();
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
            "Provide the missing tensor rank",
            Some("reviewer".to_string()),
        )
        .unwrap();

        assert_eq!(request.executor_contract(), "external_resolution_review");
        assert_eq!(request.operation(), "request_information");
        assert!(preflight_response_dispatch_request(&inbox, &intent, &request).ready);
        assert_eq!(resolution, before);
        assert!(request.to_json().contains("Provide the missing tensor rank"));
    }

    #[test]
    fn revision_request_reuses_canonical_key_validation_and_does_not_open_revision() {
        let (resolution, _, inbox, review) = fixture();
        let chain = ResolutionRevisionChain::from_approved(review.snapshot()).unwrap();
        let before = chain.snapshot();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent",
        )
        .unwrap();

        let request = create_revision_dispatch_request(
            &inbox,
            &intent,
            &chain,
            "fix-shape",
            None,
        )
        .unwrap();

        assert_eq!(request.executor_contract(), "ResolutionRevisionChain");
        assert_eq!(request.operation(), "open_revision");
        assert!(preflight_response_dispatch_request(&inbox, &intent, &request).ready);
        assert_eq!(chain.snapshot(), before);

        let invalid = create_revision_dispatch_request(
            &inbox,
            &intent,
            &chain,
            "invalid key with spaces",
            None,
        );
        assert!(invalid.is_err());
    }

    #[test]
    fn revision_request_rejects_unrelated_revision_chain() {
        let (_, _, inbox, _) = fixture();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent",
        )
        .unwrap();

        let mut foreign_review = ResolutionReviewSession::new("other-intent").unwrap();
        foreign_review.submit("agent").unwrap();
        foreign_review.approve("customer").unwrap();
        let foreign_chain =
            ResolutionRevisionChain::from_approved(foreign_review.snapshot()).unwrap();

        assert!(create_revision_dispatch_request(
            &inbox,
            &intent,
            &foreign_chain,
            "fix",
            None,
        )
        .is_err());
    }

    #[test]
    fn request_fingerprint_is_deterministic_and_payload_sensitive() {
        let (_, _, inbox, _) = fixture();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::RequestInformation,
            "agent",
        )
        .unwrap();

        let a = create_information_dispatch_request(
            &inbox,
            &intent,
            "Need rank",
            None,
        )
        .unwrap();
        let b = create_information_dispatch_request(
            &inbox,
            &intent,
            "Need rank",
            None,
        )
        .unwrap();
        let c = create_information_dispatch_request(
            &inbox,
            &intent,
            "Need dtype",
            None,
        )
        .unwrap();

        assert_eq!(a.request_fingerprint(), b.request_fingerprint());
        assert_ne!(a.request_fingerprint(), c.request_fingerprint());
    }

    #[test]
    fn capability_contract_declares_runtime_handles_deferred_and_no_execution() {
        let contract = response_dispatch_request_capabilities();
        assert!(contract.contains("\"runtime_handle_binding_deferred\""));
        assert!(contract.contains("\"execution_authorized\": false"));
        assert!(contract.contains("\"bound_serializable_dispatch_request_nonexecuting\""));
    }
}
