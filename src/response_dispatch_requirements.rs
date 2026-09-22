use crate::agent_response_intent::AgentResponseIntent;
use crate::response_intent_execution_gate::{
    response_intent_execution_gate, ResponseExecutionGate,
};
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const RESPONSE_DISPATCH_REQUIREMENTS_V1: &str =
    include_str!("../docs/response-dispatch-requirements.v1.json");

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchRequirement {
    pub name: String,
    pub required: bool,
    pub value_class: String,
    pub source_authority: String,
}

impl DispatchRequirement {
    fn new(
        name: &'static str,
        required: bool,
        value_class: &'static str,
        source_authority: &'static str,
    ) -> Self {
        Self {
            name: name.to_string(),
            required,
            value_class: value_class.to_string(),
            source_authority: source_authority.to_string(),
        }
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"name\":\"{}\",",
                "\"required\":{},",
                "\"value_class\":\"{}\",",
                "\"source_authority\":\"{}\"" ,
                "}}"
            ),
            json_escape(&self.name),
            self.required,
            json_escape(&self.value_class),
            json_escape(&self.source_authority),
        )
    }

    fn canonical(&self) -> String {
        format!(
            "{}:{}:{}:{}",
            self.name, self.required, self.value_class, self.source_authority
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseDispatchRequirements {
    pub ready: bool,
    pub status: String,
    pub selected_action: EvidenceResponseAction,
    pub evidence_kind: String,
    pub response_intent_fingerprint: String,
    pub dispatch_fingerprint: Option<String>,
    pub requirements_fingerprint: Option<String>,
    pub authority: Option<String>,
    pub operation: Option<String>,
    pub executor_contract: Option<String>,
    pub payload_mode: Option<String>,
    pub requirements: Vec<DispatchRequirement>,
    pub execution_authorized: bool,
    pub mutation: String,
    pub gate: ResponseExecutionGate,
}

impl ResponseDispatchRequirements {
    pub fn to_json(&self) -> String {
        let dispatch_fingerprint = self
            .dispatch_fingerprint
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let requirements_fingerprint = self
            .requirements_fingerprint
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let authority = self
            .authority
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let operation = self
            .operation
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let executor_contract = self
            .executor_contract
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let payload_mode = self
            .payload_mode
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());
        let requirements = self
            .requirements
            .iter()
            .map(DispatchRequirement::to_json)
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-dispatch-requirements.v1\",",
                "\"role\":\"typed_handoff_requirements_projection_only\",",
                "\"ready\":{},",
                "\"status\":\"{}\",",
                "\"selected_action\":\"{}\",",
                "\"evidence_kind\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"dispatch_fingerprint\":{},",
                "\"requirements_fingerprint\":{},",
                "\"authority\":{},",
                "\"operation\":{},",
                "\"executor_contract\":{},",
                "\"payload_mode\":{},",
                "\"requirements\":[{}],",
                "\"execution_authorized\":{},",
                "\"mutation\":\"{}\",",
                "\"gate\":{}",
                "}}"
            ),
            self.ready,
            json_escape(&self.status),
            self.selected_action.as_str(),
            json_escape(&self.evidence_kind),
            json_escape(&self.response_intent_fingerprint),
            dispatch_fingerprint,
            requirements_fingerprint,
            authority,
            operation,
            executor_contract,
            payload_mode,
            requirements,
            self.execution_authorized,
            json_escape(&self.mutation),
            self.gate.to_json(),
        )
    }
}

fn requirement(
    name: &'static str,
    required: bool,
    value_class: &'static str,
    source_authority: &'static str,
) -> DispatchRequirement {
    DispatchRequirement::new(name, required, value_class, source_authority)
}

fn reverify_requirements(
    evidence_kind: &str,
) -> Result<(&'static str, &'static str, Vec<DispatchRequirement>), String> {
    match evidence_kind {
        "graph_verifier_receipt" => Ok((
            "CompiledGraph.verifyFlat",
            "typed_runtime_verifier_inputs",
            vec![
                requirement("compiled_graph", true, "runtime_graph_handle", "caller"),
                requirement("layer_registry", true, "runtime_registry_handle", "caller"),
                requirement("input_tensor", true, "runtime_tensor", "caller"),
                requirement("candidate", true, "f32_buffer", "caller"),
                requirement("abs_tol", true, "f32_scalar", "caller"),
                requirement("rel_tol", true, "f32_scalar", "caller"),
            ],
        )),
        "math_program_verifier_receipt" => Ok((
            "MathProgram.verifyFlat",
            "typed_runtime_verifier_inputs",
            vec![
                requirement("math_program", true, "runtime_program_handle", "caller"),
                requirement("program_inputs", true, "runtime_tensor_set", "caller"),
                requirement("candidate", true, "f32_buffer", "caller"),
                requirement("abs_tol", true, "f32_scalar", "caller"),
                requirement("rel_tol", true, "f32_scalar", "caller"),
            ],
        )),
        "direct_math_verifier_receipt" => Ok((
            "DirectMath.verifyAgainstMathProgramV9",
            "typed_operation_verifier_inputs",
            vec![
                requirement("canonical_operation", true, "operation_id", "response_intent"),
                requirement("operation_inputs", true, "runtime_tensor_set", "caller"),
                requirement("u32_params", true, "u32_parameter_buffer", "caller"),
                requirement("f32_params", true, "f32_parameter_buffer", "caller"),
                requirement("abs_tol", true, "f32_scalar", "caller"),
                requirement("rel_tol", true, "f32_scalar", "caller"),
            ],
        )),
        "vector_verifier_receipt" => Ok((
            "mathVerifyVectors",
            "typed_runtime_verifier_inputs",
            vec![
                requirement("reference", true, "f32_buffer", "caller"),
                requirement("candidate", true, "f32_buffer", "caller"),
                requirement("abs_tol", true, "f32_scalar", "caller"),
                requirement("rel_tol", true, "f32_scalar", "caller"),
            ],
        )),
        other => Err(format!(
            "ResponseDispatchRequirements: no reverify requirement profile for evidence kind {other}"
        )),
    }
}

fn requirement_profile(
    intent: &AgentResponseIntent,
) -> Result<(&'static str, &'static str, Vec<DispatchRequirement>), String> {
    match intent.selected_action() {
        EvidenceResponseAction::Ignore => Ok(("none", "none", Vec::new())),
        EvidenceResponseAction::Reverify => reverify_requirements(intent.evidence_kind()),
        EvidenceResponseAction::RequestInformation => Ok((
            "external_resolution_review.request_information",
            "caller_structured_payload",
            vec![
                requirement("information_request", true, "structured_request", "caller"),
                requirement("actor", false, "identity", "caller_identity_layer"),
                requirement(
                    "diagnostic_materialization",
                    false,
                    "resolution_diagnostic_request",
                    "caller",
                ),
            ],
        )),
        EvidenceResponseAction::ProposeRevision => Ok((
            "ResolutionRevisionChain.open_revision",
            "caller_structured_payload",
            vec![
                requirement("revision_key", true, "revision_key", "caller"),
                requirement(
                    "parent_revision_id",
                    false,
                    "revision_id",
                    "caller_or_resolution_lineage",
                ),
            ],
        )),
    }
}

fn requirements_fingerprint(
    dispatch_fingerprint: &str,
    executor_contract: &str,
    payload_mode: &str,
    requirements: &[DispatchRequirement],
) -> String {
    let canonical = format!(
        "v1|dispatch={dispatch_fingerprint}|contract={executor_contract}|mode={payload_mode}|requirements={}|",
        requirements
            .iter()
            .map(DispatchRequirement::canonical)
            .collect::<Vec<_>>()
            .join(";")
    );
    fnv1a64(canonical.bytes())
}

pub fn response_dispatch_requirements_capabilities() -> &'static str {
    RESPONSE_DISPATCH_REQUIREMENTS_V1
}

pub fn response_dispatch_requirements(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> ResponseDispatchRequirements {
    let gate = response_intent_execution_gate(inbox, intent);

    if !gate.dispatchable {
        return ResponseDispatchRequirements {
            ready: false,
            status: format!("closed:{}", gate.gate_status),
            selected_action: intent.selected_action(),
            evidence_kind: intent.evidence_kind().to_string(),
            response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
            dispatch_fingerprint: None,
            requirements_fingerprint: None,
            authority: None,
            operation: None,
            executor_contract: None,
            payload_mode: None,
            requirements: Vec::new(),
            execution_authorized: false,
            mutation: "none".to_string(),
            gate,
        };
    }

    let route = match gate.route.as_ref() {
        Some(route) => route,
        None => {
            return ResponseDispatchRequirements {
                ready: false,
                status: "closed:dispatchable_gate_missing_route".to_string(),
                selected_action: intent.selected_action(),
                evidence_kind: intent.evidence_kind().to_string(),
                response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
                dispatch_fingerprint: None,
                requirements_fingerprint: None,
                authority: None,
                operation: None,
                executor_contract: None,
                payload_mode: None,
                requirements: Vec::new(),
                execution_authorized: false,
                mutation: "none".to_string(),
                gate,
            }
        }
    };

    let (executor_contract, payload_mode, requirements) = match requirement_profile(intent) {
        Ok(profile) => profile,
        Err(error) => {
            return ResponseDispatchRequirements {
                ready: false,
                status: format!("closed:requirement_profile_error:{error}"),
                selected_action: intent.selected_action(),
                evidence_kind: intent.evidence_kind().to_string(),
                response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
                dispatch_fingerprint: gate.dispatch_fingerprint.clone(),
                requirements_fingerprint: None,
                authority: Some(route.authority.clone()),
                operation: Some(route.operation.clone()),
                executor_contract: None,
                payload_mode: None,
                requirements: Vec::new(),
                execution_authorized: false,
                mutation: "none".to_string(),
                gate,
            }
        }
    };

    if executor_contract != "none" && route.operation != executor_contract {
        return ResponseDispatchRequirements {
            ready: false,
            status: format!(
                "closed:executor_contract_mismatch:route={} profile={executor_contract}",
                route.operation
            ),
            selected_action: intent.selected_action(),
            evidence_kind: intent.evidence_kind().to_string(),
            response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
            dispatch_fingerprint: gate.dispatch_fingerprint.clone(),
            requirements_fingerprint: None,
            authority: Some(route.authority.clone()),
            operation: Some(route.operation.clone()),
            executor_contract: Some(executor_contract.to_string()),
            payload_mode: Some(payload_mode.to_string()),
            requirements,
            execution_authorized: false,
            mutation: "none".to_string(),
            gate,
        };
    }

    let dispatch_fingerprint = gate
        .dispatch_fingerprint
        .clone()
        .expect("dispatchable gate must carry dispatch fingerprint");
    let fingerprint = requirements_fingerprint(
        &dispatch_fingerprint,
        executor_contract,
        payload_mode,
        &requirements,
    );

    ResponseDispatchRequirements {
        ready: true,
        status: "ready_nonexecuting".to_string(),
        selected_action: intent.selected_action(),
        evidence_kind: intent.evidence_kind().to_string(),
        response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
        dispatch_fingerprint: Some(dispatch_fingerprint),
        requirements_fingerprint: Some(fingerprint),
        authority: Some(route.authority.clone()),
        operation: Some(route.operation.clone()),
        executor_contract: Some(executor_contract.to_string()),
        payload_mode: Some(payload_mode.to_string()),
        requirements,
        execution_authorized: false,
        mutation: "none".to_string(),
        gate,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        response_dispatch_requirements, response_dispatch_requirements_capabilities,
    };
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution::ResolutionWorkflow;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn fixture(
        passed: bool,
    ) -> (
        crate::resolution::ResolutionSnapshot,
        RuntimeSubjectProjection,
        ResolutionEvidenceInbox,
    ) {
        let mut workflow = ResolutionWorkflow::new("intent-dispatch-requirements").unwrap();
        workflow.submit().unwrap();
        workflow.finalize_resolution().unwrap();
        let resolution = workflow.snapshot();
        let projection = RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: resolution.intent_id.clone(),
            workflow_revision: resolution.revision,
            approval_id: "approval-dispatch-requirements".to_string(),
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec-dispatch-requirements".to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: "spec-dispatch-requirements".to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-dispatch-requirements".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "probe".to_string(),
            fields: Vec::new(),
        };
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "graph",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"graph\"}",
            passed,
            if passed { "match" } else { "mismatch" },
        )
        .unwrap();
        inbox.record(evidence).unwrap();
        (resolution, projection, inbox)
    }

    #[test]
    fn failed_graph_reverify_has_machine_readable_verifier_requirements() {
        let (_, _, inbox) = fixture(false);
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        let projection = response_dispatch_requirements(&inbox, &intent);
        assert!(projection.ready);
        assert_eq!(projection.status, "ready_nonexecuting");
        assert_eq!(
            projection.executor_contract.as_deref(),
            Some("CompiledGraph.verifyFlat")
        );
        assert_eq!(
            projection.payload_mode.as_deref(),
            Some("typed_runtime_verifier_inputs")
        );
        assert!(projection
            .requirements
            .iter()
            .any(|item| item.name == "compiled_graph" && item.required));
        assert!(projection
            .requirements
            .iter()
            .any(|item| item.name == "candidate" && item.required));
        assert!(projection.requirements_fingerprint.is_some());
        assert!(!projection.execution_authorized);
    }

    #[test]
    fn propose_revision_exposes_exact_structured_caller_payload() {
        let (_, _, inbox) = fixture(false);
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent",
        )
        .unwrap();
        let projection = response_dispatch_requirements(&inbox, &intent);

        assert!(projection.ready);
        assert_eq!(
            projection.executor_contract.as_deref(),
            Some("ResolutionRevisionChain.open_revision")
        );
        assert!(projection
            .requirements
            .iter()
            .any(|item| item.name == "revision_key" && item.required));
        assert!(projection
            .requirements
            .iter()
            .any(|item| item.name == "parent_revision_id" && !item.required));
        assert!(!projection.execution_authorized);
    }

    #[test]
    fn ignore_has_explicit_empty_payload_contract() {
        let (_, _, inbox) = fixture(false);
        let intent =
            create_agent_response_intent(&inbox, 0, EvidenceResponseAction::Ignore, "agent")
                .unwrap();
        let projection = response_dispatch_requirements(&inbox, &intent);

        assert!(projection.ready);
        assert_eq!(projection.executor_contract.as_deref(), Some("none"));
        assert_eq!(projection.payload_mode.as_deref(), Some("none"));
        assert!(projection.requirements.is_empty());
        assert_eq!(projection.authority.as_deref(), Some("none"));
        assert!(!projection.execution_authorized);
    }

    #[test]
    fn stale_intent_closes_requirement_projection() {
        let (resolution, projection, inbox_a) = fixture(false);
        let intent = create_agent_response_intent(
            &inbox_a,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        let mut inbox_b = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let different = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            2,
            "different",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"different\"}",
            true,
            "match",
        )
        .unwrap();
        inbox_b.record(different).unwrap();

        let closed = response_dispatch_requirements(&inbox_b, &intent);
        assert!(!closed.ready);
        assert!(closed.status.contains("closed:"));
        assert!(closed.requirements.is_empty());
        assert!(closed.requirements_fingerprint.is_none());
        assert!(!closed.execution_authorized);
    }

    #[test]
    fn capability_contract_keeps_requirements_projection_nonexecuting() {
        let contract = response_dispatch_requirements_capabilities();
        assert!(contract.contains("\"typed_handoff_requirements_projection_only\""));
        assert!(contract.contains("\"execution_authorized\": false"));
        assert!(contract.contains("\"requirements_fingerprint\""));
    }
}
