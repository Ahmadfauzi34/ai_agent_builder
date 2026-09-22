use crate::agent_response_intent::{
    preflight_response_intent, AgentResponseIntent, ResponseIntentPreflight,
};
use crate::runtime_evidence_interpretation::EvidenceResponseAction;
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const RESPONSE_INTENT_EXECUTION_GATE_V1: &str =
    include_str!("../docs/response-intent-execution-gate.v1.json");

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
pub struct ResponseDispatchRoute {
    pub authority: String,
    pub operation: String,
    pub route_mode: String,
    pub explicit_executor_required: bool,
    pub prerequisite: String,
}

impl ResponseDispatchRoute {
    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"authority\":\"{}\",",
                "\"operation\":\"{}\",",
                "\"route_mode\":\"{}\",",
                "\"explicit_executor_required\":{},",
                "\"prerequisite\":\"{}\"" ,
                "}}"
            ),
            json_escape(&self.authority),
            json_escape(&self.operation),
            json_escape(&self.route_mode),
            self.explicit_executor_required,
            json_escape(&self.prerequisite),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseExecutionGate {
    pub dispatchable: bool,
    pub gate_status: String,
    pub selected_action: EvidenceResponseAction,
    pub response_intent_fingerprint: String,
    pub route: Option<ResponseDispatchRoute>,
    pub dispatch_fingerprint: Option<String>,
    pub execution_authorized: bool,
    pub mutation: String,
    pub preflight: ResponseIntentPreflight,
}

impl ResponseExecutionGate {
    pub fn to_json(&self) -> String {
        let route = self
            .route
            .as_ref()
            .map(ResponseDispatchRoute::to_json)
            .unwrap_or_else(|| "null".to_string());
        let dispatch_fingerprint = self
            .dispatch_fingerprint
            .as_ref()
            .map(|value| format!("\"{}\"", json_escape(value)))
            .unwrap_or_else(|| "null".to_string());

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.response-intent-execution-gate.v1\",",
                "\"role\":\"validated_dispatch_projection_only\",",
                "\"dispatchable\":{},",
                "\"gate_status\":\"{}\",",
                "\"selected_action\":\"{}\",",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"route\":{},",
                "\"dispatch_fingerprint\":{},",
                "\"execution_authorized\":{},",
                "\"mutation\":\"{}\",",
                "\"preflight\":{}",
                "}}"
            ),
            self.dispatchable,
            json_escape(&self.gate_status),
            self.selected_action.as_str(),
            json_escape(&self.response_intent_fingerprint),
            route,
            dispatch_fingerprint,
            self.execution_authorized,
            json_escape(&self.mutation),
            self.preflight.to_json(),
        )
    }
}

fn reverify_operation(intent: &AgentResponseIntent) -> Result<&'static str, String> {
    match intent.evidence_kind() {
        "graph_verifier_receipt" => Ok("CompiledGraph.verifyFlat"),
        "math_program_verifier_receipt" => Ok("MathProgram.verifyFlat"),
        "direct_math_verifier_receipt" => Ok("DirectMath.verifyAgainstMathProgramV9"),
        "vector_verifier_receipt" => Ok("mathVerifyVectors"),
        other => Err(format!(
            "ResponseIntentExecutionGate: response action reverify has no typed verifier route for evidence kind {other}"
        )),
    }
}

fn route_for(intent: &AgentResponseIntent) -> Result<ResponseDispatchRoute, String> {
    match intent.selected_action() {
        EvidenceResponseAction::Ignore => Ok(ResponseDispatchRoute {
            authority: "none".to_string(),
            operation: "no_op".to_string(),
            route_mode: "terminal_noop".to_string(),
            explicit_executor_required: false,
            prerequisite: "keep evidence recorded; perform no Resolution mutation".to_string(),
        }),
        EvidenceResponseAction::Reverify => Ok(ResponseDispatchRoute {
            authority: "runtime_verifier".to_string(),
            operation: reverify_operation(intent)?.to_string(),
            route_mode: "authority_handoff".to_string(),
            explicit_executor_required: true,
            prerequisite:
                "caller must explicitly invoke the projected verifier with its required runtime inputs"
                    .to_string(),
        }),
        EvidenceResponseAction::RequestInformation => Ok(ResponseDispatchRoute {
            authority: "external_resolution_review".to_string(),
            operation: "request_information".to_string(),
            route_mode: "authority_handoff".to_string(),
            explicit_executor_required: true,
            prerequisite: "caller supplies the information request; if materialized as ResolutionDiagnostic, use existing review/revision diagnostic boundary".to_string(),
        }),
        EvidenceResponseAction::ProposeRevision => Ok(ResponseDispatchRoute {
            authority: "ResolutionRevisionChain".to_string(),
            operation: "open_revision".to_string(),
            route_mode: "authority_handoff".to_string(),
            explicit_executor_required: true,
            prerequisite:
                "caller supplies revision_key and optional parent_revision_id to the existing revision boundary"
                    .to_string(),
        }),
    }
}

pub fn response_intent_execution_gate_capabilities() -> &'static str {
    RESPONSE_INTENT_EXECUTION_GATE_V1
}

pub fn response_intent_execution_gate(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> ResponseExecutionGate {
    let preflight = preflight_response_intent(inbox, intent);

    if !preflight.ready {
        return ResponseExecutionGate {
            dispatchable: false,
            gate_status: format!("closed:{}", preflight.status),
            selected_action: intent.selected_action(),
            response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
            route: None,
            dispatch_fingerprint: None,
            execution_authorized: false,
            mutation: "none".to_string(),
            preflight,
        };
    }

    let route = match route_for(intent) {
        Ok(route) => route,
        Err(error) => {
            return ResponseExecutionGate {
                dispatchable: false,
                gate_status: format!("closed:route_error:{error}"),
                selected_action: intent.selected_action(),
                response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
                route: None,
                dispatch_fingerprint: None,
                execution_authorized: false,
                mutation: "none".to_string(),
                preflight,
            }
        }
    };

    let canonical = format!(
        "v1|intent={}|action={}|authority={}|operation={}|mode={}|",
        intent.response_intent_fingerprint(),
        intent.selected_action().as_str(),
        route.authority,
        route.operation,
        route.route_mode,
    );
    let dispatch_fingerprint = fnv1a64(canonical.bytes());

    ResponseExecutionGate {
        dispatchable: true,
        gate_status: "dispatchable_nonexecuting".to_string(),
        selected_action: intent.selected_action(),
        response_intent_fingerprint: intent.response_intent_fingerprint().to_string(),
        route: Some(route),
        dispatch_fingerprint: Some(dispatch_fingerprint),
        execution_authorized: false,
        mutation: "none".to_string(),
        preflight,
    }
}

#[cfg(test)]
mod tests {
    use super::{response_intent_execution_gate, response_intent_execution_gate_capabilities};
    use crate::agent_response_intent::create_agent_response_intent;
    use crate::resolution::ResolutionWorkflow;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
    use crate::runtime_evidence_interpretation::EvidenceResponseAction;
    use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

    fn resolved_snapshot(intent_id: &str) -> crate::resolution::ResolutionSnapshot {
        let mut workflow = ResolutionWorkflow::new(intent_id).unwrap();
        workflow.submit().unwrap();
        workflow.finalize_resolution().unwrap();
        workflow.snapshot()
    }

    fn projection(
        intent_id: &str,
        workflow_revision: u64,
        subject_identity: &str,
    ) -> RuntimeSubjectProjection {
        RuntimeSubjectProjection {
            schema: "burn-research.runtime-subject-projection.v1".to_string(),
            intent_id: intent_id.to_string(),
            workflow_revision,
            approval_id: "approval-gate".to_string(),
            subject_kind: "effective-spec".to_string(),
            subject_identity: subject_identity.to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: subject_identity.to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy-gate".to_string(),
            authorization_policy_revision: 1,
            authorization_is_revision: false,
            approver: "owner".to_string(),
            fields: Vec::new(),
        }
    }

    fn failed_graph_inbox() -> (crate::resolution::ResolutionSnapshot, ResolutionEvidenceInbox) {
        let resolution = resolved_snapshot("intent-gate");
        let projection = projection("intent-gate", resolution.revision, "spec-gate");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();
        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "failed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"x\"}",
            false,
            "mismatch",
        )
        .unwrap();
        inbox.record(evidence).unwrap();
        (resolution, inbox)
    }

    #[test]
    fn propose_revision_routes_to_revision_authority_without_execution_authority() {
        let (resolution, inbox) = failed_graph_inbox();
        let before = resolution.clone();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent",
        )
        .unwrap();

        let gate = response_intent_execution_gate(&inbox, &intent);
        assert!(gate.dispatchable);
        assert_eq!(gate.gate_status, "dispatchable_nonexecuting");
        assert!(!gate.execution_authorized);
        assert_eq!(gate.mutation, "none");

        let route = gate.route.as_ref().unwrap();
        assert_eq!(route.authority, "ResolutionRevisionChain");
        assert_eq!(route.operation, "open_revision");
        assert_eq!(route.route_mode, "authority_handoff");
        assert!(route.explicit_executor_required);
        assert!(gate.dispatch_fingerprint.is_some());
        assert_eq!(resolution, before);
    }

    #[test]
    fn reverify_routes_by_typed_evidence_kind() {
        let (_, inbox) = failed_graph_inbox();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        let gate = response_intent_execution_gate(&inbox, &intent);
        assert!(gate.dispatchable);
        let route = gate.route.as_ref().unwrap();
        assert_eq!(route.authority, "runtime_verifier");
        assert_eq!(route.operation, "CompiledGraph.verifyFlat");
        assert!(route.explicit_executor_required);
        assert!(!gate.execution_authorized);
    }

    #[test]
    fn ignore_is_explicit_noop_route_and_keeps_evidence_recorded() {
        let (_, inbox) = failed_graph_inbox();
        let before_len = inbox.len();
        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Ignore,
            "agent",
        )
        .unwrap();

        let gate = response_intent_execution_gate(&inbox, &intent);
        assert!(gate.dispatchable);
        let route = gate.route.as_ref().unwrap();
        assert_eq!(route.authority, "none");
        assert_eq!(route.operation, "no_op");
        assert_eq!(route.route_mode, "terminal_noop");
        assert!(!route.explicit_executor_required);
        assert_eq!(inbox.len(), before_len);
    }

    #[test]
    fn tampered_intent_closes_gate_without_route() {
        let (_, inbox) = failed_graph_inbox();
        let mut intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent",
        )
        .unwrap();

        // Test-only module access proves the gate consumes the existing preflight,
        // rather than silently repairing a modified intent.
        intent.response_intent_fingerprint = "fnv1a64:tampered".to_string();

        let gate = response_intent_execution_gate(&inbox, &intent);
        assert!(!gate.dispatchable);
        assert!(gate.gate_status.contains("response_intent_fingerprint_mismatch"));
        assert!(gate.route.is_none());
        assert!(gate.dispatch_fingerprint.is_none());
        assert!(!gate.execution_authorized);
    }

    #[test]
    fn capability_contract_declares_dispatch_is_not_authorization() {
        let contract = response_intent_execution_gate_capabilities();
        assert!(contract.contains("\"validated_dispatch_projection_only\""));
        assert!(contract.contains("\"execution_authorized\": false"));
        assert!(contract.contains("\"dispatchable does not mean authorized\""));
    }
}
