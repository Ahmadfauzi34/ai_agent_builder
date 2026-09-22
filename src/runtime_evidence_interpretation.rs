use crate::runtime_resolution_evidence::{ResolutionEvidenceInbox, RuntimeEvidence};

const RUNTIME_EVIDENCE_INTERPRETATION_V1: &str =
    include_str!("../docs/runtime-evidence-interpretation.v1.json");

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\""),
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceResponseAction {
    Ignore,
    Reverify,
    RequestInformation,
    ProposeRevision,
}

impl EvidenceResponseAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ignore => "ignore",
            Self::Reverify => "reverify",
            Self::RequestInformation => "request_information",
            Self::ProposeRevision => "propose_revision",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceResponseCandidate {
    pub action: EvidenceResponseAction,
    pub available: bool,
    pub evidence_relation: String,
    pub target_boundary: String,
    pub handoff: String,
    pub requires_external_judgment: bool,
}

impl EvidenceResponseCandidate {
    fn new(
        action: EvidenceResponseAction,
        available: bool,
        evidence_relation: impl Into<String>,
        target_boundary: impl Into<String>,
        handoff: impl Into<String>,
    ) -> Self {
        Self {
            action,
            available,
            evidence_relation: evidence_relation.into(),
            target_boundary: target_boundary.into(),
            handoff: handoff.into(),
            requires_external_judgment: true,
        }
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"action\":\"{}\",",
                "\"available\":{},",
                "\"evidence_relation\":\"{}\",",
                "\"target_boundary\":\"{}\",",
                "\"handoff\":\"{}\",",
                "\"requires_external_judgment\":{},",
                "\"projection_mutation\":\"none\"",
                "}}"
            ),
            self.action.as_str(),
            self.available,
            json_escape(&self.evidence_relation),
            json_escape(&self.target_boundary),
            json_escape(&self.handoff),
            self.requires_external_judgment,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceInterpretationSnapshot {
    pub entry_index: usize,
    pub evidence_kind: String,
    pub outcome: String,
    pub source_authority: String,
    pub evidence_authority: String,
    pub transport_integrity: String,
    pub candidates: Vec<EvidenceResponseCandidate>,
}

impl EvidenceInterpretationSnapshot {
    pub fn candidate(
        &self,
        action: EvidenceResponseAction,
    ) -> Option<&EvidenceResponseCandidate> {
        self.candidates
            .iter()
            .find(|candidate| candidate.action == action)
    }

    pub fn available(&self, action: EvidenceResponseAction) -> bool {
        self.candidate(action)
            .is_some_and(|candidate| candidate.available)
    }

    pub fn to_json(&self) -> String {
        let candidates = self
            .candidates
            .iter()
            .map(EvidenceResponseCandidate::to_json)
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.runtime-evidence-interpretation.v1\",",
                "\"role\":\"candidate_projection_only\",",
                "\"entry_index\":{},",
                "\"evidence\":{{",
                    "\"kind\":\"{}\",",
                    "\"outcome\":\"{}\",",
                    "\"source_authority\":\"{}\",",
                    "\"evidence_authority\":\"{}\",",
                    "\"transport_integrity\":\"{}\"",
                "}},",
                "\"selection\":{{",
                    "\"selected_action\":null,",
                    "\"default_action\":null,",
                    "\"ranking\":\"none\",",
                    "\"ordering\":\"stable_protocol_order_not_preference\"",
                "}},",
                "\"mutation\":{{",
                    "\"interpreter\":\"none\",",
                    "\"inbox\":\"none\",",
                    "\"resolution\":\"none\",",
                    "\"revision\":\"none\",",
                    "\"diagnostic\":\"none\",",
                    "\"verifier_execution\":\"none\"",
                "}},",
                "\"candidates\":[{}]",
                "}}"
            ),
            self.entry_index,
            json_escape(&self.evidence_kind),
            json_escape(&self.outcome),
            json_escape(&self.source_authority),
            json_escape(&self.evidence_authority),
            json_escape(&self.transport_integrity),
            candidates,
        )
    }
}

fn verifier_handoff(evidence: &RuntimeEvidence) -> &'static str {
    match evidence.kind() {
        "graph_verifier_receipt" => {
            "CompiledGraph.verifyFlat / workspaceVerifyGraphReceipt / workspaceVerifySemanticGraphReceipt"
        }
        "math_program_verifier_receipt" => "MathProgram.verifyFlat",
        "direct_math_verifier_receipt" => "DirectMath.verifyAgainstMathProgramV9",
        "vector_verifier_receipt" => "mathVerifyVectors",
        _ => "not_applicable",
    }
}

pub fn runtime_evidence_interpretation_capabilities() -> &'static str {
    RUNTIME_EVIDENCE_INTERPRETATION_V1
}

pub fn interpret_recorded_evidence(
    inbox: &ResolutionEvidenceInbox,
    entry_index: usize,
) -> Result<EvidenceInterpretationSnapshot, String> {
    let evidence = inbox.observations().get(entry_index).ok_or_else(|| {
        format!(
            "RuntimeEvidenceInterpretation: entry index {entry_index} is outside recorded inbox length {}",
            inbox.len()
        )
    })?;

    let kind = evidence.kind();
    let outcome = evidence.outcome();
    let verifier_receipt = matches!(
        kind,
        "graph_verifier_receipt"
            | "math_program_verifier_receipt"
            | "direct_math_verifier_receipt"
            | "vector_verifier_receipt"
    );
    let negative_observation = matches!(outcome, "fault" | "failed");

    let candidates = vec![
        EvidenceResponseCandidate::new(
            EvidenceResponseAction::Ignore,
            true,
            "recorded_evidence_may_remain_without_resolution_change",
            "ResolutionEvidenceInbox",
            "keep evidence recorded; perform no Resolution action",
        ),
        EvidenceResponseCandidate::new(
            EvidenceResponseAction::Reverify,
            verifier_receipt,
            if verifier_receipt {
                "typed_verifier_receipt_can_be_checked_again"
            } else {
                "agent_fault_has_no_verifier_receipt_to_repeat"
            },
            "runtime_verifier",
            verifier_handoff(evidence),
        ),
        EvidenceResponseCandidate::new(
            EvidenceResponseAction::RequestInformation,
            negative_observation,
            if negative_observation {
                "fault_or_failed_verification_may_require_external_context"
            } else {
                "passed_verification_does_not_itself_request_more_information"
            },
            "external_resolution_review",
            "caller-defined information request; if a revision review is opened, ResolutionRevisionChain::request_diagnostic remains the existing diagnostic boundary",
        ),
        EvidenceResponseCandidate::new(
            EvidenceResponseAction::ProposeRevision,
            negative_observation,
            if negative_observation {
                "fault_or_failed_verification_can_be_presented_as_revision_evidence"
            } else {
                "passed_verification_does_not_itself_propose_a_revision"
            },
            "resolution_revision",
            "ResolutionRevisionChain::open_revision or SubjectBoundRevisionChain::open_revision",
        ),
    ];

    Ok(EvidenceInterpretationSnapshot {
        entry_index,
        evidence_kind: kind.to_string(),
        outcome: outcome.to_string(),
        source_authority: evidence.source_authority().to_string(),
        evidence_authority: evidence.evidence_authority().to_string(),
        transport_integrity: evidence.transport_integrity().to_string(),
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        interpret_recorded_evidence, runtime_evidence_interpretation_capabilities,
        EvidenceResponseAction,
    };
    use crate::resolution::ResolutionWorkflow;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;
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
            approval_id: "approval-1".to_string(),
            subject_kind: "effective-spec".to_string(),
            subject_identity: subject_identity.to_string(),
            effective_spec_schema: "burn-research.effective-spec.v1".to_string(),
            effective_spec_identity: subject_identity.to_string(),
            authorization_schema: "burn-research.authorization.v1".to_string(),
            authorization_policy_id: "policy".to_string(),
            authorization_policy_revision: 3,
            authorization_is_revision: false,
            approver: "owner".to_string(),
            fields: Vec::new(),
        }
    }

    #[test]
    fn failed_verifier_exposes_candidates_without_selecting_or_mutating() {
        let resolution = resolved_snapshot("intent-a");
        let before_resolution = resolution.clone();
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let failed = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "failed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"x\"}",
            false,
            "mismatch",
        )
        .unwrap();
        inbox.record(failed).unwrap();
        let before_len = inbox.len();

        let view = interpret_recorded_evidence(&inbox, 0).unwrap();
        assert!(view.available(EvidenceResponseAction::Ignore));
        assert!(view.available(EvidenceResponseAction::Reverify));
        assert!(view.available(EvidenceResponseAction::RequestInformation));
        assert!(view.available(EvidenceResponseAction::ProposeRevision));
        assert_eq!(inbox.len(), before_len);
        assert_eq!(resolution, before_resolution);

        let json = view.to_json();
        assert!(json.contains("\"selected_action\":null"));
        assert!(json.contains("\"default_action\":null"));
        assert!(json.contains("\"ranking\":\"none\""));
        assert!(json.contains("\"resolution\":\"none\""));
    }

    #[test]
    fn passed_verifier_keeps_reverify_but_does_not_derive_revision_or_information_request() {
        let resolution = resolved_snapshot("intent-a");
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let passed = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            2,
            "passed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"x\"}",
            true,
            "match",
        )
        .unwrap();
        inbox.record(passed).unwrap();

        let view = interpret_recorded_evidence(&inbox, 0).unwrap();
        assert!(view.available(EvidenceResponseAction::Ignore));
        assert!(view.available(EvidenceResponseAction::Reverify));
        assert!(!view.available(EvidenceResponseAction::RequestInformation));
        assert!(!view.available(EvidenceResponseAction::ProposeRevision));
    }

    #[test]
    fn agent_fault_projects_information_and_revision_candidates_not_reverify() {
        let resolution = resolved_snapshot("intent-a");
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let fault = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_RUNTIME",
            "semantic_precondition",
            "workspaceInitUnary",
            "layout.compatible",
            true,
            "layout requires attention",
        )
        .unwrap();
        inbox.record(fault).unwrap();

        let view = interpret_recorded_evidence(&inbox, 0).unwrap();
        assert!(view.available(EvidenceResponseAction::Ignore));
        assert!(!view.available(EvidenceResponseAction::Reverify));
        assert!(view.available(EvidenceResponseAction::RequestInformation));
        assert!(view.available(EvidenceResponseAction::ProposeRevision));
    }

    #[test]
    fn interpreter_cannot_read_unrecorded_entry() {
        let resolution = resolved_snapshot("intent-a");
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let err = interpret_recorded_evidence(&inbox, 0).unwrap_err();
        assert!(err.contains("outside recorded inbox length 0"));
    }

    #[test]
    fn capability_contract_declares_candidate_only_no_selection() {
        let contract = runtime_evidence_interpretation_capabilities();
        assert!(contract.contains("\"candidate_projection_only\""));
        assert!(contract.contains("\"selected_action\": null"));
        assert!(contract.contains("\"ranking\": \"none\""));
        assert!(contract.contains("\"interpreter_mutation\": \"none\""));
    }
}
