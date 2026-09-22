use crate::runtime_evidence_interpretation::{
    interpret_recorded_evidence, EvidenceResponseAction,
};
use crate::runtime_resolution_evidence::ResolutionEvidenceInbox;

const AGENT_RESPONSE_INTENT_V1: &str =
    include_str!("../docs/agent-response-intent.v1.json");

const MAX_SELECTOR_BYTES: usize = 256;

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

fn validate_selector(selector: impl Into<String>) -> Result<String, String> {
    let selector = selector.into();
    if selector.trim().is_empty() {
        return Err("AgentResponseIntent: selector must not be empty".to_string());
    }
    if selector.len() > MAX_SELECTOR_BYTES {
        return Err(format!(
            "AgentResponseIntent: selector {} bytes exceeds limit {MAX_SELECTOR_BYTES}",
            selector.len()
        ));
    }
    Ok(selector)
}

fn response_intent_fingerprint_for(
    selector: &str,
    entry_index: usize,
    evidence_fingerprint: &str,
    evidence_kind: &str,
    evidence_outcome: &str,
    action: EvidenceResponseAction,
    target_boundary: &str,
    handoff: &str,
) -> String {
    let canonical = format!(
        "v1|selector={selector}|entry={entry_index}|evidence={evidence_fingerprint}|kind={evidence_kind}|outcome={evidence_outcome}|action={}|target={target_boundary}|handoff={handoff}|",
        action.as_str(),
    );
    fnv1a64(canonical.bytes())
}

fn evidence_fingerprint(inbox: &ResolutionEvidenceInbox, entry_index: usize) -> Result<String, String> {
    let evidence = inbox.observations().get(entry_index).ok_or_else(|| {
        format!(
            "AgentResponseIntent: entry index {entry_index} is outside recorded inbox length {}",
            inbox.len()
        )
    })?;
    Ok(fnv1a64(evidence.to_json().bytes()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AgentResponseIntent {
    selector: String,
    entry_index: usize,
    evidence_kind: String,
    evidence_outcome: String,
    evidence_fingerprint: String,
    selected_action: EvidenceResponseAction,
    target_boundary: String,
    handoff: String,
    response_intent_fingerprint: String,
}

impl AgentResponseIntent {
    pub fn selector(&self) -> &str {
        &self.selector
    }

    pub fn entry_index(&self) -> usize {
        self.entry_index
    }

    pub fn evidence_kind(&self) -> &str {
        &self.evidence_kind
    }

    pub fn evidence_outcome(&self) -> &str {
        &self.evidence_outcome
    }

    pub fn evidence_fingerprint(&self) -> &str {
        &self.evidence_fingerprint
    }

    pub fn selected_action(&self) -> EvidenceResponseAction {
        self.selected_action
    }

    pub fn target_boundary(&self) -> &str {
        &self.target_boundary
    }

    pub fn handoff(&self) -> &str {
        &self.handoff
    }

    pub fn response_intent_fingerprint(&self) -> &str {
        &self.response_intent_fingerprint
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.agent-response-intent.v1\",",
                "\"role\":\"explicit_agent_selection_nonexecuting\",",
                "\"selector\":\"{}\",",
                "\"entry_index\":{},",
                "\"evidence\":{{",
                    "\"kind\":\"{}\",",
                    "\"outcome\":\"{}\",",
                    "\"fingerprint\":\"{}\"",
                "}},",
                "\"selection\":{{",
                    "\"authority\":\"explicit_caller_agent\",",
                    "\"selected_action\":\"{}\",",
                    "\"target_boundary\":\"{}\",",
                    "\"handoff\":\"{}\"",
                "}},",
                "\"response_intent_fingerprint\":\"{}\",",
                "\"fingerprint_algorithm\":\"fnv1a64_noncryptographic\",",
                "\"execution_authorized\":false,",
                "\"execution_effect\":\"none\",",
                "\"mutation\":{{",
                    "\"inbox\":\"none\",",
                    "\"resolution\":\"none\",",
                    "\"revision\":\"none\",",
                    "\"diagnostic\":\"none\",",
                    "\"verifier\":\"none\"",
                "}}",
                "}}"
            ),
            json_escape(&self.selector),
            self.entry_index,
            json_escape(&self.evidence_kind),
            json_escape(&self.evidence_outcome),
            json_escape(&self.evidence_fingerprint),
            self.selected_action.as_str(),
            json_escape(&self.target_boundary),
            json_escape(&self.handoff),
            json_escape(&self.response_intent_fingerprint),
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResponseIntentPreflight {
    pub ready: bool,
    pub status: String,
    pub entry_exists: bool,
    pub evidence_matches: bool,
    pub candidate_available: bool,
    pub candidate_snapshot_matches: bool,
    pub intent_fingerprint_matches: bool,
    pub execution_authorized: bool,
}

impl ResponseIntentPreflight {
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.agent-response-intent-preflight.v1\",",
                "\"ready\":{},",
                "\"status\":\"{}\",",
                "\"checks\":{{",
                    "\"entry_exists\":{},",
                    "\"evidence_matches\":{},",
                    "\"candidate_available\":{},",
                    "\"candidate_snapshot_matches\":{},",
                    "\"intent_fingerprint_matches\":{}",
                "}},",
                "\"execution_authorized\":{},",
                "\"execution_effect\":\"none\",",
                "\"mutation\":\"none\"",
                "}}"
            ),
            self.ready,
            json_escape(&self.status),
            self.entry_exists,
            self.evidence_matches,
            self.candidate_available,
            self.candidate_snapshot_matches,
            self.intent_fingerprint_matches,
            self.execution_authorized,
        )
    }
}

pub fn agent_response_intent_capabilities() -> &'static str {
    AGENT_RESPONSE_INTENT_V1
}

pub fn create_agent_response_intent(
    inbox: &ResolutionEvidenceInbox,
    entry_index: usize,
    action: EvidenceResponseAction,
    selector: impl Into<String>,
) -> Result<AgentResponseIntent, String> {
    let selector = validate_selector(selector)?;
    let interpretation = interpret_recorded_evidence(inbox, entry_index)?;
    let candidate = interpretation.candidate(action).ok_or_else(|| {
        format!(
            "AgentResponseIntent: response action {} is not present in interpretation",
            action.as_str()
        )
    })?;

    if !candidate.available {
        return Err(format!(
            "AgentResponseIntent: response action {} is not available from recorded evidence entry {entry_index}; use an independent review/revision API if another reason exists",
            action.as_str()
        ));
    }

    let target_boundary = candidate.target_boundary.clone();
    let handoff = candidate.handoff.clone();
    let evidence_fingerprint = evidence_fingerprint(inbox, entry_index)?;
    let response_intent_fingerprint = response_intent_fingerprint_for(
        &selector,
        entry_index,
        &evidence_fingerprint,
        &interpretation.evidence_kind,
        &interpretation.outcome,
        action,
        &target_boundary,
        &handoff,
    );

    Ok(AgentResponseIntent {
        selector,
        entry_index,
        evidence_kind: interpretation.evidence_kind,
        evidence_outcome: interpretation.outcome,
        evidence_fingerprint,
        selected_action: action,
        target_boundary,
        handoff,
        response_intent_fingerprint,
    })
}

pub fn preflight_response_intent(
    inbox: &ResolutionEvidenceInbox,
    intent: &AgentResponseIntent,
) -> ResponseIntentPreflight {
    let Some(_) = inbox.observations().get(intent.entry_index) else {
        return ResponseIntentPreflight {
            ready: false,
            status: "evidence_entry_missing".to_string(),
            entry_exists: false,
            evidence_matches: false,
            candidate_available: false,
            candidate_snapshot_matches: false,
            intent_fingerprint_matches: false,
            execution_authorized: false,
        };
    };

    let current_fingerprint = match evidence_fingerprint(inbox, intent.entry_index) {
        Ok(value) => value,
        Err(_) => {
            return ResponseIntentPreflight {
                ready: false,
                status: "evidence_entry_missing".to_string(),
                entry_exists: false,
                evidence_matches: false,
                candidate_available: false,
                candidate_snapshot_matches: false,
                intent_fingerprint_matches: false,
                execution_authorized: false,
            }
        }
    };
    let evidence_matches = current_fingerprint == intent.evidence_fingerprint;
    if !evidence_matches {
        return ResponseIntentPreflight {
            ready: false,
            status: "evidence_fingerprint_mismatch".to_string(),
            entry_exists: true,
            evidence_matches: false,
            candidate_available: false,
            candidate_snapshot_matches: false,
            intent_fingerprint_matches: false,
            execution_authorized: false,
        };
    }

    let interpretation = match interpret_recorded_evidence(inbox, intent.entry_index) {
        Ok(value) => value,
        Err(_) => {
            return ResponseIntentPreflight {
                ready: false,
                status: "interpretation_unavailable".to_string(),
                entry_exists: true,
                evidence_matches: true,
                candidate_available: false,
                candidate_snapshot_matches: false,
                intent_fingerprint_matches: false,
                execution_authorized: false,
            }
        }
    };
    let Some(candidate) = interpretation.candidate(intent.selected_action) else {
        return ResponseIntentPreflight {
            ready: false,
            status: "candidate_missing".to_string(),
            entry_exists: true,
            evidence_matches: true,
            candidate_available: false,
            candidate_snapshot_matches: false,
            intent_fingerprint_matches: false,
            execution_authorized: false,
        };
    };

    let candidate_available = candidate.available;
    let candidate_snapshot_matches = candidate.target_boundary == intent.target_boundary
        && candidate.handoff == intent.handoff
        && interpretation.evidence_kind == intent.evidence_kind
        && interpretation.outcome == intent.evidence_outcome;
    let expected_intent_fingerprint = response_intent_fingerprint_for(
        &intent.selector,
        intent.entry_index,
        &intent.evidence_fingerprint,
        &intent.evidence_kind,
        &intent.evidence_outcome,
        intent.selected_action,
        &intent.target_boundary,
        &intent.handoff,
    );
    let intent_fingerprint_matches =
        expected_intent_fingerprint == intent.response_intent_fingerprint;

    let ready = candidate_available && candidate_snapshot_matches && intent_fingerprint_matches;
    ResponseIntentPreflight {
        ready,
        status: if ready {
            "ready_nonexecuting".to_string()
        } else if !candidate_available {
            "candidate_no_longer_available".to_string()
        } else if !candidate_snapshot_matches {
            "candidate_snapshot_mismatch".to_string()
        } else {
            "response_intent_fingerprint_mismatch".to_string()
        },
        entry_exists: true,
        evidence_matches: true,
        candidate_available,
        candidate_snapshot_matches,
        intent_fingerprint_matches,
        execution_authorized: false,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        agent_response_intent_capabilities, create_agent_response_intent,
        preflight_response_intent, MAX_SELECTOR_BYTES,
    };
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
    fn explicit_failed_verifier_selection_becomes_nonexecuting_intent() {
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

        let intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent-a",
        )
        .unwrap();

        assert_eq!(
            intent.selected_action(),
            EvidenceResponseAction::ProposeRevision
        );
        assert_eq!(intent.entry_index(), 0);
        assert!(!intent.response_intent_fingerprint().is_empty());
        assert_eq!(inbox.len(), before_len);
        assert_eq!(resolution, before_resolution);

        let json = intent.to_json();
        assert!(json.contains("\"selected_action\":\"propose_revision\""));
        assert!(json.contains("\"execution_authorized\":false"));
        assert!(json.contains("\"execution_effect\":\"none\""));

        let preflight = preflight_response_intent(&inbox, &intent);
        assert!(preflight.ready);
        assert_eq!(preflight.status, "ready_nonexecuting");
        assert!(preflight.intent_fingerprint_matches);
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn unavailable_evidence_derived_action_fails_closed() {
        let resolution = resolved_snapshot("intent-a");
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let passed = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "passed",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"x\"}",
            true,
            "match",
        )
        .unwrap();
        inbox.record(passed).unwrap();

        let err = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent-a",
        )
        .unwrap_err();
        assert!(err.contains("not available from recorded evidence"));
        assert!(err.contains("independent review/revision API"));
    }

    #[test]
    fn intent_fingerprint_is_deterministic_and_action_specific() {
        let resolution = resolved_snapshot("intent-a");
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

        let first = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent-a",
        )
        .unwrap();
        let second = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent-a",
        )
        .unwrap();
        let revision = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::ProposeRevision,
            "agent-a",
        )
        .unwrap();

        assert_eq!(
            first.response_intent_fingerprint(),
            second.response_intent_fingerprint()
        );
        assert_ne!(
            first.response_intent_fingerprint(),
            revision.response_intent_fingerprint()
        );
    }

    #[test]
    fn preflight_fails_closed_when_intent_snapshot_is_tampered() {
        let resolution = resolved_snapshot("intent-a");
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

        let mut intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent-a",
        )
        .unwrap();
        intent.evidence_fingerprint = "fnv1a64:tampered".to_string();

        let preflight = preflight_response_intent(&inbox, &intent);
        assert!(!preflight.ready);
        assert_eq!(preflight.status, "evidence_fingerprint_mismatch");
        assert!(preflight.entry_exists);
        assert!(!preflight.evidence_matches);
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn preflight_rejects_tampered_response_intent_fingerprint() {
        let resolution = resolved_snapshot("intent-a");
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

        let mut intent = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "agent-a",
        )
        .unwrap();
        intent.response_intent_fingerprint = "fnv1a64:tampered".to_string();

        let preflight = preflight_response_intent(&inbox, &intent);
        assert!(!preflight.ready);
        assert_eq!(
            preflight.status,
            "response_intent_fingerprint_mismatch"
        );
        assert!(preflight.evidence_matches);
        assert!(preflight.candidate_available);
        assert!(preflight.candidate_snapshot_matches);
        assert!(!preflight.intent_fingerprint_matches);
        assert!(!preflight.execution_authorized);
    }

    #[test]
    fn selector_is_required_and_bounded() {
        let resolution = resolved_snapshot("intent-a");
        let projection = projection("intent-a", resolution.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&resolution, &projection).unwrap();

        let failed = RuntimeEvidence::bound_vector_verifier_receipt(
            &projection,
            1,
            "failed",
            false,
            "mismatch",
        )
        .unwrap();
        inbox.record(failed).unwrap();

        let err = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "",
        )
        .unwrap_err();
        assert!(err.contains("selector must not be empty"));

        let err = create_agent_response_intent(
            &inbox,
            0,
            EvidenceResponseAction::Reverify,
            "x".repeat(MAX_SELECTOR_BYTES + 1),
        )
        .unwrap_err();
        assert!(err.contains("exceeds limit"));
    }

    #[test]
    fn capability_contract_keeps_intent_nonexecuting() {
        let contract = agent_response_intent_capabilities();
        assert!(contract.contains("\"explicit_agent_selection_nonexecuting\""));
        assert!(contract.contains("\"execution\": \"none\""));
        assert!(contract.contains("\"preflight only validates current correspondence"));
    }
}
