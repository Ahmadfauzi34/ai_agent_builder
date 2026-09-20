use wasm_bindgen::prelude::*;

use crate::resolution::ResolutionSnapshot;
use crate::resolution_runtime_bridge::RuntimeSubjectProjection;

const RUNTIME_RESOLUTION_EVIDENCE_V1: &str =
    include_str!("../docs/runtime-resolution-evidence.v1.json");

pub const MAX_RUNTIME_EVIDENCE_ENTRIES: usize = 64;
const MAX_SHORT_FIELD_BYTES: usize = 256;
const MAX_DETAIL_BYTES: usize = 4096;
const MAX_PROGRAM_IDENTITY_BYTES: usize = 16_384;

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

fn validate_nonempty_bounded(
    value: &str,
    max_bytes: usize,
    context: &str,
) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{context}: value must be non-empty"));
    }
    if value.len() > max_bytes {
        return Err(format!(
            "{context}: {} bytes exceeds limit {max_bytes}",
            value.len()
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEvidenceSubject {
    pub intent_id: String,
    pub workflow_revision: u64,
    pub approval_id: String,
    pub subject_kind: String,
    pub subject_identity: String,
    pub authorization_policy_id: String,
    pub authorization_policy_revision: u64,
    pub authorization_is_revision: bool,
}

impl RuntimeEvidenceSubject {
    pub fn from_projection(projection: &RuntimeSubjectProjection) -> Self {
        Self {
            intent_id: projection.intent_id.clone(),
            workflow_revision: projection.workflow_revision,
            approval_id: projection.approval_id.clone(),
            subject_kind: projection.subject_kind.clone(),
            subject_identity: projection.subject_identity.clone(),
            authorization_policy_id: projection.authorization_policy_id.clone(),
            authorization_policy_revision: projection.authorization_policy_revision,
            authorization_is_revision: projection.authorization_is_revision,
        }
    }

    fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"intent_id\":\"{}\",",
                "\"workflow_revision\":{},",
                "\"approval_id\":\"{}\",",
                "\"subject_kind\":\"{}\",",
                "\"subject_identity\":\"{}\",",
                "\"authorization_policy_id\":\"{}\",",
                "\"authorization_policy_revision\":{},",
                "\"authorization_is_revision\":{}",
                "}}"
            ),
            json_escape(&self.intent_id),
            self.workflow_revision,
            json_escape(&self.approval_id),
            json_escape(&self.subject_kind),
            json_escape(&self.subject_identity),
            json_escape(&self.authorization_policy_id),
            self.authorization_policy_revision,
            self.authorization_is_revision,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RuntimeEvidencePayload {
    AgentFault {
        code: String,
        class: String,
        operation: String,
        predicate: String,
        recoverable: bool,
        message: String,
    },
    GraphVerifierReceipt {
        receipt_id: u32,
        label: String,
        program_identity: String,
        passed: bool,
        detail: String,
    },
    VectorVerifierReceipt {
        receipt_id: u32,
        label: String,
        passed: bool,
        detail: String,
    },
}

impl RuntimeEvidencePayload {
    pub fn authority(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "agent_fault_preflight",
            Self::GraphVerifierReceipt { .. } => "wasm_verifier",
            Self::VectorVerifierReceipt { .. } => "wasm_comparator",
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "agent_fault",
            Self::GraphVerifierReceipt { .. } => "graph_verifier_receipt",
            Self::VectorVerifierReceipt { .. } => "vector_verifier_receipt",
        }
    }

    pub fn outcome(&self) -> &'static str {
        match self {
            Self::AgentFault { .. } => "fault",
            Self::GraphVerifierReceipt { passed, .. }
            | Self::VectorVerifierReceipt { passed, .. } => {
                if *passed {
                    "passed"
                } else {
                    "failed"
                }
            }
        }
    }

    fn to_json(&self) -> String {
        match self {
            Self::AgentFault {
                code,
                class,
                operation,
                predicate,
                recoverable,
                message,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"agent_fault\",",
                    "\"authority\":\"agent_fault_preflight\",",
                    "\"outcome\":\"fault\",",
                    "\"code\":\"{}\",",
                    "\"class\":\"{}\",",
                    "\"operation\":\"{}\",",
                    "\"predicate\":\"{}\",",
                    "\"recoverable\":{},",
                    "\"message\":\"{}\"",
                    "}}"
                ),
                json_escape(code),
                json_escape(class),
                json_escape(operation),
                json_escape(predicate),
                recoverable,
                json_escape(message),
            ),
            Self::GraphVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                passed,
                detail,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"graph_verifier_receipt\",",
                    "\"authority\":\"wasm_verifier\",",
                    "\"reference_authority\":\"burn_compiled_graph\",",
                    "\"outcome\":\"{}\",",
                    "\"receipt_id\":{},",
                    "\"label\":\"{}\",",
                    "\"program_identity\":\"{}\",",
                    "\"passed\":{},",
                    "\"detail\":\"{}\"",
                    "}}"
                ),
                if *passed { "passed" } else { "failed" },
                receipt_id,
                json_escape(label),
                json_escape(program_identity),
                passed,
                json_escape(detail),
            ),
            Self::VectorVerifierReceipt {
                receipt_id,
                label,
                passed,
                detail,
            } => format!(
                concat!(
                    "{{",
                    "\"kind\":\"vector_verifier_receipt\",",
                    "\"authority\":\"wasm_comparator\",",
                    "\"reference_authority\":\"caller_supplied\",",
                    "\"outcome\":\"{}\",",
                    "\"receipt_id\":{},",
                    "\"label\":\"{}\",",
                    "\"passed\":{},",
                    "\"detail\":\"{}\"",
                    "}}"
                ),
                if *passed { "passed" } else { "failed" },
                receipt_id,
                json_escape(label),
                passed,
                json_escape(detail),
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeEvidence {
    subject: Option<RuntimeEvidenceSubject>,
    payload: RuntimeEvidencePayload,
}

impl RuntimeEvidence {
    pub fn subject(&self) -> Option<&RuntimeEvidenceSubject> {
        self.subject.as_ref()
    }

    pub fn authority(&self) -> &'static str {
        self.payload.authority()
    }

    pub fn kind(&self) -> &'static str {
        self.payload.kind()
    }

    pub fn outcome(&self) -> &'static str {
        self.payload.outcome()
    }

    pub fn agent_fault(
        subject: Option<RuntimeEvidenceSubject>,
        code: impl Into<String>,
        class: impl Into<String>,
        operation: impl Into<String>,
        predicate: impl Into<String>,
        recoverable: bool,
        message: impl Into<String>,
    ) -> Result<Self, String> {
        let code = code.into();
        let class = class.into();
        let operation = operation.into();
        let predicate = predicate.into();
        let message = message.into();

        validate_nonempty_bounded(&code, MAX_SHORT_FIELD_BYTES, "RuntimeEvidence.agent_fault.code")?;
        validate_nonempty_bounded(
            &class,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.class",
        )?;
        validate_nonempty_bounded(
            &operation,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.operation",
        )?;
        validate_nonempty_bounded(
            &predicate,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.agent_fault.predicate",
        )?;
        validate_nonempty_bounded(
            &message,
            MAX_DETAIL_BYTES,
            "RuntimeEvidence.agent_fault.message",
        )?;

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::AgentFault {
                code,
                class,
                operation,
                predicate,
                recoverable,
                message,
            },
        })
    }

    pub fn graph_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err("RuntimeEvidence.graph_verifier_receipt: receipt id must be > 0".to_string());
        }
        let label = label.into();
        let program_identity = program_identity.into();
        let detail = detail.into();
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.graph_verifier_receipt.label",
        )?;
        validate_nonempty_bounded(
            &program_identity,
            MAX_PROGRAM_IDENTITY_BYTES,
            "RuntimeEvidence.graph_verifier_receipt.program_identity",
        )?;
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "RuntimeEvidence.graph_verifier_receipt.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::GraphVerifierReceipt {
                receipt_id,
                label,
                program_identity,
                passed,
                detail,
            },
        })
    }

    pub fn vector_verifier_receipt(
        subject: Option<RuntimeEvidenceSubject>,
        receipt_id: u32,
        label: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        if receipt_id == 0 {
            return Err("RuntimeEvidence.vector_verifier_receipt: receipt id must be > 0".to_string());
        }
        let label = label.into();
        let detail = detail.into();
        validate_nonempty_bounded(
            &label,
            MAX_SHORT_FIELD_BYTES,
            "RuntimeEvidence.vector_verifier_receipt.label",
        )?;
        if detail.len() > MAX_DETAIL_BYTES {
            return Err(format!(
                "RuntimeEvidence.vector_verifier_receipt.detail: {} bytes exceeds limit {MAX_DETAIL_BYTES}",
                detail.len()
            ));
        }

        Ok(Self {
            subject,
            payload: RuntimeEvidencePayload::VectorVerifierReceipt {
                receipt_id,
                label,
                passed,
                detail,
            },
        })
    }

    pub fn bound_agent_fault(
        projection: &RuntimeSubjectProjection,
        code: impl Into<String>,
        class: impl Into<String>,
        operation: impl Into<String>,
        predicate: impl Into<String>,
        recoverable: bool,
        message: impl Into<String>,
    ) -> Result<Self, String> {
        Self::agent_fault(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            code,
            class,
            operation,
            predicate,
            recoverable,
            message,
        )
    }

    pub fn bound_graph_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        program_identity: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::graph_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            program_identity,
            passed,
            detail,
        )
    }

    pub fn bound_vector_verifier_receipt(
        projection: &RuntimeSubjectProjection,
        receipt_id: u32,
        label: impl Into<String>,
        passed: bool,
        detail: impl Into<String>,
    ) -> Result<Self, String> {
        Self::vector_verifier_receipt(
            Some(RuntimeEvidenceSubject::from_projection(projection)),
            receipt_id,
            label,
            passed,
            detail,
        )
    }

    pub fn to_json(&self) -> String {
        let subject = self
            .subject
            .as_ref()
            .map(RuntimeEvidenceSubject::to_json)
            .unwrap_or_else(|| "{\"status\":\"unbound\"}".to_string());
        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.runtime-evidence.v1\",",
                "\"subject\":{},",
                "\"payload\":{}",
                "}}"
            ),
            subject,
            self.payload.to_json(),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejoinStatus {
    Exact,
    Unbound,
    ForeignIntent,
    StaleRuntimeRevision,
    FutureRuntimeRevision,
    ApprovalMismatch,
    SubjectMismatch,
    AuthorizationMismatch,
}

impl RejoinStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::Unbound => "unbound",
            Self::ForeignIntent => "foreign_intent",
            Self::StaleRuntimeRevision => "stale_runtime_revision",
            Self::FutureRuntimeRevision => "future_runtime_revision",
            Self::ApprovalMismatch => "approval_mismatch",
            Self::SubjectMismatch => "subject_mismatch",
            Self::AuthorizationMismatch => "authorization_mismatch",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionEvidenceInbox {
    target: RuntimeEvidenceSubject,
    entries: Vec<RuntimeEvidence>,
}

impl ResolutionEvidenceInbox {
    pub fn new(
        resolution: &ResolutionSnapshot,
        projection: &RuntimeSubjectProjection,
    ) -> Result<Self, String> {
        if !resolution.compile_eligible() {
            return Err(
                "ResolutionEvidenceInbox: ResolutionSnapshot must be compile-eligible".to_string(),
            );
        }
        if resolution.intent_id != projection.intent_id {
            return Err(format!(
                "ResolutionEvidenceInbox: resolution intent {} does not match runtime intent {}",
                resolution.intent_id, projection.intent_id
            ));
        }
        if resolution.revision != projection.workflow_revision {
            return Err(format!(
                "ResolutionEvidenceInbox: resolution revision {} does not match runtime revision {}",
                resolution.revision, projection.workflow_revision
            ));
        }

        Ok(Self {
            target: RuntimeEvidenceSubject::from_projection(projection),
            entries: Vec::new(),
        })
    }

    pub fn classify(&self, evidence: &RuntimeEvidence) -> RejoinStatus {
        let Some(subject) = &evidence.subject else {
            return RejoinStatus::Unbound;
        };
        if subject.intent_id != self.target.intent_id {
            return RejoinStatus::ForeignIntent;
        }
        if subject.workflow_revision < self.target.workflow_revision {
            return RejoinStatus::StaleRuntimeRevision;
        }
        if subject.workflow_revision > self.target.workflow_revision {
            return RejoinStatus::FutureRuntimeRevision;
        }
        if subject.approval_id != self.target.approval_id {
            return RejoinStatus::ApprovalMismatch;
        }
        if subject.subject_kind != self.target.subject_kind
            || subject.subject_identity != self.target.subject_identity
        {
            return RejoinStatus::SubjectMismatch;
        }
        if subject.authorization_policy_id != self.target.authorization_policy_id
            || subject.authorization_policy_revision != self.target.authorization_policy_revision
            || subject.authorization_is_revision != self.target.authorization_is_revision
        {
            return RejoinStatus::AuthorizationMismatch;
        }
        RejoinStatus::Exact
    }

    pub fn record(&mut self, evidence: RuntimeEvidence) -> Result<bool, String> {
        let status = self.classify(&evidence);
        if status != RejoinStatus::Exact {
            return Err(format!(
                "ResolutionEvidenceInbox: runtime evidence rejoin status {} is not admissible",
                status.as_str()
            ));
        }
        if self.entries.iter().any(|existing| existing == &evidence) {
            return Ok(false);
        }
        if self.entries.len() >= MAX_RUNTIME_EVIDENCE_ENTRIES {
            return Err(format!(
                "ResolutionEvidenceInbox: evidence limit {MAX_RUNTIME_EVIDENCE_ENTRIES} reached"
            ));
        }
        self.entries.push(evidence);
        Ok(true)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn observations(&self) -> &[RuntimeEvidence] {
        &self.entries
    }

    pub fn to_json(&self) -> String {
        let mut fault_count = 0usize;
        let mut graph_passed = 0usize;
        let mut graph_failed = 0usize;
        let mut vector_passed = 0usize;
        let mut vector_failed = 0usize;

        for evidence in &self.entries {
            match &evidence.payload {
                RuntimeEvidencePayload::AgentFault { .. } => fault_count += 1,
                RuntimeEvidencePayload::GraphVerifierReceipt { passed, .. } => {
                    if *passed {
                        graph_passed += 1;
                    } else {
                        graph_failed += 1;
                    }
                }
                RuntimeEvidencePayload::VectorVerifierReceipt { passed, .. } => {
                    if *passed {
                        vector_passed += 1;
                    } else {
                        vector_failed += 1;
                    }
                }
            }
        }

        let entries = self
            .entries
            .iter()
            .map(RuntimeEvidence::to_json)
            .collect::<Vec<_>>()
            .join(",");

        format!(
            concat!(
                "{{",
                "\"schema_version\":1,",
                "\"schema_id\":\"burn-research.resolution-evidence-inbox.v1\",",
                "\"target\":{},",
                "\"entry_count\":{},",
                "\"max_entries\":{},",
                "\"summary\":{{",
                    "\"agent_faults\":{},",
                    "\"graph_verifier_passed\":{},",
                    "\"graph_verifier_failed\":{},",
                    "\"vector_verifier_passed\":{},",
                    "\"vector_verifier_failed\":{}",
                "}},",
                "\"resolution_effect\":{{",
                    "\"diagnostic_created\":false,",
                    "\"state_transition\":\"none\",",
                    "\"revision_created\":false,",
                    "\"action_selected\":false,",
                    "\"interpretation_required\":true",
                "}},",
                "\"entries\":[{}]",
                "}}"
            ),
            self.target.to_json(),
            self.entries.len(),
            MAX_RUNTIME_EVIDENCE_ENTRIES,
            fault_count,
            graph_passed,
            graph_failed,
            vector_passed,
            vector_failed,
            entries,
        )
    }
}

#[wasm_bindgen(js_name = runtimeResolutionEvidenceCapabilities)]
pub fn runtime_resolution_evidence_capabilities() -> String {
    RUNTIME_RESOLUTION_EVIDENCE_V1.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        RejoinStatus, ResolutionEvidenceInbox, RuntimeEvidence, RuntimeEvidenceSubject,
        MAX_RUNTIME_EVIDENCE_ENTRIES,
    };
    use crate::resolution::ResolutionWorkflow;
    use crate::resolution_runtime_bridge::RuntimeSubjectProjection;

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
            authorization_schema: "burn-research.authorization-snapshot.v1".to_string(),
            authorization_policy_id: "policy".to_string(),
            authorization_policy_revision: 3,
            authorization_is_revision: false,
            approver: "owner".to_string(),
            fields: Vec::new(),
        }
    }

    #[test]
    fn exact_fault_rejoins_without_mutating_resolution_state() {
        let snapshot = resolved_snapshot("intent-a");
        let before = snapshot.clone();
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let fault = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_LAYOUT_PREFLIGHT",
            "semantic_precondition",
            "workspaceInitUnary",
            "layout.compatible",
            true,
            "known incompatible layout",
        )
        .unwrap();

        assert_eq!(inbox.classify(&fault), RejoinStatus::Exact);
        assert!(inbox.record(fault).unwrap());
        assert_eq!(inbox.len(), 1);
        assert_eq!(snapshot, before);

        let json = inbox.to_json();
        assert!(json.contains("\"agent_faults\":1"));
        assert!(json.contains("\"diagnostic_created\":false"));
        assert!(json.contains("\"state_transition\":\"none\""));
    }

    #[test]
    fn failed_verifier_is_evidence_not_automatic_resolution_failure() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let evidence = RuntimeEvidence::bound_graph_verifier_receipt(
            &projection,
            1,
            "graph-check",
            "{\"schema\":\"burn-research.program-identity.v1\",\"plan\":\"abc\"}",
            false,
            "candidate mismatch",
        )
        .unwrap();

        assert!(inbox.record(evidence).unwrap());
        let json = inbox.to_json();
        assert!(json.contains("\"graph_verifier_failed\":1"));
        assert!(json.contains("\"interpretation_required\":true"));
        assert!(snapshot.compile_eligible());
    }

    #[test]
    fn stale_foreign_and_unbound_evidence_are_classified_and_rejected() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let mut stale_subject = RuntimeEvidenceSubject::from_projection(&projection);
        stale_subject.workflow_revision -= 1;
        let stale = RuntimeEvidence::agent_fault(
            Some(stale_subject),
            "E_STALE",
            "control_precondition",
            "op",
            "revision.match",
            true,
            "stale",
        )
        .unwrap();
        assert_eq!(inbox.classify(&stale), RejoinStatus::StaleRuntimeRevision);
        assert!(inbox.record(stale).is_err());

        let mut foreign_subject = RuntimeEvidenceSubject::from_projection(&projection);
        foreign_subject.intent_id = "intent-b".to_string();
        let foreign = RuntimeEvidence::agent_fault(
            Some(foreign_subject),
            "E_FOREIGN",
            "control_precondition",
            "op",
            "intent.match",
            true,
            "foreign",
        )
        .unwrap();
        assert_eq!(inbox.classify(&foreign), RejoinStatus::ForeignIntent);
        assert!(inbox.record(foreign).is_err());

        let unbound = RuntimeEvidence::agent_fault(
            None,
            "E_UNBOUND",
            "control_precondition",
            "op",
            "subject.bound",
            true,
            "unbound",
        )
        .unwrap();
        assert_eq!(inbox.classify(&unbound), RejoinStatus::Unbound);
        assert!(inbox.record(unbound).is_err());
        assert!(inbox.is_empty());
    }

    #[test]
    fn approval_subject_and_authorization_mismatch_remain_distinct() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let mut approval_subject = RuntimeEvidenceSubject::from_projection(&projection);
        approval_subject.approval_id = "approval-other".to_string();
        let approval = RuntimeEvidence::agent_fault(
            Some(approval_subject),
            "E_APPROVAL",
            "control_precondition",
            "op",
            "approval.match",
            true,
            "approval mismatch",
        )
        .unwrap();
        assert_eq!(inbox.classify(&approval), RejoinStatus::ApprovalMismatch);

        let mut semantic_subject = RuntimeEvidenceSubject::from_projection(&projection);
        semantic_subject.subject_identity = "spec-other".to_string();
        let subject = RuntimeEvidence::agent_fault(
            Some(semantic_subject),
            "E_SUBJECT",
            "control_precondition",
            "op",
            "subject.match",
            true,
            "subject mismatch",
        )
        .unwrap();
        assert_eq!(inbox.classify(&subject), RejoinStatus::SubjectMismatch);

        let mut authorization_subject = RuntimeEvidenceSubject::from_projection(&projection);
        authorization_subject.authorization_policy_revision = 4;
        let authorization = RuntimeEvidence::agent_fault(
            Some(authorization_subject),
            "E_AUTH",
            "control_precondition",
            "op",
            "authorization.match",
            true,
            "authorization mismatch",
        )
        .unwrap();
        assert_eq!(
            inbox.classify(&authorization),
            RejoinStatus::AuthorizationMismatch
        );
    }

    #[test]
    fn inbox_requires_compile_eligible_exact_resolution_revision() {
        let mut workflow = ResolutionWorkflow::new("intent-a").unwrap();
        workflow.submit().unwrap();
        let submitted = workflow.snapshot();
        let projection = projection("intent-a", submitted.revision, "spec-a");
        assert!(ResolutionEvidenceInbox::new(&submitted, &projection).is_err());

        workflow.finalize_resolution().unwrap();
        let resolved = workflow.snapshot();
        let stale_projection = projection("intent-a", resolved.revision - 1, "spec-a");
        assert!(ResolutionEvidenceInbox::new(&resolved, &stale_projection).is_err());
    }

    #[test]
    fn duplicate_is_idempotent_and_inbox_is_bounded() {
        let snapshot = resolved_snapshot("intent-a");
        let projection = projection("intent-a", snapshot.revision, "spec-a");
        let mut inbox = ResolutionEvidenceInbox::new(&snapshot, &projection).unwrap();

        let first = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_0",
            "control_precondition",
            "op",
            "p",
            true,
            "detail",
        )
        .unwrap();
        assert!(inbox.record(first.clone()).unwrap());
        assert!(!inbox.record(first).unwrap());

        for index in 1..MAX_RUNTIME_EVIDENCE_ENTRIES {
            let evidence = RuntimeEvidence::bound_agent_fault(
                &projection,
                format!("E_{index}"),
                "control_precondition",
                "op",
                "p",
                true,
                format!("detail-{index}"),
            )
            .unwrap();
            assert!(inbox.record(evidence).unwrap());
        }
        assert_eq!(inbox.len(), MAX_RUNTIME_EVIDENCE_ENTRIES);

        let overflow = RuntimeEvidence::bound_agent_fault(
            &projection,
            "E_OVERFLOW",
            "control_precondition",
            "op",
            "p",
            true,
            "overflow",
        )
        .unwrap();
        assert!(inbox.record(overflow).is_err());
        assert_eq!(inbox.len(), MAX_RUNTIME_EVIDENCE_ENTRIES);
    }
}
