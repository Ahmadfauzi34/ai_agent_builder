//! GitHub-like review/provenance protocol layered above Resolution Workflow v1.
//!
//! This module deliberately does not execute math, construct MathProgram, or
//! perform proof. `ResolutionWorkflow` remains the compiler-style state truth;
//! this wrapper records who performed each accepted transition and captures an
//! immutable approval snapshot once the workflow is explicitly finalized.
//!
//! Actor strings are provenance labels only in v2. Authorization policy is
//! intentionally out of scope and must be enforced by a higher-level caller.

use crate::resolution::{
    DiagnosticStatus, ResolutionDiagnostic, ResolutionResponse, ResolutionSnapshot,
    ResolutionState, ResolutionWorkflow,
};

pub const APPROVAL_SNAPSHOT_SCHEMA: &str = "burn-research.resolution-approval.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReviewEventKind {
    Submitted,
    DiagnosticRequested { code: String },
    ResponseAccepted { code: String, value: String },
    Approved { approval_id: String },
    Contradiction { reason: String },
    Unsupported { reason: String },
    Invalid { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReviewEvent {
    pub sequence: u64,
    pub actor: String,
    pub workflow_revision: u64,
    pub kind: ReviewEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalDecision {
    pub diagnostic_code: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalSnapshot {
    pub schema: String,
    pub approval_id: String,
    pub intent_id: String,
    pub workflow_revision: u64,
    pub approver: String,
    pub decisions: Vec<ApprovalDecision>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionReviewSnapshot {
    pub workflow: ResolutionSnapshot,
    pub events: Vec<ReviewEvent>,
    pub approval: Option<ApprovalSnapshot>,
}

impl ResolutionReviewSnapshot {
    pub fn compile_eligible(&self) -> bool {
        self.workflow.compile_eligible() && self.approval.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionReviewSession {
    workflow: ResolutionWorkflow,
    events: Vec<ReviewEvent>,
    approval: Option<ApprovalSnapshot>,
}

impl ResolutionReviewSession {
    pub fn new(intent_id: impl Into<String>) -> Result<Self, String> {
        Ok(Self {
            workflow: ResolutionWorkflow::new(intent_id)?,
            events: Vec::new(),
            approval: None,
        })
    }

    pub fn snapshot(&self) -> ResolutionReviewSnapshot {
        ResolutionReviewSnapshot {
            workflow: self.workflow.snapshot(),
            events: self.events.clone(),
            approval: self.approval.clone(),
        }
    }

    pub fn workflow_state(&self) -> ResolutionState {
        self.workflow.state()
    }

    pub fn workflow_revision(&self) -> u64 {
        self.workflow.revision()
    }

    pub fn events(&self) -> &[ReviewEvent] {
        &self.events
    }

    pub fn approval_snapshot(&self) -> Option<&ApprovalSnapshot> {
        self.approval.as_ref()
    }

    pub fn compile_eligible(&self) -> bool {
        self.workflow.compile_eligible() && self.approval.is_some()
    }

    pub fn submit(&mut self, actor: impl Into<String>) -> Result<(), String> {
        let actor = validate_actor(actor)?;
        let sequence = self.next_sequence()?;
        self.workflow.submit()?;
        self.record(sequence, actor, ReviewEventKind::Submitted);
        Ok(())
    }

    pub fn request_diagnostic(
        &mut self,
        actor: impl Into<String>,
        diagnostic: ResolutionDiagnostic,
    ) -> Result<(), String> {
        let actor = validate_actor(actor)?;
        let sequence = self.next_sequence()?;
        let code = diagnostic.code.clone();
        self.workflow.add_diagnostic(diagnostic)?;
        self.record(
            sequence,
            actor,
            ReviewEventKind::DiagnosticRequested { code },
        );
        Ok(())
    }

    pub fn respond(
        &mut self,
        actor: impl Into<String>,
        response: ResolutionResponse,
    ) -> Result<(), String> {
        let actor = validate_actor(actor)?;
        let sequence = self.next_sequence()?;
        let code = response.diagnostic_code.clone();
        let value = response.value.clone();
        self.workflow.apply_response(response)?;
        self.record(
            sequence,
            actor,
            ReviewEventKind::ResponseAccepted { code, value },
        );
        Ok(())
    }

    pub fn approve(
        &mut self,
        actor: impl Into<String>,
    ) -> Result<ApprovalSnapshot, String> {
        let actor = validate_actor(actor)?;
        if self.approval.is_some() {
            return Err("ResolutionReviewSession: approval already exists".to_string());
        }
        let sequence = self.next_sequence()?;

        // v1 remains the authority for unresolved-blocker and state checks.
        self.workflow.finalize_resolution()?;
        let workflow = self.workflow.snapshot();
        if !workflow.compile_eligible() {
            return Err(
                "ResolutionReviewSession: finalized workflow is not compile eligible".to_string(),
            );
        }

        let decisions = workflow
            .diagnostics
            .iter()
            .filter_map(|diagnostic| {
                if diagnostic.status != DiagnosticStatus::Resolved {
                    return None;
                }
                diagnostic.resolution.as_ref().map(|value| ApprovalDecision {
                    diagnostic_code: diagnostic.code.clone(),
                    value: value.clone(),
                })
            })
            .collect::<Vec<_>>();

        let approval_id = format!(
            "{}:approval:r{}",
            workflow.intent_id, workflow.revision
        );
        let approval = ApprovalSnapshot {
            schema: APPROVAL_SNAPSHOT_SCHEMA.to_string(),
            approval_id: approval_id.clone(),
            intent_id: workflow.intent_id.clone(),
            workflow_revision: workflow.revision,
            approver: actor.clone(),
            decisions,
        };

        self.approval = Some(approval.clone());
        self.record(
            sequence,
            actor,
            ReviewEventKind::Approved { approval_id },
        );
        Ok(approval)
    }

    pub fn mark_contradiction(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.mark_terminal(actor, reason, TerminalKind::Contradiction)
    }

    pub fn mark_unsupported(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.mark_terminal(actor, reason, TerminalKind::Unsupported)
    }

    pub fn mark_invalid(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.mark_terminal(actor, reason, TerminalKind::Invalid)
    }

    fn mark_terminal(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
        kind: TerminalKind,
    ) -> Result<(), String> {
        let actor = validate_actor(actor)?;
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err("ResolutionReviewSession: terminal reason must not be empty".to_string());
        }
        let sequence = self.next_sequence()?;

        let event = match kind {
            TerminalKind::Contradiction => {
                self.workflow.mark_contradiction(reason.clone())?;
                ReviewEventKind::Contradiction { reason }
            }
            TerminalKind::Unsupported => {
                self.workflow.mark_unsupported(reason.clone())?;
                ReviewEventKind::Unsupported { reason }
            }
            TerminalKind::Invalid => {
                self.workflow.mark_invalid(reason.clone())?;
                ReviewEventKind::Invalid { reason }
            }
        };
        self.record(sequence, actor, event);
        Ok(())
    }

    fn next_sequence(&self) -> Result<u64, String> {
        let len = u64::try_from(self.events.len())
            .map_err(|_| "ResolutionReviewSession: event count exceeds u64".to_string())?;
        len.checked_add(1)
            .ok_or_else(|| "ResolutionReviewSession: event sequence overflow".to_string())
    }

    fn record(&mut self, sequence: u64, actor: String, kind: ReviewEventKind) {
        self.events.push(ReviewEvent {
            sequence,
            actor,
            workflow_revision: self.workflow.revision(),
            kind,
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TerminalKind {
    Contradiction,
    Unsupported,
    Invalid,
}

pub(crate) fn validate_actor(actor: impl Into<String>) -> Result<String, String> {
    let actor = actor.into();
    if actor.trim().is_empty() {
        return Err("ResolutionReviewSession: actor must not be empty".to_string());
    }
    Ok(actor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{DiagnosticKind, DiagnosticSeverity};

    fn normalization_diagnostic() -> ResolutionDiagnostic {
        ResolutionDiagnostic::blocking(
            "E-MATH-001",
            DiagnosticKind::MultipleInterpretations,
            "normalization",
            "multiple valid normalization semantics exist",
            "choose one normalization semantic",
            vec!["l2".to_string(), "probability".to_string()],
        )
        .unwrap()
    }

    #[test]
    fn request_response_approval_chain_is_fully_auditable() {
        let mut review = ResolutionReviewSession::new("intent-42").unwrap();
        review.submit("agent").unwrap();
        review
            .request_diagnostic("validator", normalization_diagnostic())
            .unwrap();
        review
            .respond(
                "customer",
                ResolutionResponse::new("E-MATH-001", "l2").unwrap(),
            )
            .unwrap();
        let approval = review.approve("customer").unwrap();

        assert!(review.compile_eligible());
        assert_eq!(approval.schema, APPROVAL_SNAPSHOT_SCHEMA);
        assert_eq!(approval.approval_id, "intent-42:approval:r4");
        assert_eq!(approval.workflow_revision, 4);
        assert_eq!(approval.approver, "customer");
        assert_eq!(
            approval.decisions,
            vec![ApprovalDecision {
                diagnostic_code: "E-MATH-001".to_string(),
                value: "l2".to_string(),
            }]
        );

        let events = review.events();
        assert_eq!(events.len(), 4);
        assert_eq!(
            events.iter().map(|event| event.sequence).collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            events
                .iter()
                .map(|event| event.workflow_revision)
                .collect::<Vec<_>>(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(events[0].actor, "agent");
        assert_eq!(events[1].actor, "validator");
        assert_eq!(events[2].actor, "customer");
        assert_eq!(events[3].actor, "customer");
    }

    #[test]
    fn approval_is_blocked_by_unresolved_diagnostic_without_mutation() {
        let mut review = ResolutionReviewSession::new("intent-blocked").unwrap();
        review.submit("agent").unwrap();
        review
            .request_diagnostic("validator", normalization_diagnostic())
            .unwrap();
        let before = review.snapshot();

        let err = review.approve("customer").unwrap_err();
        assert!(err.contains("unresolved blocking diagnostics"));
        assert_eq!(review.snapshot(), before);
    }

    #[test]
    fn invalid_actor_and_invalid_response_are_atomic() {
        let mut review = ResolutionReviewSession::new("intent-atomic").unwrap();
        let before_submit = review.snapshot();
        assert!(review.submit("  ").is_err());
        assert_eq!(review.snapshot(), before_submit);

        review.submit("agent").unwrap();
        review
            .request_diagnostic("validator", normalization_diagnostic())
            .unwrap();
        let before_response = review.snapshot();
        assert!(review
            .respond(
                "customer",
                ResolutionResponse::new("E-MATH-001", "zscore").unwrap(),
            )
            .is_err());
        assert_eq!(review.snapshot(), before_response);
    }

    #[test]
    fn approval_snapshot_contains_only_accepted_resolutions() {
        let mut review = ResolutionReviewSession::new("intent-decisions").unwrap();
        review.submit("agent").unwrap();
        review
            .request_diagnostic("validator", normalization_diagnostic())
            .unwrap();
        review
            .request_diagnostic(
                "validator",
                ResolutionDiagnostic::new(
                    "W-MATH-001",
                    DiagnosticKind::MissingShapeAssumption,
                    DiagnosticSeverity::Warning,
                    "shape",
                    "runtime shape is not yet known",
                    "provide shape later if runtime validation requires it",
                    Vec::new(),
                )
                .unwrap(),
            )
            .unwrap();
        review
            .respond(
                "customer",
                ResolutionResponse::new("E-MATH-001", "probability").unwrap(),
            )
            .unwrap();

        let approval = review.approve("customer").unwrap();
        assert_eq!(approval.decisions.len(), 1);
        assert_eq!(approval.decisions[0].diagnostic_code, "E-MATH-001");
        assert_eq!(approval.decisions[0].value, "probability");
    }

    #[test]
    fn approval_is_deterministic_for_same_intent_and_transitions() {
        fn build() -> ApprovalSnapshot {
            let mut review = ResolutionReviewSession::new("intent-deterministic").unwrap();
            review.submit("agent").unwrap();
            review
                .request_diagnostic("validator", normalization_diagnostic())
                .unwrap();
            review
                .respond(
                    "customer",
                    ResolutionResponse::new("E-MATH-001", "l2").unwrap(),
                )
                .unwrap();
            review.approve("customer").unwrap()
        }

        assert_eq!(build(), build());
    }

    #[test]
    fn resolved_session_rejects_duplicate_approval_and_further_review_mutation() {
        let mut review = ResolutionReviewSession::new("intent-closed").unwrap();
        review.submit("agent").unwrap();
        review.approve("customer").unwrap();
        let before = review.snapshot();

        assert!(review.approve("customer").is_err());
        assert!(review
            .request_diagnostic("validator", normalization_diagnostic())
            .is_err());
        assert_eq!(review.snapshot(), before);
    }

    #[test]
    fn terminal_outcomes_are_auditable_and_never_approvable() {
        let cases = [
            (ResolutionState::Contradiction, "constraints conflict"),
            (ResolutionState::Unsupported, "capability absent"),
            (ResolutionState::Invalid, "intent malformed"),
        ];

        for (expected, reason) in cases {
            let mut review = ResolutionReviewSession::new(format!("intent-{expected:?}")).unwrap();
            review.submit("agent").unwrap();
            match expected {
                ResolutionState::Contradiction => review
                    .mark_contradiction("validator", reason)
                    .unwrap(),
                ResolutionState::Unsupported => {
                    review.mark_unsupported("validator", reason).unwrap()
                }
                ResolutionState::Invalid => review.mark_invalid("validator", reason).unwrap(),
                _ => unreachable!(),
            }

            assert_eq!(review.workflow_state(), expected);
            assert!(!review.compile_eligible());
            assert!(review.approve("customer").is_err());
            assert!(review.approval_snapshot().is_none());
            assert_eq!(review.events().len(), 2);
            assert_eq!(review.events()[1].actor, "validator");
        }
    }
}
