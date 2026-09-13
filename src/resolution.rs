//! Compiler-style intent resolution workflow.
//!
//! This module deliberately lives above Math Program. It does not execute math,
//! build `MathProgram`, or decide semantic intent by guessing. Its only job is
//! to make unresolved intent explicit, track deterministic review-like
//! diagnostics, accept explicit responses, and report whether an intent is
//! eligible to cross into a later compilation/proof boundary.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolutionState {
    Draft,
    Submitted,
    NeedsResolution,
    Resolved,
    Contradiction,
    Unsupported,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticKind {
    MissingParameter,
    MultipleInterpretations,
    MissingShapeAssumption,
    MissingDomainAssumption,
    ConflictingConstraints,
    UnsupportedCapability,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Note,
    Warning,
    Blocking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticStatus {
    Open,
    Resolved,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionDiagnostic {
    pub code: String,
    pub kind: DiagnosticKind,
    pub severity: DiagnosticSeverity,
    pub subject: String,
    pub reason: String,
    pub required_information: String,
    pub candidates: Vec<String>,
    pub status: DiagnosticStatus,
    pub resolution: Option<String>,
}

impl ResolutionDiagnostic {
    pub fn blocking(
        code: impl Into<String>,
        kind: DiagnosticKind,
        subject: impl Into<String>,
        reason: impl Into<String>,
        required_information: impl Into<String>,
        candidates: Vec<String>,
    ) -> Result<Self, String> {
        Self::new(
            code,
            kind,
            DiagnosticSeverity::Blocking,
            subject,
            reason,
            required_information,
            candidates,
        )
    }

    pub fn new(
        code: impl Into<String>,
        kind: DiagnosticKind,
        severity: DiagnosticSeverity,
        subject: impl Into<String>,
        reason: impl Into<String>,
        required_information: impl Into<String>,
        candidates: Vec<String>,
    ) -> Result<Self, String> {
        let code = code.into();
        let subject = subject.into();
        let reason = reason.into();
        let required_information = required_information.into();

        if code.trim().is_empty() {
            return Err("ResolutionDiagnostic: code must not be empty".to_string());
        }
        if subject.trim().is_empty() {
            return Err(format!(
                "ResolutionDiagnostic {code}: subject must not be empty"
            ));
        }
        if reason.trim().is_empty() {
            return Err(format!(
                "ResolutionDiagnostic {code}: reason must not be empty"
            ));
        }
        if required_information.trim().is_empty() {
            return Err(format!(
                "ResolutionDiagnostic {code}: required information must not be empty"
            ));
        }
        if candidates.iter().any(|value| value.trim().is_empty()) {
            return Err(format!(
                "ResolutionDiagnostic {code}: candidates must not contain empty values"
            ));
        }

        Ok(Self {
            code,
            kind,
            severity,
            subject,
            reason,
            required_information,
            candidates,
            status: DiagnosticStatus::Open,
            resolution: None,
        })
    }

    pub fn is_open_blocker(&self) -> bool {
        self.severity == DiagnosticSeverity::Blocking && self.status == DiagnosticStatus::Open
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionResponse {
    pub diagnostic_code: String,
    pub value: String,
}

impl ResolutionResponse {
    pub fn new(
        diagnostic_code: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self, String> {
        let diagnostic_code = diagnostic_code.into();
        let value = value.into();
        if diagnostic_code.trim().is_empty() {
            return Err("ResolutionResponse: diagnostic code must not be empty".to_string());
        }
        if value.trim().is_empty() {
            return Err("ResolutionResponse: value must not be empty".to_string());
        }
        Ok(Self {
            diagnostic_code,
            value,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionSnapshot {
    pub intent_id: String,
    pub revision: u64,
    pub state: ResolutionState,
    pub diagnostics: Vec<ResolutionDiagnostic>,
}

impl ResolutionSnapshot {
    pub fn compile_eligible(&self) -> bool {
        self.state == ResolutionState::Resolved
            && !self
                .diagnostics
                .iter()
                .any(ResolutionDiagnostic::is_open_blocker)
    }

    pub fn open_blockers(&self) -> Vec<&ResolutionDiagnostic> {
        self.diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_open_blocker())
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionWorkflow {
    intent_id: String,
    revision: u64,
    state: ResolutionState,
    diagnostics: Vec<ResolutionDiagnostic>,
}

impl ResolutionWorkflow {
    pub fn new(intent_id: impl Into<String>) -> Result<Self, String> {
        let intent_id = intent_id.into();
        if intent_id.trim().is_empty() {
            return Err("ResolutionWorkflow: intent id must not be empty".to_string());
        }
        Ok(Self {
            intent_id,
            revision: 0,
            state: ResolutionState::Draft,
            diagnostics: Vec::new(),
        })
    }

    pub fn snapshot(&self) -> ResolutionSnapshot {
        ResolutionSnapshot {
            intent_id: self.intent_id.clone(),
            revision: self.revision,
            state: self.state,
            diagnostics: self.diagnostics.clone(),
        }
    }

    pub fn state(&self) -> ResolutionState {
        self.state
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn submit(&mut self) -> Result<(), String> {
        self.require_state(&[ResolutionState::Draft], "submit")?;
        self.state = ResolutionState::Submitted;
        self.advance_revision("submit")
    }

    pub fn add_diagnostic(&mut self, diagnostic: ResolutionDiagnostic) -> Result<(), String> {
        self.require_state(
            &[ResolutionState::Submitted, ResolutionState::NeedsResolution],
            "add diagnostic",
        )?;
        if self
            .diagnostics
            .iter()
            .any(|existing| existing.code == diagnostic.code)
        {
            return Err(format!(
                "ResolutionWorkflow: duplicate diagnostic code {}",
                diagnostic.code
            ));
        }
        let blocking = diagnostic.severity == DiagnosticSeverity::Blocking;
        self.diagnostics.push(diagnostic);
        if blocking {
            self.state = ResolutionState::NeedsResolution;
        }
        self.advance_revision("add diagnostic")
    }

    pub fn apply_response(&mut self, response: ResolutionResponse) -> Result<(), String> {
        self.require_state(
            &[ResolutionState::Submitted, ResolutionState::NeedsResolution],
            "apply resolution response",
        )?;

        let diagnostic = self
            .diagnostics
            .iter_mut()
            .find(|diagnostic| diagnostic.code == response.diagnostic_code)
            .ok_or_else(|| {
                format!(
                    "ResolutionWorkflow: unknown diagnostic code {}",
                    response.diagnostic_code
                )
            })?;

        if diagnostic.status == DiagnosticStatus::Resolved {
            return Err(format!(
                "ResolutionWorkflow: diagnostic {} is already resolved",
                diagnostic.code
            ));
        }

        if !diagnostic.candidates.is_empty()
            && !diagnostic
                .candidates
                .iter()
                .any(|candidate| candidate == &response.value)
        {
            return Err(format!(
                "ResolutionWorkflow: response '{}' is not an allowed candidate for {}",
                response.value, diagnostic.code
            ));
        }

        diagnostic.status = DiagnosticStatus::Resolved;
        diagnostic.resolution = Some(response.value);
        self.advance_revision("apply resolution response")
    }

    pub fn finalize_resolution(&mut self) -> Result<(), String> {
        self.require_state(
            &[ResolutionState::Submitted, ResolutionState::NeedsResolution],
            "finalize resolution",
        )?;
        let open = self
            .diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_open_blocker())
            .map(|diagnostic| diagnostic.code.as_str())
            .collect::<Vec<_>>();
        if !open.is_empty() {
            return Err(format!(
                "ResolutionWorkflow: unresolved blocking diagnostics: {}",
                open.join(", ")
            ));
        }
        self.state = ResolutionState::Resolved;
        self.advance_revision("finalize resolution")
    }

    pub fn mark_contradiction(&mut self, reason: impl Into<String>) -> Result<(), String> {
        self.mark_terminal(ResolutionState::Contradiction, "contradiction", reason)
    }

    pub fn mark_unsupported(&mut self, reason: impl Into<String>) -> Result<(), String> {
        self.mark_terminal(ResolutionState::Unsupported, "unsupported", reason)
    }

    pub fn mark_invalid(&mut self, reason: impl Into<String>) -> Result<(), String> {
        self.mark_terminal(ResolutionState::Invalid, "invalid", reason)
    }

    pub fn compile_eligible(&self) -> bool {
        self.snapshot().compile_eligible()
    }

    fn mark_terminal(
        &mut self,
        state: ResolutionState,
        label: &str,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.require_state(
            &[ResolutionState::Submitted, ResolutionState::NeedsResolution],
            label,
        )?;
        let reason = reason.into();
        if reason.trim().is_empty() {
            return Err(format!("ResolutionWorkflow: {label} reason must not be empty"));
        }
        self.state = state;
        self.advance_revision(label)
    }

    fn require_state(&self, allowed: &[ResolutionState], action: &str) -> Result<(), String> {
        if allowed.contains(&self.state) {
            return Ok(());
        }
        Err(format!(
            "ResolutionWorkflow: cannot {action} while state is {:?}",
            self.state
        ))
    }

    fn advance_revision(&mut self, action: &str) -> Result<(), String> {
        self.revision = self.revision.checked_add(1).ok_or_else(|| {
            format!("ResolutionWorkflow: revision overflow while attempting to {action}")
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn ambiguity_requires_explicit_resolution_before_compile_eligibility() {
        let mut workflow = ResolutionWorkflow::new("intent-42").unwrap();
        workflow.submit().unwrap();
        workflow.add_diagnostic(normalization_diagnostic()).unwrap();

        assert_eq!(workflow.state(), ResolutionState::NeedsResolution);
        assert!(!workflow.compile_eligible());
        assert!(workflow.finalize_resolution().is_err());

        workflow
            .apply_response(ResolutionResponse::new("E-MATH-001", "l2").unwrap())
            .unwrap();
        assert!(!workflow.compile_eligible());

        workflow.finalize_resolution().unwrap();
        assert_eq!(workflow.state(), ResolutionState::Resolved);
        assert!(workflow.compile_eligible());
    }

    #[test]
    fn resolution_response_must_match_explicit_candidate_when_candidates_exist() {
        let mut workflow = ResolutionWorkflow::new("intent-1").unwrap();
        workflow.submit().unwrap();
        workflow.add_diagnostic(normalization_diagnostic()).unwrap();
        let revision = workflow.revision();

        let err = workflow
            .apply_response(ResolutionResponse::new("E-MATH-001", "zscore").unwrap())
            .unwrap_err();
        assert!(err.contains("not an allowed candidate"));
        assert_eq!(workflow.revision(), revision);
        assert_eq!(workflow.state(), ResolutionState::NeedsResolution);
    }

    #[test]
    fn failed_or_unknown_resolution_never_consumes_revision() {
        let mut workflow = ResolutionWorkflow::new("intent-2").unwrap();
        workflow.submit().unwrap();
        workflow.add_diagnostic(normalization_diagnostic()).unwrap();
        let before = workflow.snapshot();

        assert!(workflow
            .apply_response(ResolutionResponse::new("missing", "l2").unwrap())
            .is_err());
        assert_eq!(workflow.snapshot(), before);
    }

    #[test]
    fn contradiction_unsupported_and_invalid_are_distinct_terminal_states() {
        let cases = [
            (ResolutionState::Contradiction, "contradiction"),
            (ResolutionState::Unsupported, "unsupported"),
            (ResolutionState::Invalid, "invalid"),
        ];

        for (expected, label) in cases {
            let mut workflow = ResolutionWorkflow::new(format!("intent-{label}")).unwrap();
            workflow.submit().unwrap();
            match expected {
                ResolutionState::Contradiction => {
                    workflow.mark_contradiction("constraints conflict").unwrap()
                }
                ResolutionState::Unsupported => {
                    workflow.mark_unsupported("capability absent").unwrap()
                }
                ResolutionState::Invalid => workflow.mark_invalid("malformed intent").unwrap(),
                _ => unreachable!(),
            }
            assert_eq!(workflow.state(), expected);
            assert!(!workflow.compile_eligible());
            assert!(workflow.finalize_resolution().is_err());
        }
    }

    #[test]
    fn non_blocking_diagnostics_do_not_prevent_explicit_finalize() {
        let mut workflow = ResolutionWorkflow::new("intent-warning").unwrap();
        workflow.submit().unwrap();
        workflow
            .add_diagnostic(
                ResolutionDiagnostic::new(
                    "W-MATH-001",
                    DiagnosticKind::MissingShapeAssumption,
                    DiagnosticSeverity::Warning,
                    "shape",
                    "runtime shape may narrow available operations",
                    "provide a shape later if runtime validation fails",
                    Vec::new(),
                )
                .unwrap(),
            )
            .unwrap();
        workflow.finalize_resolution().unwrap();
        assert!(workflow.compile_eligible());
    }

    #[test]
    fn duplicate_diagnostic_code_rejects_without_mutation() {
        let mut workflow = ResolutionWorkflow::new("intent-duplicate").unwrap();
        workflow.submit().unwrap();
        workflow.add_diagnostic(normalization_diagnostic()).unwrap();
        let before = workflow.snapshot();

        assert!(workflow.add_diagnostic(normalization_diagnostic()).is_err());
        assert_eq!(workflow.snapshot(), before);
    }

    #[test]
    fn workflow_revisions_are_monotonic_and_auditable() {
        let mut workflow = ResolutionWorkflow::new("intent-audit").unwrap();
        assert_eq!(workflow.revision(), 0);
        workflow.submit().unwrap();
        assert_eq!(workflow.revision(), 1);
        workflow.add_diagnostic(normalization_diagnostic()).unwrap();
        assert_eq!(workflow.revision(), 2);
        workflow
            .apply_response(ResolutionResponse::new("E-MATH-001", "probability").unwrap())
            .unwrap();
        assert_eq!(workflow.revision(), 3);
        workflow.finalize_resolution().unwrap();
        assert_eq!(workflow.revision(), 4);

        let snapshot = workflow.snapshot();
        assert_eq!(snapshot.intent_id, "intent-audit");
        assert_eq!(snapshot.state, ResolutionState::Resolved);
        assert_eq!(snapshot.diagnostics[0].resolution.as_deref(), Some("probability"));
    }

    #[test]
    fn resolved_workflow_is_closed_to_further_mutation() {
        let mut workflow = ResolutionWorkflow::new("intent-closed").unwrap();
        workflow.submit().unwrap();
        workflow.finalize_resolution().unwrap();
        let before = workflow.snapshot();

        assert!(workflow.add_diagnostic(normalization_diagnostic()).is_err());
        assert!(workflow
            .apply_response(ResolutionResponse::new("E-MATH-001", "l2").unwrap())
            .is_err());
        assert_eq!(workflow.snapshot(), before);
    }
}
