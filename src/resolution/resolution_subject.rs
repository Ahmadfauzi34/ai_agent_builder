//! Immutable approval-subject binding layered above resolution review/revision protocols.
//!
//! GitHub reviews are meaningful because they apply to a particular immutable
//! commit/head. This module provides the analogous boundary for resolution:
//! approval is not only "intent resolved", but "this exact immutable subject
//! was approved". The subject is selected before approval and has no mutation
//! path through this API.
//!
//! This layer deliberately does not define the subject contents, perform proof,
//! construct MathProgram, execute code, or enforce actor authorization.

use crate::resolution::{ResolutionDiagnostic, ResolutionResponse};
use crate::resolution_review::{
    ApprovalSnapshot, ResolutionReviewSession, ResolutionReviewSnapshot,
};
use crate::resolution_revision::{
    ResolutionRevisionChain, ResolutionRevisionChainSnapshot, RevisionApprovalSnapshot,
};

pub const SUBJECT_BOUND_APPROVAL_SCHEMA: &str = "burn-research.resolution-subject-approval.v1";
pub const SUBJECT_BOUND_REVISION_SCHEMA: &str =
    "burn-research.resolution-subject-revision-approval.v1";
pub const SUBJECT_BOUND_CHAIN_SCHEMA: &str = "burn-research.resolution-subject-chain.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApprovalSubject {
    kind: String,
    identity: String,
}

impl ApprovalSubject {
    pub fn new(kind: impl Into<String>, identity: impl Into<String>) -> Result<Self, String> {
        let kind = validate_kind(kind)?;
        let identity = validate_identity(identity)?;
        Ok(Self { kind, identity })
    }

    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundApprovalSnapshot {
    pub schema: String,
    pub subject: ApprovalSubject,
    pub approval: ApprovalSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundReviewSnapshot {
    pub subject: ApprovalSubject,
    pub review: ResolutionReviewSnapshot,
    pub bound_approval: Option<SubjectBoundApprovalSnapshot>,
}

impl SubjectBoundReviewSnapshot {
    pub fn compile_eligible(&self) -> bool {
        self.review.compile_eligible() && self.bound_approval.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundReviewSession {
    subject: ApprovalSubject,
    review: ResolutionReviewSession,
    bound_approval: Option<SubjectBoundApprovalSnapshot>,
}

impl SubjectBoundReviewSession {
    pub fn new(
        intent_id: impl Into<String>,
        subject: ApprovalSubject,
    ) -> Result<Self, String> {
        Ok(Self {
            subject,
            review: ResolutionReviewSession::new(intent_id)?,
            bound_approval: None,
        })
    }

    pub fn subject(&self) -> &ApprovalSubject {
        &self.subject
    }

    pub fn snapshot(&self) -> SubjectBoundReviewSnapshot {
        SubjectBoundReviewSnapshot {
            subject: self.subject.clone(),
            review: self.review.snapshot(),
            bound_approval: self.bound_approval.clone(),
        }
    }

    pub fn compile_eligible(&self) -> bool {
        self.review.compile_eligible() && self.bound_approval.is_some()
    }

    pub fn submit(&mut self, actor: impl Into<String>) -> Result<(), String> {
        self.review.submit(actor)
    }

    pub fn request_diagnostic(
        &mut self,
        actor: impl Into<String>,
        diagnostic: ResolutionDiagnostic,
    ) -> Result<(), String> {
        self.review.request_diagnostic(actor, diagnostic)
    }

    pub fn respond(
        &mut self,
        actor: impl Into<String>,
        response: ResolutionResponse,
    ) -> Result<(), String> {
        self.review.respond(actor, response)
    }

    pub fn approve(
        &mut self,
        actor: impl Into<String>,
    ) -> Result<SubjectBoundApprovalSnapshot, String> {
        if self.bound_approval.is_some() {
            return Err("SubjectBoundReviewSession: bound approval already exists".to_string());
        }
        let approval = self.review.approve(actor)?;
        let snapshot = SubjectBoundApprovalSnapshot {
            schema: SUBJECT_BOUND_APPROVAL_SCHEMA.to_string(),
            subject: self.subject.clone(),
            approval,
        };
        self.bound_approval = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub fn mark_contradiction(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.review.mark_contradiction(actor, reason)
    }

    pub fn mark_unsupported(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.review.mark_unsupported(actor, reason)
    }

    pub fn mark_invalid(
        &mut self,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.review.mark_invalid(actor, reason)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevisionSubjectBinding {
    pub revision_id: String,
    pub subject: ApprovalSubject,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundRevisionApprovalSnapshot {
    pub schema: String,
    pub subject: ApprovalSubject,
    pub approval: RevisionApprovalSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundRevisionChainSnapshot {
    pub schema: String,
    pub root: SubjectBoundReviewSnapshot,
    pub chain: ResolutionRevisionChainSnapshot,
    pub revision_subjects: Vec<RevisionSubjectBinding>,
    pub bound_revision_approvals: Vec<SubjectBoundRevisionApprovalSnapshot>,
}

impl SubjectBoundRevisionChainSnapshot {
    pub fn revision_subject(&self, revision_id: &str) -> Option<&ApprovalSubject> {
        self.revision_subjects
            .iter()
            .find(|binding| binding.revision_id == revision_id)
            .map(|binding| &binding.subject)
    }

    pub fn bound_revision_approval(
        &self,
        revision_id: &str,
    ) -> Option<&SubjectBoundRevisionApprovalSnapshot> {
        self.bound_revision_approvals
            .iter()
            .find(|approval| approval.approval.revision_id == revision_id)
    }

    pub fn revision_compile_eligible(&self, revision_id: &str) -> bool {
        self.chain
            .revision(revision_id)
            .is_some_and(|revision| revision.compile_eligible())
            && self.bound_revision_approval(revision_id).is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubjectBoundRevisionChain {
    root: SubjectBoundReviewSnapshot,
    chain: ResolutionRevisionChain,
    revision_subjects: Vec<RevisionSubjectBinding>,
    bound_revision_approvals: Vec<SubjectBoundRevisionApprovalSnapshot>,
}

impl SubjectBoundRevisionChain {
    pub fn from_approved_root(root: SubjectBoundReviewSnapshot) -> Result<Self, String> {
        if !root.compile_eligible() {
            return Err(
                "SubjectBoundRevisionChain: root must already have a subject-bound approval"
                    .to_string(),
            );
        }
        let bound = root.bound_approval.as_ref().ok_or_else(|| {
            "SubjectBoundRevisionChain: root bound approval disappeared".to_string()
        })?;
        let review_approval = root.review.approval.as_ref().ok_or_else(|| {
            "SubjectBoundRevisionChain: root review approval disappeared".to_string()
        })?;
        if &bound.approval != review_approval {
            return Err(
                "SubjectBoundRevisionChain: bound root approval does not match review approval"
                    .to_string(),
            );
        }

        let chain = ResolutionRevisionChain::from_approved(root.review.clone())?;
        Ok(Self {
            root,
            chain,
            revision_subjects: Vec::new(),
            bound_revision_approvals: Vec::new(),
        })
    }

    pub fn root(&self) -> &SubjectBoundReviewSnapshot {
        &self.root
    }

    pub fn snapshot(&self) -> SubjectBoundRevisionChainSnapshot {
        SubjectBoundRevisionChainSnapshot {
            schema: SUBJECT_BOUND_CHAIN_SCHEMA.to_string(),
            root: self.root.clone(),
            chain: self.chain.snapshot(),
            revision_subjects: self.revision_subjects.clone(),
            bound_revision_approvals: self.bound_revision_approvals.clone(),
        }
    }

    pub fn open_revision(
        &mut self,
        parent_revision_id: Option<&str>,
        revision_key: impl Into<String>,
        subject: ApprovalSubject,
    ) -> Result<String, String> {
        if let Some(parent_revision_id) = parent_revision_id {
            if self.bound_revision_approval(parent_revision_id).is_none() {
                return Err(format!(
                    "SubjectBoundRevisionChain: parent revision lacks a subject-bound approval: {parent_revision_id}"
                ));
            }
        }

        let revision_id = self.chain.open_revision(parent_revision_id, revision_key)?;
        self.revision_subjects.push(RevisionSubjectBinding {
            revision_id: revision_id.clone(),
            subject,
        });
        Ok(revision_id)
    }

    pub fn revision_subject(&self, revision_id: &str) -> Option<&ApprovalSubject> {
        self.revision_subjects
            .iter()
            .find(|binding| binding.revision_id == revision_id)
            .map(|binding| &binding.subject)
    }

    pub fn bound_revision_approval(
        &self,
        revision_id: &str,
    ) -> Option<&SubjectBoundRevisionApprovalSnapshot> {
        self.bound_revision_approvals
            .iter()
            .find(|approval| approval.approval.revision_id == revision_id)
    }

    pub fn revision_compile_eligible(&self, revision_id: &str) -> Result<bool, String> {
        Ok(self.chain.revision_compile_eligible(revision_id)?
            && self.bound_revision_approval(revision_id).is_some())
    }

    pub fn submit_revision(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.submit_revision(revision_id, actor)
    }

    pub fn request_diagnostic(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        diagnostic: ResolutionDiagnostic,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.request_diagnostic(revision_id, actor, diagnostic)
    }

    pub fn respond(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        response: ResolutionResponse,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.respond(revision_id, actor, response)
    }

    pub fn approve_revision(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
    ) -> Result<SubjectBoundRevisionApprovalSnapshot, String> {
        if self.bound_revision_approval(revision_id).is_some() {
            return Err(format!(
                "SubjectBoundRevisionChain: bound approval already exists: {revision_id}"
            ));
        }
        let subject = self.require_subject(revision_id)?.clone();
        let approval = self.chain.approve_revision(revision_id, actor)?;
        let snapshot = SubjectBoundRevisionApprovalSnapshot {
            schema: SUBJECT_BOUND_REVISION_SCHEMA.to_string(),
            subject,
            approval,
        };
        self.bound_revision_approvals.push(snapshot.clone());
        Ok(snapshot)
    }

    pub fn mark_contradiction(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.mark_contradiction(revision_id, actor, reason)
    }

    pub fn mark_unsupported(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.mark_unsupported(revision_id, actor, reason)
    }

    pub fn mark_invalid(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.require_subject(revision_id)?;
        self.chain.mark_invalid(revision_id, actor, reason)
    }

    fn require_subject(&self, revision_id: &str) -> Result<&ApprovalSubject, String> {
        self.revision_subject(revision_id).ok_or_else(|| {
            format!(
                "SubjectBoundRevisionChain: unknown subject binding for revision: {revision_id}"
            )
        })
    }
}

fn validate_kind(kind: impl Into<String>) -> Result<String, String> {
    let kind = kind.into();
    if kind.is_empty() || kind.len() > 64 {
        return Err("ApprovalSubject: kind length must be between 1 and 64".to_string());
    }
    if !kind
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "ApprovalSubject: kind must use only ASCII letters, digits, '-', '_' or '.'"
                .to_string(),
        );
    }
    Ok(kind)
}

fn validate_identity(identity: impl Into<String>) -> Result<String, String> {
    let identity = identity.into();
    if identity.is_empty() || identity.len() > 4096 {
        return Err("ApprovalSubject: identity length must be between 1 and 4096".to_string());
    }
    if identity.chars().any(char::is_control) {
        return Err("ApprovalSubject: identity must not contain control characters".to_string());
    }
    Ok(identity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{DiagnosticKind, ResolutionState};

    fn subject(identity: &str) -> ApprovalSubject {
        ApprovalSubject::new("effective-spec", identity).unwrap()
    }

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

    fn approved_bound_root(identity: &str) -> SubjectBoundReviewSnapshot {
        let mut review = SubjectBoundReviewSession::new("intent-root", subject(identity)).unwrap();
        review.submit("agent").unwrap();
        review.approve("customer").unwrap();
        review.snapshot()
    }

    #[test]
    fn same_transitions_bind_distinct_subjects_to_distinct_approvals() {
        fn build(identity: &str) -> SubjectBoundApprovalSnapshot {
            let mut review = SubjectBoundReviewSession::new("intent-42", subject(identity)).unwrap();
            review.submit("agent").unwrap();
            review.approve("customer").unwrap()
        }

        let a = build("spec:aaa");
        let b = build("spec:bbb");
        assert_eq!(a.approval, b.approval);
        assert_ne!(a.subject, b.subject);
        assert_ne!(a, b);
    }

    #[test]
    fn subject_exists_before_submit_and_has_no_session_mutation_path() {
        let review = SubjectBoundReviewSession::new("intent-subject", subject("spec:stable")).unwrap();
        assert_eq!(review.subject().kind(), "effective-spec");
        assert_eq!(review.subject().identity(), "spec:stable");
        assert!(!review.compile_eligible());
    }

    #[test]
    fn unresolved_approval_failure_is_atomic_and_yields_no_bound_approval() {
        let mut review = SubjectBoundReviewSession::new("intent-blocked", subject("spec:v1")).unwrap();
        review.submit("agent").unwrap();
        review
            .request_diagnostic("validator", normalization_diagnostic())
            .unwrap();
        let before = review.snapshot();

        assert!(review.approve("customer").is_err());
        assert_eq!(review.snapshot(), before);
        assert!(review.snapshot().bound_approval.is_none());
    }

    #[test]
    fn approved_bound_root_seeds_subject_bound_revision_chain() {
        let root = approved_bound_root("spec:root");
        let original = root.clone();
        let chain = SubjectBoundRevisionChain::from_approved_root(root).unwrap();
        assert_eq!(chain.root(), &original);
        assert_eq!(chain.root().subject.identity(), "spec:root");
    }

    #[test]
    fn child_approval_binds_exact_subject_supplied_at_open() {
        let mut chain =
            SubjectBoundRevisionChain::from_approved_root(approved_bound_root("spec:root"))
                .unwrap();
        let child = chain
            .open_revision(None, "child", subject("spec:child"))
            .unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        let approval = chain.approve_revision(&child, "customer").unwrap();

        assert_eq!(approval.subject.identity(), "spec:child");
        assert_eq!(chain.revision_subject(&child).unwrap().identity(), "spec:child");
        assert!(chain.revision_compile_eligible(&child).unwrap());
    }

    #[test]
    fn sibling_subjects_and_approvals_are_independent() {
        let mut chain =
            SubjectBoundRevisionChain::from_approved_root(approved_bound_root("spec:root"))
                .unwrap();
        let left = chain
            .open_revision(None, "left", subject("spec:left"))
            .unwrap();
        let right = chain
            .open_revision(None, "right", subject("spec:right"))
            .unwrap();
        chain.submit_revision(&left, "agent-left").unwrap();
        chain.approve_revision(&left, "customer").unwrap();

        assert!(chain.revision_compile_eligible(&left).unwrap());
        assert!(!chain.revision_compile_eligible(&right).unwrap());
        assert_eq!(chain.revision_subject(&right).unwrap().identity(), "spec:right");
        assert!(chain.bound_revision_approval(&right).is_none());
    }

    #[test]
    fn deeper_revision_requires_bound_parent_and_binds_its_own_subject() {
        let mut chain =
            SubjectBoundRevisionChain::from_approved_root(approved_bound_root("spec:root"))
                .unwrap();
        let parent = chain
            .open_revision(None, "parent", subject("spec:parent"))
            .unwrap();

        assert!(chain
            .open_revision(Some(&parent), "too-early", subject("spec:too-early"))
            .is_err());

        chain.submit_revision(&parent, "agent").unwrap();
        let parent_approval = chain.approve_revision(&parent, "customer").unwrap();
        let child = chain
            .open_revision(Some(&parent), "child", subject("spec:deep-child"))
            .unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        let child_approval = chain.approve_revision(&child, "customer").unwrap();

        assert_eq!(child_approval.subject.identity(), "spec:deep-child");
        assert_eq!(
            child_approval.approval.parent_approval_id,
            parent_approval.approval.revision_approval_id
        );
    }

    #[test]
    fn terminal_child_never_yields_bound_approval_or_corrupts_parent() {
        let root = approved_bound_root("spec:root");
        let original = root.clone();
        let mut chain = SubjectBoundRevisionChain::from_approved_root(root).unwrap();
        let child = chain
            .open_revision(None, "unsupported", subject("spec:unsupported"))
            .unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        chain
            .mark_unsupported(&child, "validator", "capability absent")
            .unwrap();

        assert!(chain.approve_revision(&child, "customer").is_err());
        assert!(chain.bound_revision_approval(&child).is_none());
        assert_eq!(chain.root(), &original);
        assert_eq!(
            chain
                .snapshot()
                .chain
                .revision(&child)
                .unwrap()
                .review
                .workflow
                .state,
            ResolutionState::Unsupported
        );
    }

    #[test]
    fn identical_bound_workflows_are_deterministic() {
        fn build() -> SubjectBoundRevisionChainSnapshot {
            let mut chain = SubjectBoundRevisionChain::from_approved_root(approved_bound_root(
                "spec:root-stable",
            ))
            .unwrap();
            let child = chain
                .open_revision(None, "stable", subject("spec:child-stable"))
                .unwrap();
            chain.submit_revision(&child, "agent").unwrap();
            chain.approve_revision(&child, "customer").unwrap();
            chain.snapshot()
        }

        assert_eq!(build(), build());
    }

    #[test]
    fn subject_validation_rejects_unstable_or_ambiguous_tokens() {
        assert!(ApprovalSubject::new("", "spec:x").is_err());
        assert!(ApprovalSubject::new("bad kind", "spec:x").is_err());
        assert!(ApprovalSubject::new("spec", "").is_err());
        assert!(ApprovalSubject::new("spec", "contains\nnewline").is_err());
    }
}
