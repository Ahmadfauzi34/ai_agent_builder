//! Immutable revision lineage layered above Resolution Review Protocol v2.
//!
//! Approved review history is never edited. A change request opens a fresh
//! review session whose lineage points at an already-approved parent. This
//! keeps the protocol flexible without weakening the compiler-style proof
//! boundary established by resolution v1/v2.

use crate::resolution::{ResolutionDiagnostic, ResolutionResponse, ResolutionState};
use crate::resolution_review::{
    ApprovalSnapshot, ResolutionReviewSession, ResolutionReviewSnapshot,
};

pub const REVISION_LINEAGE_SCHEMA: &str = "burn-research.resolution-revision-lineage.v1";
pub const REVISION_APPROVAL_SCHEMA: &str = "burn-research.resolution-revision-approval.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevisionApprovalSnapshot {
    pub schema: String,
    pub revision_approval_id: String,
    pub lineage_id: String,
    pub revision_id: String,
    pub parent_approval_id: String,
    pub supersedes_approval_id: String,
    pub approval: ApprovalSnapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionRevisionSnapshot {
    pub revision_id: String,
    pub revision_key: String,
    pub parent_revision_id: Option<String>,
    pub parent_approval_id: String,
    pub review: ResolutionReviewSnapshot,
    pub approval: Option<RevisionApprovalSnapshot>,
}

impl ResolutionRevisionSnapshot {
    pub fn compile_eligible(&self) -> bool {
        self.review.compile_eligible() && self.approval.is_some()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionRevisionChainSnapshot {
    pub schema: String,
    pub lineage_id: String,
    pub root: ResolutionReviewSnapshot,
    pub revisions: Vec<ResolutionRevisionSnapshot>,
}

impl ResolutionRevisionChainSnapshot {
    pub fn revision(&self, revision_id: &str) -> Option<&ResolutionRevisionSnapshot> {
        self.revisions
            .iter()
            .find(|revision| revision.revision_id == revision_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResolutionRevision {
    revision_id: String,
    revision_key: String,
    parent_revision_id: Option<String>,
    parent_approval_id: String,
    review: ResolutionReviewSession,
    approval: Option<RevisionApprovalSnapshot>,
}

impl ResolutionRevision {
    fn snapshot(&self) -> ResolutionRevisionSnapshot {
        ResolutionRevisionSnapshot {
            revision_id: self.revision_id.clone(),
            revision_key: self.revision_key.clone(),
            parent_revision_id: self.parent_revision_id.clone(),
            parent_approval_id: self.parent_approval_id.clone(),
            review: self.review.snapshot(),
            approval: self.approval.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolutionRevisionChain {
    lineage_id: String,
    root: ResolutionReviewSnapshot,
    revisions: Vec<ResolutionRevision>,
}

impl ResolutionRevisionChain {
    pub fn from_approved(root: ResolutionReviewSnapshot) -> Result<Self, String> {
        if !root.compile_eligible() {
            return Err(
                "ResolutionRevisionChain: root review must already be explicitly approved"
                    .to_string(),
            );
        }
        let approval = root.approval.as_ref().ok_or_else(|| {
            "ResolutionRevisionChain: approved root is missing approval snapshot".to_string()
        })?;
        if approval.intent_id != root.workflow.intent_id {
            return Err(
                "ResolutionRevisionChain: root approval intent does not match workflow intent"
                    .to_string(),
            );
        }

        let lineage_id = format!(
            "{}:lineage:{}",
            root.workflow.intent_id, approval.approval_id
        );
        Ok(Self {
            lineage_id,
            root,
            revisions: Vec::new(),
        })
    }

    pub fn lineage_id(&self) -> &str {
        &self.lineage_id
    }

    pub fn root_snapshot(&self) -> &ResolutionReviewSnapshot {
        &self.root
    }

    pub fn snapshot(&self) -> ResolutionRevisionChainSnapshot {
        ResolutionRevisionChainSnapshot {
            schema: REVISION_LINEAGE_SCHEMA.to_string(),
            lineage_id: self.lineage_id.clone(),
            root: self.root.clone(),
            revisions: self
                .revisions
                .iter()
                .map(ResolutionRevision::snapshot)
                .collect(),
        }
    }

    pub fn open_revision(
        &mut self,
        parent_revision_id: Option<&str>,
        revision_key: impl Into<String>,
    ) -> Result<String, String> {
        let revision_key = validate_revision_key(revision_key)?;
        let (parent_revision_owned, parent_approval_id, parent_token) =
            self.resolve_parent(parent_revision_id)?;

        let revision_id = format!(
            "{}:revision:{}:from:{}",
            self.lineage_id, revision_key, parent_token
        );
        if self
            .revisions
            .iter()
            .any(|revision| revision.revision_id == revision_id)
        {
            return Err(format!(
                "ResolutionRevisionChain: revision already exists: {revision_id}"
            ));
        }

        let child_intent_id = format!("{}::{}", self.root.workflow.intent_id, revision_id);
        let review = ResolutionReviewSession::new(child_intent_id)?;
        self.revisions.push(ResolutionRevision {
            revision_id: revision_id.clone(),
            revision_key,
            parent_revision_id: parent_revision_owned,
            parent_approval_id,
            review,
            approval: None,
        });
        Ok(revision_id)
    }

    pub fn revision_snapshot(
        &self,
        revision_id: &str,
    ) -> Result<ResolutionRevisionSnapshot, String> {
        Ok(self.revision(revision_id)?.snapshot())
    }

    pub fn revision_compile_eligible(&self, revision_id: &str) -> Result<bool, String> {
        let revision = self.revision(revision_id)?;
        Ok(revision.review.compile_eligible() && revision.approval.is_some())
    }

    pub fn submit_revision(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?.review.submit(actor)
    }

    pub fn request_diagnostic(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        diagnostic: ResolutionDiagnostic,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?
            .review
            .request_diagnostic(actor, diagnostic)
    }

    pub fn respond(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        response: ResolutionResponse,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?
            .review
            .respond(actor, response)
    }

    pub fn approve_revision(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
    ) -> Result<RevisionApprovalSnapshot, String> {
        let lineage_id = self.lineage_id.clone();
        let revision = self.revision_mut(revision_id)?;
        if revision.approval.is_some() {
            return Err(format!(
                "ResolutionRevisionChain: revision already approved: {revision_id}"
            ));
        }

        let parent_approval_id = revision.parent_approval_id.clone();
        let stable_revision_id = revision.revision_id.clone();
        let approval = revision.review.approve(actor)?;
        let revision_approval_id = format!(
            "{}:approval:r{}",
            stable_revision_id, approval.workflow_revision
        );
        let snapshot = RevisionApprovalSnapshot {
            schema: REVISION_APPROVAL_SCHEMA.to_string(),
            revision_approval_id,
            lineage_id,
            revision_id: stable_revision_id,
            parent_approval_id: parent_approval_id.clone(),
            supersedes_approval_id: parent_approval_id,
            approval,
        };
        revision.approval = Some(snapshot.clone());
        Ok(snapshot)
    }

    pub fn mark_contradiction(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?
            .review
            .mark_contradiction(actor, reason)
    }

    pub fn mark_unsupported(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?
            .review
            .mark_unsupported(actor, reason)
    }

    pub fn mark_invalid(
        &mut self,
        revision_id: &str,
        actor: impl Into<String>,
        reason: impl Into<String>,
    ) -> Result<(), String> {
        self.revision_mut(revision_id)?
            .review
            .mark_invalid(actor, reason)
    }

    fn resolve_parent(
        &self,
        parent_revision_id: Option<&str>,
    ) -> Result<(Option<String>, String, String), String> {
        match parent_revision_id {
            None => {
                let approval = self.root.approval.as_ref().ok_or_else(|| {
                    "ResolutionRevisionChain: root approval disappeared".to_string()
                })?;
                Ok((None, approval.approval_id.clone(), "root".to_string()))
            }
            Some(parent_revision_id) => {
                let parent = self.revision(parent_revision_id)?;
                let approval = parent.approval.as_ref().ok_or_else(|| {
                    format!(
                        "ResolutionRevisionChain: parent revision is not approved: {parent_revision_id}"
                    )
                })?;
                Ok((
                    Some(parent_revision_id.to_string()),
                    approval.revision_approval_id.clone(),
                    parent_revision_id.to_string(),
                ))
            }
        }
    }

    fn revision(&self, revision_id: &str) -> Result<&ResolutionRevision, String> {
        self.revisions
            .iter()
            .find(|revision| revision.revision_id == revision_id)
            .ok_or_else(|| format!("ResolutionRevisionChain: unknown revision: {revision_id}"))
    }

    fn revision_mut(&mut self, revision_id: &str) -> Result<&mut ResolutionRevision, String> {
        self.revisions
            .iter_mut()
            .find(|revision| revision.revision_id == revision_id)
            .ok_or_else(|| format!("ResolutionRevisionChain: unknown revision: {revision_id}"))
    }
}

pub(crate) fn validate_revision_key(revision_key: impl Into<String>) -> Result<String, String> {
    let revision_key = revision_key.into();
    if revision_key.is_empty() || revision_key.len() > 64 {
        return Err(
            "ResolutionRevisionChain: revision key length must be between 1 and 64".to_string(),
        );
    }
    if !revision_key
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(
            "ResolutionRevisionChain: revision key must use only ASCII letters, digits, '-', '_' or '.'"
                .to_string(),
        );
    }
    Ok(revision_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::DiagnosticKind;

    fn approved_root() -> ResolutionReviewSnapshot {
        let mut review = ResolutionReviewSession::new("intent-root").unwrap();
        review.submit("agent").unwrap();
        review.approve("customer").unwrap();
        review.snapshot()
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

    #[test]
    fn opening_child_preserves_parent_and_starts_unapproved() {
        let root = approved_root();
        let original = root.clone();
        let mut chain = ResolutionRevisionChain::from_approved(root).unwrap();
        let child = chain.open_revision(None, "normalize-v2").unwrap();

        assert_eq!(chain.root_snapshot(), &original);
        assert!(!chain.revision_compile_eligible(&child).unwrap());
        assert_eq!(
            chain.revision_snapshot(&child).unwrap().review.workflow.state,
            ResolutionState::Draft
        );
    }

    #[test]
    fn child_approval_records_parent_and_supersession_without_rewriting_root() {
        let root = approved_root();
        let root_approval = root.approval.as_ref().unwrap().approval_id.clone();
        let original = root.clone();
        let mut chain = ResolutionRevisionChain::from_approved(root).unwrap();
        let child = chain.open_revision(None, "candidate-a").unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        let approval = chain.approve_revision(&child, "customer").unwrap();

        assert_eq!(approval.parent_approval_id, root_approval);
        assert_eq!(approval.supersedes_approval_id, root_approval);
        assert_eq!(approval.revision_id, child);
        assert!(chain.revision_compile_eligible(&child).unwrap());
        assert_eq!(chain.root_snapshot(), &original);
        assert!(chain.root_snapshot().compile_eligible());
    }

    #[test]
    fn sibling_branches_are_independent() {
        let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
        let left = chain.open_revision(None, "left").unwrap();
        let right = chain.open_revision(None, "right").unwrap();

        chain.submit_revision(&left, "agent-left").unwrap();
        chain.approve_revision(&left, "customer").unwrap();

        assert!(chain.revision_compile_eligible(&left).unwrap());
        assert!(!chain.revision_compile_eligible(&right).unwrap());
        assert_eq!(
            chain.revision_snapshot(&right).unwrap().review.workflow.state,
            ResolutionState::Draft
        );
    }

    #[test]
    fn failed_child_response_is_atomic_for_parent_and_sibling_history() {
        let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
        let left = chain.open_revision(None, "left").unwrap();
        let right = chain.open_revision(None, "right").unwrap();
        chain.submit_revision(&left, "agent").unwrap();
        chain
            .request_diagnostic(&left, "validator", normalization_diagnostic())
            .unwrap();
        let before = chain.snapshot();

        assert!(chain
            .respond(
                &left,
                "customer",
                ResolutionResponse::new("E-MATH-001", "zscore").unwrap(),
            )
            .is_err());
        assert_eq!(chain.snapshot(), before);
        assert_eq!(
            chain.revision_snapshot(&right).unwrap().review.workflow.state,
            ResolutionState::Draft
        );
    }

    #[test]
    fn lineage_and_revision_ids_are_deterministic() {
        fn build() -> (String, String, RevisionApprovalSnapshot) {
            let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
            let lineage = chain.lineage_id().to_string();
            let child = chain.open_revision(None, "stable").unwrap();
            chain.submit_revision(&child, "agent").unwrap();
            let approval = chain.approve_revision(&child, "customer").unwrap();
            (lineage, child, approval)
        }

        assert_eq!(build(), build());
    }

    #[test]
    fn terminal_child_does_not_corrupt_approved_parent() {
        let root = approved_root();
        let original = root.clone();
        let mut chain = ResolutionRevisionChain::from_approved(root).unwrap();
        let child = chain.open_revision(None, "unsupported-path").unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        chain
            .mark_unsupported(&child, "validator", "capability absent")
            .unwrap();

        let child_snapshot = chain.revision_snapshot(&child).unwrap();
        assert_eq!(child_snapshot.review.workflow.state, ResolutionState::Unsupported);
        assert!(!child_snapshot.compile_eligible());
        assert_eq!(chain.root_snapshot(), &original);
        assert!(chain.root_snapshot().compile_eligible());
    }

    #[test]
    fn approved_child_can_become_parent_of_a_deeper_revision() {
        let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
        let child = chain.open_revision(None, "child").unwrap();
        chain.submit_revision(&child, "agent").unwrap();
        let child_approval = chain.approve_revision(&child, "customer").unwrap();

        let grandchild = chain.open_revision(Some(&child), "grandchild").unwrap();
        let grandchild_snapshot = chain.revision_snapshot(&grandchild).unwrap();
        assert_eq!(grandchild_snapshot.parent_revision_id.as_deref(), Some(child.as_str()));
        assert_eq!(
            grandchild_snapshot.parent_approval_id,
            child_approval.revision_approval_id
        );
        assert!(!grandchild_snapshot.compile_eligible());
    }

    #[test]
    fn unapproved_parent_cannot_spawn_descendant() {
        let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
        let child = chain.open_revision(None, "child").unwrap();
        let before = chain.snapshot();

        assert!(chain.open_revision(Some(&child), "grandchild").is_err());
        assert_eq!(chain.snapshot(), before);
    }

    #[test]
    fn invalid_or_duplicate_revision_keys_reject_without_mutation() {
        let mut chain = ResolutionRevisionChain::from_approved(approved_root()).unwrap();
        let before = chain.snapshot();
        assert!(chain.open_revision(None, "bad key").is_err());
        assert_eq!(chain.snapshot(), before);

        let child = chain.open_revision(None, "stable").unwrap();
        let after = chain.snapshot();
        assert!(chain.open_revision(None, "stable").is_err());
        assert_eq!(chain.snapshot(), after);
        assert!(!chain.revision_compile_eligible(&child).unwrap());
    }
}
