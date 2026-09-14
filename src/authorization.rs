//! Deterministic customer/delegate authorization layered above subject-bound resolution approval.
//!
//! This module is deliberately a policy/provenance boundary only. It does not resolve intent,
//! approve a resolution, perform proof, construct `MathProgram`, or execute code. Authorization
//! starts from an already subject-bound approval and records whether the approval actor is the
//! policy owner or an explicitly scoped delegate for the exact intent + immutable subject.
//!
//! Threat model: actor strings are identities supplied by a trusted caller/identity layer. This
//! module does not authenticate principals or provide cryptographic signatures. It enforces
//! deterministic authorization semantics over those supplied identities and fails closed when
//! approval/schema/scope/policy provenance does not match exactly.

use crate::resolution_review::{ApprovalSnapshot, APPROVAL_SNAPSHOT_SCHEMA};
use crate::resolution_revision::REVISION_APPROVAL_SCHEMA;
use crate::resolution_subject::{
    ApprovalSubject, SubjectBoundApprovalSnapshot, SubjectBoundRevisionApprovalSnapshot,
    SUBJECT_BOUND_APPROVAL_SCHEMA, SUBJECT_BOUND_REVISION_SCHEMA,
};

pub const AUTHORIZATION_SNAPSHOT_SCHEMA: &str = "burn-research.authorization.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScopedAuthorizationGrant {
    delegate: String,
    intent_id: String,
    subject: ApprovalSubject,
}

impl ScopedAuthorizationGrant {
    pub fn new(
        delegate: impl Into<String>,
        intent_id: impl Into<String>,
        subject: ApprovalSubject,
    ) -> Result<Self, String> {
        Ok(Self {
            delegate: validate_nonempty(delegate, "ScopedAuthorizationGrant: delegate")?,
            intent_id: validate_nonempty(intent_id, "ScopedAuthorizationGrant: intent id")?,
            subject,
        })
    }

    pub fn delegate(&self) -> &str {
        &self.delegate
    }

    pub fn intent_id(&self) -> &str {
        &self.intent_id
    }

    pub fn subject(&self) -> &ApprovalSubject {
        &self.subject
    }

    fn canonical_key(&self) -> (&str, &str, &str, &str) {
        (
            &self.delegate,
            &self.intent_id,
            self.subject.kind(),
            self.subject.identity(),
        )
    }

    fn matches(&self, binding: &AuthorizationApprovalBinding) -> bool {
        self.delegate == binding.approver()
            && self.intent_id == binding.intent_id()
            && &self.subject == binding.subject()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationPolicy {
    policy_id: String,
    revision: u64,
    owner: String,
    grants: Vec<ScopedAuthorizationGrant>,
}

impl AuthorizationPolicy {
    pub fn new(
        policy_id: impl Into<String>,
        revision: u64,
        owner: impl Into<String>,
        mut grants: Vec<ScopedAuthorizationGrant>,
    ) -> Result<Self, String> {
        let policy_id = validate_nonempty(policy_id, "AuthorizationPolicy: policy id")?;
        let owner = validate_nonempty(owner, "AuthorizationPolicy: owner")?;

        grants.sort_by(|left, right| left.canonical_key().cmp(&right.canonical_key()));
        if grants.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err("AuthorizationPolicy: duplicate scoped authorization grant".to_string());
        }

        Ok(Self {
            policy_id,
            revision,
            owner,
            grants,
        })
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn grants(&self) -> &[ScopedAuthorizationGrant] {
        &self.grants
    }

    pub fn is_authorized(&self, approval: &SubjectBoundApprovalSnapshot) -> bool {
        root_binding(approval)
            .is_ok_and(|binding| self.actor_is_currently_authorized(&binding))
    }

    pub fn is_revision_authorized(&self, approval: &SubjectBoundRevisionApprovalSnapshot) -> bool {
        revision_binding(approval)
            .is_ok_and(|binding| self.actor_is_currently_authorized(&binding))
    }

    pub fn authorize(
        &self,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> Result<AuthorizationSnapshot, String> {
        self.authorize_binding(root_binding(approval)?)
    }

    pub fn authorize_revision(
        &self,
        approval: &SubjectBoundRevisionApprovalSnapshot,
    ) -> Result<AuthorizationSnapshot, String> {
        self.authorize_binding(revision_binding(approval)?)
    }

    pub fn validate_authorization(
        &self,
        authorization: &AuthorizationSnapshot,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> Result<(), String> {
        self.validate_binding(authorization, root_binding(approval)?)
    }

    pub fn validate_revision_authorization(
        &self,
        authorization: &AuthorizationSnapshot,
        approval: &SubjectBoundRevisionApprovalSnapshot,
    ) -> Result<(), String> {
        self.validate_binding(authorization, revision_binding(approval)?)
    }

    fn authorize_binding(
        &self,
        binding: AuthorizationApprovalBinding,
    ) -> Result<AuthorizationSnapshot, String> {
        if !self.actor_is_currently_authorized(&binding) {
            return Err(
                "AuthorizationPolicy: approval actor is outside owner/delegation scope".to_string(),
            );
        }

        Ok(AuthorizationSnapshot {
            schema: AUTHORIZATION_SNAPSHOT_SCHEMA.to_string(),
            policy_id: self.policy_id.clone(),
            policy_revision: self.revision,
            binding,
        })
    }

    fn validate_binding(
        &self,
        authorization: &AuthorizationSnapshot,
        binding: AuthorizationApprovalBinding,
    ) -> Result<(), String> {
        if authorization.schema != AUTHORIZATION_SNAPSHOT_SCHEMA {
            return Err("AuthorizationPolicy: unsupported authorization snapshot schema".to_string());
        }
        if authorization.policy_id != self.policy_id
            || authorization.policy_revision != self.revision
        {
            return Err("AuthorizationPolicy: authorization snapshot is stale for current policy".to_string());
        }
        if authorization.binding != binding {
            return Err(
                "AuthorizationPolicy: authorization snapshot does not match exact subject-bound approval"
                    .to_string(),
            );
        }
        if !self.actor_is_currently_authorized(&binding) {
            return Err(
                "AuthorizationPolicy: authorization is no longer valid under current owner/delegation scope"
                    .to_string(),
            );
        }

        Ok(())
    }

    fn actor_is_currently_authorized(&self, binding: &AuthorizationApprovalBinding) -> bool {
        binding.approver() == self.owner
            || self.grants.iter().any(|grant| grant.matches(binding))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum AuthorizationApprovalBinding {
    Root {
        subject_bound_schema: String,
        review_approval_schema: String,
        approval_id: String,
        workflow_revision: u64,
        intent_id: String,
        subject: ApprovalSubject,
        approver: String,
    },
    Revision {
        subject_bound_schema: String,
        revision_approval_schema: String,
        revision_approval_id: String,
        lineage_id: String,
        revision_id: String,
        parent_approval_id: String,
        supersedes_approval_id: String,
        review_approval_schema: String,
        review_approval_id: String,
        workflow_revision: u64,
        intent_id: String,
        subject: ApprovalSubject,
        approver: String,
    },
}

impl AuthorizationApprovalBinding {
    fn approval_id(&self) -> &str {
        match self {
            Self::Root { approval_id, .. } => approval_id,
            Self::Revision {
                revision_approval_id,
                ..
            } => revision_approval_id,
        }
    }

    fn workflow_revision(&self) -> u64 {
        match self {
            Self::Root {
                workflow_revision, ..
            }
            | Self::Revision {
                workflow_revision, ..
            } => *workflow_revision,
        }
    }

    fn intent_id(&self) -> &str {
        match self {
            Self::Root { intent_id, .. } | Self::Revision { intent_id, .. } => intent_id,
        }
    }

    fn subject(&self) -> &ApprovalSubject {
        match self {
            Self::Root { subject, .. } | Self::Revision { subject, .. } => subject,
        }
    }

    fn approver(&self) -> &str {
        match self {
            Self::Root { approver, .. } | Self::Revision { approver, .. } => approver,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationSnapshot {
    schema: String,
    policy_id: String,
    policy_revision: u64,
    binding: AuthorizationApprovalBinding,
}

impl AuthorizationSnapshot {
    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub fn policy_revision(&self) -> u64 {
        self.policy_revision
    }

    pub fn approval_id(&self) -> &str {
        self.binding.approval_id()
    }

    pub fn workflow_revision(&self) -> u64 {
        self.binding.workflow_revision()
    }

    pub fn intent_id(&self) -> &str {
        self.binding.intent_id()
    }

    pub fn subject(&self) -> &ApprovalSubject {
        self.binding.subject()
    }

    pub fn approver(&self) -> &str {
        self.binding.approver()
    }

    pub fn is_revision(&self) -> bool {
        matches!(self.binding, AuthorizationApprovalBinding::Revision { .. })
    }
}

fn root_binding(
    approval: &SubjectBoundApprovalSnapshot,
) -> Result<AuthorizationApprovalBinding, String> {
    if approval.schema != SUBJECT_BOUND_APPROVAL_SCHEMA {
        return Err("AuthorizationPolicy: unsupported subject-bound approval schema".to_string());
    }
    validate_review_approval(&approval.approval)?;

    Ok(AuthorizationApprovalBinding::Root {
        subject_bound_schema: approval.schema.clone(),
        review_approval_schema: approval.approval.schema.clone(),
        approval_id: approval.approval.approval_id.clone(),
        workflow_revision: approval.approval.workflow_revision,
        intent_id: approval.approval.intent_id.clone(),
        subject: approval.subject.clone(),
        approver: approval.approval.approver.clone(),
    })
}

fn revision_binding(
    approval: &SubjectBoundRevisionApprovalSnapshot,
) -> Result<AuthorizationApprovalBinding, String> {
    if approval.schema != SUBJECT_BOUND_REVISION_SCHEMA {
        return Err(
            "AuthorizationPolicy: unsupported subject-bound revision approval schema".to_string(),
        );
    }
    if approval.approval.schema != REVISION_APPROVAL_SCHEMA {
        return Err("AuthorizationPolicy: unsupported revision approval schema".to_string());
    }

    validate_present(
        &approval.approval.revision_approval_id,
        "AuthorizationPolicy: revision approval id",
    )?;
    validate_present(
        &approval.approval.lineage_id,
        "AuthorizationPolicy: lineage id",
    )?;
    validate_present(
        &approval.approval.revision_id,
        "AuthorizationPolicy: revision id",
    )?;
    validate_present(
        &approval.approval.parent_approval_id,
        "AuthorizationPolicy: parent approval id",
    )?;
    validate_present(
        &approval.approval.supersedes_approval_id,
        "AuthorizationPolicy: supersedes approval id",
    )?;
    validate_review_approval(&approval.approval.approval)?;

    if approval.approval.parent_approval_id != approval.approval.supersedes_approval_id {
        return Err(
            "AuthorizationPolicy: revision parent/supersedes approval provenance mismatch"
                .to_string(),
        );
    }

    let expected_revision_approval_id = format!(
        "{}:approval:r{}",
        approval.approval.revision_id, approval.approval.approval.workflow_revision
    );
    if approval.approval.revision_approval_id != expected_revision_approval_id {
        return Err(
            "AuthorizationPolicy: revision approval id does not match revision/workflow provenance"
                .to_string(),
        );
    }

    let expected_revision_prefix = format!("{}:revision:", approval.approval.lineage_id);
    if !approval
        .approval
        .revision_id
        .starts_with(&expected_revision_prefix)
    {
        return Err(
            "AuthorizationPolicy: revision id does not belong to declared lineage".to_string(),
        );
    }

    let expected_intent_suffix = format!("::{}", approval.approval.revision_id);
    if !approval
        .approval
        .approval
        .intent_id
        .ends_with(&expected_intent_suffix)
    {
        return Err(
            "AuthorizationPolicy: revision review intent does not bind the declared revision id"
                .to_string(),
        );
    }

    Ok(AuthorizationApprovalBinding::Revision {
        subject_bound_schema: approval.schema.clone(),
        revision_approval_schema: approval.approval.schema.clone(),
        revision_approval_id: approval.approval.revision_approval_id.clone(),
        lineage_id: approval.approval.lineage_id.clone(),
        revision_id: approval.approval.revision_id.clone(),
        parent_approval_id: approval.approval.parent_approval_id.clone(),
        supersedes_approval_id: approval.approval.supersedes_approval_id.clone(),
        review_approval_schema: approval.approval.approval.schema.clone(),
        review_approval_id: approval.approval.approval.approval_id.clone(),
        workflow_revision: approval.approval.approval.workflow_revision,
        intent_id: approval.approval.approval.intent_id.clone(),
        subject: approval.subject.clone(),
        approver: approval.approval.approval.approver.clone(),
    })
}

fn validate_review_approval(approval: &ApprovalSnapshot) -> Result<(), String> {
    if approval.schema != APPROVAL_SNAPSHOT_SCHEMA {
        return Err("AuthorizationPolicy: unsupported review approval schema".to_string());
    }
    validate_present(&approval.approval_id, "AuthorizationPolicy: review approval id")?;
    validate_present(&approval.intent_id, "AuthorizationPolicy: approval intent id")?;
    validate_present(&approval.approver, "AuthorizationPolicy: approval actor")?;

    let expected_approval_id = format!(
        "{}:approval:r{}",
        approval.intent_id, approval.workflow_revision
    );
    if approval.approval_id != expected_approval_id {
        return Err(
            "AuthorizationPolicy: review approval id does not match intent/workflow provenance"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_present(value: &str, context: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{context} must not be empty"));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{context} must not contain control characters"));
    }
    Ok(())
}

fn validate_nonempty(value: impl Into<String>, context: &str) -> Result<String, String> {
    let value = value.into();
    validate_present(&value, context)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_subject::{SubjectBoundReviewSession, SubjectBoundRevisionChain};

    fn subject(identity: &str) -> ApprovalSubject {
        ApprovalSubject::new("effective-spec", identity).unwrap()
    }

    fn bound_approval(
        intent_id: &str,
        subject_identity: &str,
        approver: &str,
    ) -> SubjectBoundApprovalSnapshot {
        let mut review =
            SubjectBoundReviewSession::new(intent_id, subject(subject_identity)).unwrap();
        review.submit("agent").unwrap();
        review.approve(approver).unwrap()
    }

    fn bound_revision_approval(approver: &str) -> SubjectBoundRevisionApprovalSnapshot {
        let mut root = SubjectBoundReviewSession::new("intent-root", subject("spec:root")).unwrap();
        root.submit("agent").unwrap();
        root.approve("customer").unwrap();

        let mut chain = SubjectBoundRevisionChain::from_approved_root(root.snapshot()).unwrap();
        let revision_id = chain
            .open_revision(None, "child", subject("spec:child"))
            .unwrap();
        chain.submit_revision(&revision_id, "agent").unwrap();
        chain.approve_revision(&revision_id, approver).unwrap()
    }

    fn grant(delegate: &str, intent_id: &str, subject_identity: &str) -> ScopedAuthorizationGrant {
        ScopedAuthorizationGrant::new(delegate, intent_id, subject(subject_identity)).unwrap()
    }

    #[test]
    fn owner_authorizes_exact_subject_bound_approval() {
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let approval = bound_approval("intent-a", "spec:a", "customer");

        let authorization = policy.authorize(&approval).unwrap();
        assert_eq!(authorization.schema(), AUTHORIZATION_SNAPSHOT_SCHEMA);
        assert_eq!(authorization.policy_id(), "policy-a");
        assert_eq!(authorization.policy_revision(), 1);
        assert_eq!(authorization.intent_id(), "intent-a");
        assert_eq!(authorization.subject(), &subject("spec:a"));
        assert_eq!(authorization.approver(), "customer");
        assert!(!authorization.is_revision());
        assert!(policy.validate_authorization(&authorization, &approval).is_ok());
    }

    #[test]
    fn delegate_authority_is_exact_intent_and_subject_scoped() {
        let policy = AuthorizationPolicy::new(
            "policy-a",
            1,
            "customer",
            vec![grant("delegate", "intent-a", "spec:a")],
        )
        .unwrap();

        let exact = bound_approval("intent-a", "spec:a", "delegate");
        let sibling = bound_approval("intent-a", "spec:b", "delegate");
        let other_intent = bound_approval("intent-b", "spec:a", "delegate");

        assert!(policy.authorize(&exact).is_ok());
        assert!(policy.authorize(&sibling).is_err());
        assert!(policy.authorize(&other_intent).is_err());
    }

    #[test]
    fn unrelated_agent_cannot_self_authorize() {
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let approval = bound_approval("intent-a", "spec:a", "agent");

        assert!(!policy.is_authorized(&approval));
        assert!(policy.authorize(&approval).is_err());
    }

    #[test]
    fn revision_delegate_authority_preserves_exact_lineage_and_subject_provenance() {
        let approval = bound_revision_approval("delegate");
        let grant = ScopedAuthorizationGrant::new(
            "delegate",
            approval.approval.approval.intent_id.clone(),
            approval.subject.clone(),
        )
        .unwrap();
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![grant]).unwrap();

        let authorization = policy.authorize_revision(&approval).unwrap();
        assert!(authorization.is_revision());
        assert_eq!(authorization.approval_id(), approval.approval.revision_approval_id);
        assert_eq!(authorization.intent_id(), approval.approval.approval.intent_id);
        assert_eq!(authorization.subject(), &approval.subject);
        assert!(policy
            .validate_revision_authorization(&authorization, &approval)
            .is_ok());
    }

    #[test]
    fn root_authorization_cannot_be_reused_for_revision_approval() {
        let root = bound_approval("intent-a", "spec:a", "customer");
        let revision = bound_revision_approval("customer");
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let root_authorization = policy.authorize(&root).unwrap();

        assert!(policy
            .validate_revision_authorization(&root_authorization, &revision)
            .is_err());
    }

    #[test]
    fn malformed_revision_lineage_fails_closed() {
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let mut approval = bound_revision_approval("customer");
        approval.approval.parent_approval_id = "wrong-parent".to_string();

        assert!(policy.authorize_revision(&approval).is_err());
    }

    #[test]
    fn authorization_cannot_be_reused_for_another_approval_or_subject() {
        let policy = AuthorizationPolicy::new(
            "policy-a",
            1,
            "customer",
            vec![grant("delegate", "intent-a", "spec:a")],
        )
        .unwrap();
        let exact = bound_approval("intent-a", "spec:a", "delegate");
        let sibling_owner = bound_approval("intent-a", "spec:b", "customer");
        let authorization = policy.authorize(&exact).unwrap();

        assert!(policy.validate_authorization(&authorization, &sibling_owner).is_err());
    }

    #[test]
    fn policy_revision_and_revocation_stale_previous_authorization() {
        let scoped = grant("delegate", "intent-a", "spec:a");
        let policy_v1 =
            AuthorizationPolicy::new("policy-a", 1, "customer", vec![scoped.clone()]).unwrap();
        let approval = bound_approval("intent-a", "spec:a", "delegate");
        let authorization = policy_v1.authorize(&approval).unwrap();

        let policy_v2 =
            AuthorizationPolicy::new("policy-a", 2, "customer", vec![scoped]).unwrap();
        assert!(policy_v2
            .validate_authorization(&authorization, &approval)
            .is_err());

        let revoked_same_revision =
            AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        assert!(revoked_same_revision
            .validate_authorization(&authorization, &approval)
            .is_err());
    }

    #[test]
    fn identical_inputs_produce_identical_authorization_snapshots() {
        let policy = AuthorizationPolicy::new(
            "policy-a",
            7,
            "customer",
            vec![grant("delegate", "intent-a", "spec:a")],
        )
        .unwrap();
        let approval = bound_approval("intent-a", "spec:a", "delegate");

        assert_eq!(policy.authorize(&approval).unwrap(), policy.authorize(&approval).unwrap());
    }

    #[test]
    fn malformed_approval_schema_fails_closed_without_policy_mutation() {
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let before = policy.clone();
        let mut approval = bound_approval("intent-a", "spec:a", "customer");
        approval.schema = "wrong.subject.approval.schema".to_string();

        assert!(policy.authorize(&approval).is_err());
        assert_eq!(policy, before);
    }

    #[test]
    fn inconsistent_approval_id_fails_closed_without_policy_mutation() {
        let policy = AuthorizationPolicy::new("policy-a", 1, "customer", vec![]).unwrap();
        let before = policy.clone();
        let mut approval = bound_approval("intent-a", "spec:a", "customer");
        approval.approval.approval_id = "intent-a:approval:r999".to_string();

        assert!(policy.authorize(&approval).is_err());
        assert_eq!(policy, before);
    }

    #[test]
    fn grant_order_is_canonical_and_exact_duplicates_are_rejected() {
        let a = grant("delegate-a", "intent-a", "spec:a");
        let b = grant("delegate-b", "intent-b", "spec:b");

        let left = AuthorizationPolicy::new(
            "policy-a",
            1,
            "customer",
            vec![b.clone(), a.clone()],
        )
        .unwrap();
        let right = AuthorizationPolicy::new(
            "policy-a",
            1,
            "customer",
            vec![a.clone(), b.clone()],
        )
        .unwrap();
        assert_eq!(left, right);

        assert!(AuthorizationPolicy::new("policy-a", 1, "customer", vec![a.clone(), a]).is_err());
    }
}
