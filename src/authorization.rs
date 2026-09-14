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

use crate::resolution_review::APPROVAL_SNAPSHOT_SCHEMA;
use crate::resolution_subject::{
    ApprovalSubject, SubjectBoundApprovalSnapshot, SUBJECT_BOUND_APPROVAL_SCHEMA,
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

    fn matches(&self, approval: &SubjectBoundApprovalSnapshot) -> bool {
        self.delegate == approval.approval.approver
            && self.intent_id == approval.approval.intent_id
            && self.subject == approval.subject
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
        validate_approval_envelope(approval).is_ok() && self.actor_is_currently_authorized(approval)
    }

    pub fn authorize(
        &self,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> Result<AuthorizationSnapshot, String> {
        validate_approval_envelope(approval)?;
        if !self.actor_is_currently_authorized(approval) {
            return Err(
                "AuthorizationPolicy: approval actor is outside owner/delegation scope".to_string(),
            );
        }

        Ok(AuthorizationSnapshot {
            schema: AUTHORIZATION_SNAPSHOT_SCHEMA.to_string(),
            policy_id: self.policy_id.clone(),
            policy_revision: self.revision,
            approval_schema: approval.schema.clone(),
            review_approval_schema: approval.approval.schema.clone(),
            approval_id: approval.approval.approval_id.clone(),
            workflow_revision: approval.approval.workflow_revision,
            intent_id: approval.approval.intent_id.clone(),
            subject: approval.subject.clone(),
            approver: approval.approval.approver.clone(),
        })
    }

    pub fn validate_authorization(
        &self,
        authorization: &AuthorizationSnapshot,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> Result<(), String> {
        validate_approval_envelope(approval)?;

        if authorization.schema != AUTHORIZATION_SNAPSHOT_SCHEMA {
            return Err("AuthorizationPolicy: unsupported authorization snapshot schema".to_string());
        }
        if authorization.policy_id != self.policy_id || authorization.policy_revision != self.revision {
            return Err("AuthorizationPolicy: authorization snapshot is stale for current policy".to_string());
        }
        if authorization.approval_schema != approval.schema
            || authorization.review_approval_schema != approval.approval.schema
            || authorization.approval_id != approval.approval.approval_id
            || authorization.workflow_revision != approval.approval.workflow_revision
            || authorization.intent_id != approval.approval.intent_id
            || authorization.subject != approval.subject
            || authorization.approver != approval.approval.approver
        {
            return Err(
                "AuthorizationPolicy: authorization snapshot does not match exact subject-bound approval"
                    .to_string(),
            );
        }
        if !self.actor_is_currently_authorized(approval) {
            return Err(
                "AuthorizationPolicy: authorization is no longer valid under current owner/delegation scope"
                    .to_string(),
            );
        }

        Ok(())
    }

    fn actor_is_currently_authorized(&self, approval: &SubjectBoundApprovalSnapshot) -> bool {
        approval.approval.approver == self.owner
            || self.grants.iter().any(|grant| grant.matches(approval))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizationSnapshot {
    schema: String,
    policy_id: String,
    policy_revision: u64,
    approval_schema: String,
    review_approval_schema: String,
    approval_id: String,
    workflow_revision: u64,
    intent_id: String,
    subject: ApprovalSubject,
    approver: String,
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
        &self.approval_id
    }

    pub fn workflow_revision(&self) -> u64 {
        self.workflow_revision
    }

    pub fn intent_id(&self) -> &str {
        &self.intent_id
    }

    pub fn subject(&self) -> &ApprovalSubject {
        &self.subject
    }

    pub fn approver(&self) -> &str {
        &self.approver
    }
}

fn validate_approval_envelope(approval: &SubjectBoundApprovalSnapshot) -> Result<(), String> {
    if approval.schema != SUBJECT_BOUND_APPROVAL_SCHEMA {
        return Err("AuthorizationPolicy: unsupported subject-bound approval schema".to_string());
    }
    if approval.approval.schema != APPROVAL_SNAPSHOT_SCHEMA {
        return Err("AuthorizationPolicy: unsupported review approval schema".to_string());
    }
    if approval.approval.approval_id.trim().is_empty() {
        return Err("AuthorizationPolicy: approval id must not be empty".to_string());
    }
    if approval.approval.intent_id.trim().is_empty() {
        return Err("AuthorizationPolicy: approval intent id must not be empty".to_string());
    }
    if approval.approval.approver.trim().is_empty() {
        return Err("AuthorizationPolicy: approval actor must not be empty".to_string());
    }
    let expected_approval_id = format!(
        "{}:approval:r{}",
        approval.approval.intent_id, approval.approval.workflow_revision
    );
    if approval.approval.approval_id != expected_approval_id {
        return Err(
            "AuthorizationPolicy: approval id does not match intent/workflow revision provenance"
                .to_string(),
        );
    }
    Ok(())
}

fn validate_nonempty(value: impl Into<String>, context: &str) -> Result<String, String> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(format!("{context} must not be empty"));
    }
    if value.chars().any(char::is_control) {
        return Err(format!("{context} must not contain control characters"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_subject::SubjectBoundReviewSession;

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
