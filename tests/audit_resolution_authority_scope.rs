mod resolution_subject {
    pub use burn_research::resolution_subject::*;
}

#[path = "../src/effective_spec.rs"]
mod effective_spec;

use burn_research::resolution_subject::{
    ApprovalSubject, SubjectBoundApprovalSnapshot, SubjectBoundReviewSession,
};
use effective_spec::{ApprovedEffectiveSpec, EffectiveSpec, SpecDeclaration};
use serde_json::json;

#[derive(Clone, Debug)]
struct ScopedGrant {
    delegate: String,
    intent_id: String,
    subject_kind: String,
    subject_identity: String,
}

#[derive(Clone, Debug)]
struct ScopedAuthorityPolicy {
    policy_id: String,
    owner: String,
    grants: Vec<ScopedGrant>,
}

#[derive(Clone, Debug)]
struct AuthorizationDecision {
    policy_id: String,
    approval_id: String,
    intent_id: String,
    subject_kind: String,
    subject_identity: String,
    approver: String,
}

impl ScopedAuthorityPolicy {
    fn authorizes(&self, approval: &SubjectBoundApprovalSnapshot) -> bool {
        if approval.approval.approver == self.owner {
            return true;
        }
        self.grants.iter().any(|grant| {
            grant.delegate == approval.approval.approver
                && grant.intent_id == approval.approval.intent_id
                && grant.subject_kind == approval.subject.kind()
                && grant.subject_identity == approval.subject.identity()
        })
    }

    fn authorize(
        &self,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> Result<AuthorizationDecision, String> {
        if !self.authorizes(approval) {
            return Err("authority policy: approval is outside owner/delegation scope".to_string());
        }
        Ok(AuthorizationDecision {
            policy_id: self.policy_id.clone(),
            approval_id: approval.approval.approval_id.clone(),
            intent_id: approval.approval.intent_id.clone(),
            subject_kind: approval.subject.kind().to_string(),
            subject_identity: approval.subject.identity().to_string(),
            approver: approval.approval.approver.clone(),
        })
    }

    fn decision_is_current(
        &self,
        decision: &AuthorizationDecision,
        approval: &SubjectBoundApprovalSnapshot,
    ) -> bool {
        decision.policy_id == self.policy_id
            && decision.approval_id == approval.approval.approval_id
            && decision.intent_id == approval.approval.intent_id
            && decision.subject_kind == approval.subject.kind()
            && decision.subject_identity == approval.subject.identity()
            && decision.approver == approval.approval.approver
            && self.authorizes(approval)
    }
}

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

#[test]
fn audit_scoped_authority_and_stale_approval_behavior() {
    let policy_v1 = ScopedAuthorityPolicy {
        policy_id: "policy:customer:v1".to_string(),
        owner: "customer".to_string(),
        grants: vec![ScopedGrant {
            delegate: "customer-delegate".to_string(),
            intent_id: "intent-a".to_string(),
            subject_kind: "effective-spec".to_string(),
            subject_identity: "spec:a".to_string(),
        }],
    };

    let delegated_a = bound_approval("intent-a", "spec:a", "customer-delegate");
    let delegated_sibling = bound_approval("intent-a", "spec:b", "customer-delegate");
    let delegated_other_intent = bound_approval("intent-b", "spec:a", "customer-delegate");
    let owner_sibling = bound_approval("intent-a", "spec:b", "customer");

    // Exact-scoped delegation: same delegate cannot reuse authority across sibling subjects.
    assert!(policy_v1.authorizes(&delegated_a));
    assert!(!policy_v1.authorizes(&delegated_sibling));

    // Exact-scoped delegation: same subject identity cannot reuse authority across intents.
    assert!(!policy_v1.authorizes(&delegated_other_intent));

    // Owner remains authoritative without a per-subject delegate grant.
    assert!(policy_v1.authorizes(&owner_sibling));

    // Materialize an immutable authorization decision under policy v1.
    let decision_v1 = policy_v1.authorize(&delegated_a).unwrap();
    assert!(policy_v1.decision_is_current(&decision_v1, &delegated_a));
    assert!(!policy_v1.decision_is_current(&decision_v1, &delegated_sibling));

    // Policy revision/revocation makes the old authorization stale.
    let policy_v2 = ScopedAuthorityPolicy {
        policy_id: "policy:customer:v2".to_string(),
        owner: "customer".to_string(),
        grants: Vec::new(),
    };
    assert!(!policy_v2.authorizes(&delegated_a));
    assert!(!policy_v2.decision_is_current(&decision_v1, &delegated_a));

    // Existing subject binding independently makes an approval stale when the spec changes.
    let spec_a = EffectiveSpec::root(vec![
        SpecDeclaration::new("mode", "a").unwrap(),
    ])
    .unwrap();
    let spec_b = EffectiveSpec::root(vec![
        SpecDeclaration::new("mode", "b").unwrap(),
    ])
    .unwrap();

    let mut review_a = SubjectBoundReviewSession::new(
        "intent-spec",
        spec_a.approval_subject().unwrap(),
    )
    .unwrap();
    review_a.submit("agent").unwrap();
    let approval_a = review_a.approve("customer").unwrap();

    assert!(ApprovedEffectiveSpec::bind_root(spec_a, approval_a.clone()).is_ok());
    assert!(ApprovedEffectiveSpec::bind_root(spec_b, approval_a).is_err());

    let report = json!({
        "verdict": "PASS_WITH_FINDINGS",
        "task": "Scoped customer authority / stale approval simulation",
        "productionSemanticsChanged": false,
        "findings": [
            {
                "id": "delegation_exact_subject_scope",
                "score": 0,
                "status": "feasible",
                "evidence": "delegate authority for intent-a/spec:a does not authorize sibling spec:b",
                "recommendation": "BIND_DELEGATION_TO_EXACT_SUBJECT"
            },
            {
                "id": "delegation_exact_intent_scope",
                "score": 0,
                "status": "feasible",
                "evidence": "delegate authority for intent-a/spec:a does not authorize intent-b/spec:a",
                "recommendation": "BIND_DELEGATION_TO_EXACT_INTENT"
            },
            {
                "id": "policy_revision_stales_authorization",
                "score": 0,
                "status": "feasible",
                "evidence": "authorization decision created under policy v1 is rejected after policy changes to v2/revokes the delegate",
                "recommendation": "BIND_EXECUTION_GATE_TO_POLICY_IDENTITY"
            },
            {
                "id": "subject_change_stales_approval",
                "score": 0,
                "status": "already_protected_by_core",
                "evidence": "approval bound to EffectiveSpec A cannot bind a different EffectiveSpec B",
                "recommendation": "KEEP"
            },
            {
                "id": "owner_remains_final_authority",
                "score": 0,
                "status": "natural",
                "evidence": "customer owner can approve an exact subject without a delegation grant while delegate authority remains narrow",
                "recommendation": "KEEP_OWNER_AND_DELEGATE_ROLES_DISTINCT"
            }
        ]
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}